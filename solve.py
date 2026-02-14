"""
Solve the Jane Street "Dropped a Neural Net" puzzle.

The puzzle provides 97 disassembled linear layers from a residual neural network
and 10,000 rows of historical data (48-dim inputs, model predictions, true targets).
The goal is to reconstruct the correct architecture and layer ordering.

Architecture:
    48 residual Blocks (each with inp: Linear(48,96) and out: Linear(96,48))
    followed by a LastLayer: Linear(48,1)

Solution approach:

1. CLASSIFY PIECES BY SHAPE
   - 48 pieces with weight shape [96, 48] -> Block.inp layers (Linear(48, 96))
   - 48 pieces with weight shape [48, 96] -> Block.out layers (Linear(96, 48))
   - 1 piece with weight shape [1, 48]   -> LastLayer (piece 85)

2. PAIR INP/OUT LAYERS INTO BLOCKS
   For a correctly paired block, inp ≈ out.T in some sense, which means the
   product out @ inp (a 48x48 matrix) should be approximately symmetric.

   Metric: ||M - M.T||_F / ||M||_F where M = out_weight @ inp_weight

   Use the Hungarian algorithm to find the optimal 1-to-1 assignment that
   minimizes total asymmetry. This produces 48 paired blocks with high
   confidence (47/48 agree with greedy best-match, the 1 conflict is resolved
   correctly by Hungarian).

3. ORDER THE 48 BLOCKS
   Initial ordering: sort blocks by |f(x)| (magnitude of the residual addition
   out(relu(inp(x)))) computed on raw inputs. Blocks that make smaller
   adjustments tend to be earlier in the network.

   This gives a surprisingly good initial ordering (corr=0.972 with predictions).

   Refinement: greedy adjacent swaps (bubble sort). Repeatedly sweep through
   the ordering and swap adjacent blocks if it reduces MSE vs known predictions.
   Converges to MSE=0 (exact match) in 14 passes / 107 total swaps.

4. RESULT
   The final permutation perfectly reproduces the model's predictions
   (correlation = 1.0, MAE = 0.0).
"""

import torch
import torch.nn as nn
import json
import numpy as np
from scipy.optimize import linear_sum_assignment


# --- Model architecture ---

class Block(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.out = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        residual = x
        x = self.inp(x)
        x = self.activation(x)
        x = self.out(x)
        return residual + x


class LastLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.layer = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.layer(x)


# --- Load data and pieces ---

print("Loading data and pieces...")
data = torch.load("historical_data.pt", map_location="cpu", weights_only=True)
inputs = data["inputs"]    # [10000, 48]
preds = data["preds"]      # [10000]
targets = data["targets"]  # [10000]

pieces = {}
for i in range(97):
    pieces[i] = torch.load(
        f"historical_data_and_pieces/pieces/piece_{i}.pth",
        map_location="cpu", weights_only=True,
    )


# --- Step 1: Classify pieces by shape ---

inp_ids = sorted([i for i in range(97) if pieces[i]["weight"].shape == torch.Size([96, 48])])
out_ids = sorted([i for i in range(97) if pieces[i]["weight"].shape == torch.Size([48, 96])])
last_id = [i for i in range(97) if pieces[i]["weight"].shape == torch.Size([1, 48])][0]

print(f"  {len(inp_ids)} inp blocks, {len(out_ids)} out blocks, last layer = piece {last_id}")


# --- Step 2: Pair inp/out layers using product symmetry ---

print("Pairing inp/out layers...")
n = len(inp_ids)
cost = torch.zeros(n, n)
for r, ii in enumerate(inp_ids):
    for c, oi in enumerate(out_ids):
        M = pieces[oi]["weight"] @ pieces[ii]["weight"]  # [48, 48]
        cost[r, c] = (M - M.T).norm() / M.norm()

row_ind, col_ind = linear_sum_assignment(cost.numpy())
pair_map = {}
for r, c in zip(row_ind, col_ind):
    pair_map[inp_ids[r]] = out_ids[c]

print(f"  Paired {len(pair_map)} blocks")


# --- Step 3: Order blocks ---

def make_block(inp_id, out_id):
    block = Block(48, 96)
    block.inp.load_state_dict({"weight": pieces[inp_id]["weight"], "bias": pieces[inp_id]["bias"]})
    block.out.load_state_dict({"weight": pieces[out_id]["weight"], "bias": pieces[out_id]["bias"]})
    return block

# Initial order: sort by |f(x)| on raw inputs
print("Computing initial ordering by |f(x)|...")
f_norms = []
with torch.no_grad():
    for ii, oi in pair_map.items():
        block = make_block(ii, oi)
        f_x = block(inputs) - inputs
        f_norms.append((f_x.norm(dim=1).mean().item(), ii, oi))
f_norms.sort()
order = [(ii, oi) for _, ii, oi in f_norms]

def evaluate(order):
    with torch.no_grad():
        x = inputs.clone()
        for ii, oi in order:
            x = make_block(ii, oi)(x)
        out = LastLayer(48, 1).layer
        out.load_state_dict({"weight": pieces[last_id]["weight"], "bias": pieces[last_id]["bias"]})
        return ((out(x).squeeze() - preds) ** 2).mean().item()

mse = evaluate(order)
print(f"  Initial MSE vs predictions: {mse:.6f}")

# Refine: greedy adjacent swaps
print("Refining with adjacent swaps...")
iteration = 0
improved = True
while improved:
    improved = False
    swaps = 0
    for i in range(len(order) - 1):
        new_order = order.copy()
        new_order[i], new_order[i + 1] = new_order[i + 1], new_order[i]
        new_mse = evaluate(new_order)
        if new_mse < mse:
            order = new_order
            mse = new_mse
            improved = True
            swaps += 1
    iteration += 1
    print(f"  Iteration {iteration}: MSE={mse:.6f}, swaps={swaps}")

print(f"  Final MSE: {mse:.10f}")


# --- Step 4: Verify and save ---

print("\nVerifying solution...")
last_layer = LastLayer(48, 1)
last_layer.layer.load_state_dict({"weight": pieces[last_id]["weight"], "bias": pieces[last_id]["bias"]})

with torch.no_grad():
    x = inputs.clone()
    for ii, oi in order:
        x = make_block(ii, oi)(x)
    model_out = last_layer(x).squeeze()

corr_pred = np.corrcoef(model_out.numpy(), preds.numpy())[0, 1]
corr_true = np.corrcoef(model_out.numpy(), targets.numpy())[0, 1]
mae_pred = (model_out - preds).abs().mean().item()
print(f"  corr(model, pred):  {corr_pred:.6f}")
print(f"  corr(model, true):  {corr_true:.6f}")
print(f"  MAE(model, pred):   {mae_pred:.6f}")

# Build the permutation: position -> piece index
# The model processes: block_0.inp, block_0.out, block_1.inp, block_1.out, ..., last_layer
permutation = []
for ii, oi in order:
    permutation.append(ii)
    permutation.append(oi)
permutation.append(last_id)

assert len(permutation) == 97
assert sorted(permutation) == list(range(97))

# Save solution
solution = {
    "permutation": permutation,
    "order": [{"position": idx, "inp": ii, "out": oi} for idx, (ii, oi) in enumerate(order)],
    "last_layer": last_id,
    "verification": {
        "corr_with_pred": float(corr_pred),
        "corr_with_true": float(corr_true),
        "mae_with_pred": float(mae_pred),
    },
}
with open("solution.json", "w") as f:
    json.dump(solution, f, indent=2)

print(f"\nSolution saved to solution.json")
print(f"Permutation: {permutation}")
