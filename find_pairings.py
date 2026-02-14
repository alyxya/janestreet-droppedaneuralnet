import torch
import json
from scipy.optimize import linear_sum_assignment

# Load all pieces
pieces = {}
for i in range(97):
    pieces[i] = torch.load(f"historical_data_and_pieces/pieces/piece_{i}.pth", map_location="cpu", weights_only=True)

# Separate by type
inp_ids = sorted([i for i in range(97) if pieces[i]["weight"].shape == torch.Size([96, 48])])
out_ids = sorted([i for i in range(97) if pieces[i]["weight"].shape == torch.Size([48, 96])])
last_layer_id = [i for i in range(97) if pieces[i]["weight"].shape == torch.Size([1, 48])][0]

print(f"Found {len(inp_ids)} inp blocks, {len(out_ids)} out blocks, last layer = piece {last_layer_id}")

# Build cost matrix using symmetry of out @ inp
# If inp ≈ out.T, then out @ inp should be nearly symmetric
n = len(inp_ids)
cost = torch.zeros(n, n)
for r, ii in enumerate(inp_ids):
    for c, oi in enumerate(out_ids):
        M = pieces[oi]["weight"] @ pieces[ii]["weight"]  # [48, 48]
        cost[r, c] = (M - M.T).norm() / M.norm()

# Hungarian algorithm to find optimal 1-to-1 assignment (minimize asymmetry)
row_ind, col_ind = linear_sum_assignment(cost.numpy())

pairings = {}
print("\nPairings (inp -> out):")
for r, c in zip(row_ind, col_ind):
    ii = inp_ids[r]
    oi = out_ids[c]
    sym = cost[r, c].item()
    best_c = cost[r].argmin().item()
    sec_c = cost[r].argsort()[1].item()
    gap = cost[r, sec_c].item() - cost[r, best_c].item()
    match = "Y" if c == best_c else "N"
    print(f"  inp {ii:2d} -> out {oi:2d}  sym={sym:.4f}  gap={gap:.4f}  greedy_match={match}")
    pairings[ii] = oi

# Save pairings
output = {
    "last_layer": last_layer_id,
    "blocks": [{"inp": ii, "out": oi} for ii, oi in sorted(pairings.items())],
}
with open("pairings.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved to pairings.json")
