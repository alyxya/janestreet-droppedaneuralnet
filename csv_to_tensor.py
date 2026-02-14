import torch
import csv

with open("historical_data_and_pieces/historical_data.csv") as f:
    reader = csv.reader(f)
    header = next(reader)
    data = [[float(x) for x in row] for row in reader]

tensor = torch.tensor(data)
inputs = tensor[:, :48]   # measurement_0 through measurement_47
preds = tensor[:, 48]     # model predictions
targets = tensor[:, 49]   # true values

torch.save({"inputs": inputs, "preds": preds, "targets": targets}, "historical_data.pt")
print(f"inputs:  {inputs.shape}")
print(f"preds:   {preds.shape}")
print(f"targets: {targets.shape}")
