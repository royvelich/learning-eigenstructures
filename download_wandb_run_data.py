import wandb
import pandas as pd

api = wandb.Api()
run = api.run("gip-technion/neural-laplacian/rc4n2ckl")

# Convert all history to DataFrame ONCE
history = pd.DataFrame([row for row in run.scan_history()])

# Define the epoch you want
target_epoch = 98  # Change this to whatever epoch you want

# Filter for the target epoch where validation metric is not null
epoch_val = history[
    (history['epoch'] == target_epoch) &
    (history['val/ReconstructionLoss/dataloader_idx_0'].notna())
]

print(epoch_val)
epoch_val.to_csv(f'epoch_{target_epoch}_validation.csv', index=False)