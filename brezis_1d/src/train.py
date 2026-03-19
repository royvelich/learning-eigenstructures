"""
Training script for the CNN Model.
It tests the last model. Maybe now after removing the bad slides, we shouldn't test with the best model.
"""

import argparse
import torch

from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from datamodules.zero_one_datamodule import ZeroOneDatamodule
from models.decomposition_laplacian import DecompositionLaplacianNet

def list_of_strings(arg):
    """
    Convert a comma separated string to a list of strings.
    """
    return arg.split(',')

def main(args):

    seed_everything(1994)

    
    wandb_logger = WandbLogger(name=args.name, project="Neural Operators")

    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(dirpath=f'checkpoints/{args.name}', save_last=True)

    dm = ZeroOneDatamodule(args)
    dm.setup()

    model = DecompositionLaplacianNet(args)

    trainer=Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        max_epochs=args.max_epochs,
        log_every_n_steps=10,
        accelerator="gpu",
        check_val_every_n_epoch=1,
        devices=torch.cuda.device_count()
    )

    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())

def arg_parser():
    """
    Argument parser for the training script.
    """
    parser = argparse.ArgumentParser(description='Train a CNN model')

    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--signal_length', type=int, required=True)
    parser.add_argument('--activation', type=str, required=True)
    parser.add_argument('--smoothing_sigma', type=float, required=True)
    parser.add_argument('--k', type=int, required=True, help='Number of eigenvectors to calculate')
    parser.add_argument('--signals_per_epoch', type=int, required=True)
    parser.add_argument('--hidden_dim', type=int, required=True)
    parser.add_argument('--hidden_layers', type=int, required=True)
    parser.add_argument('--max_epochs', type=int, required=True)
    parser.add_argument('--num_workers', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--weight_decay', type=float, required=True)

    return parser.parse_args()

if __name__ == '__main__':
    main(arg_parser())