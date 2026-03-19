# standard library
from typing import Dict, Optional
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')
import json
import argparse
import yaml

# os.environ["WANDB_DIR"] = "C:/wandb/WANDB_DIR"
# os.environ["WANDB_ARTIFACT_DIR"] = "C:/wandb/WANDB_ARTIFACT_DIR"
# os.environ["WANDB_CACHE_DIR"] = "C:/wandb/WANDB_CACHE_DIR"
# os.environ["WANDB_CONFIG_DIR"] = "C:/wandb/WANDB_CONFIG_DIR"
# os.environ["WANDB_DATA_DIR"] = "C:/wandb/WANDB_DATA_DIR"

os.environ["WANDB_DIR"] = "C:/wandb/WANDB_DIR"
os.environ["WANDB_ARTIFACT_DIR"] = "C:/wandb/WANDB_ARTIFACT_DIR"
os.environ["WANDB_CACHE_DIR"] = "C:/wandb/WANDB_CACHE_DIR"
os.environ["WANDB_CONFIG_DIR"] = "C:/wandb/WANDB_CONFIG_DIR"
os.environ["WANDB_DATA_DIR"] = "C:/wandb/WANDB_DATA_DIR"

# hydra
import hydra

# omegaconf
from omegaconf import OmegaConf
from omegaconf import DictConfig

# torch
import torch
import torch.distributed as dist
torch.multiprocessing.set_sharing_strategy('file_system')
# torch.autograd.set_detect_anomaly(True)

# wandb
import wandb
from pytorch_lightning.loggers import WandbLogger

# lightning
import pytorch_lightning as pl
import lightning_fabric.loggers


def main(config: DictConfig) -> None:
    torch.set_float32_matmul_precision(precision='medium')
    pl.seed_everything(seed=config.globals.seed)
    data_module = hydra.utils.instantiate(config=config.data_module.module)
    model = hydra.utils.instantiate(config=config.model.module, optimizer_cfg=config.optimizer, scheduler_cfg=config.scheduler if 'scheduler' in config else None)
    trainer = hydra.utils.instantiate(config=config.trainer)
    trainer.cfg = config
    trainer.fit(model=model, datamodule=data_module, ckpt_path=config.globals.ckpt_path)


def wandb_sweep_main():
    if 'WANDB_SWEEP_CONFIG' not in os.environ:
        wandb.init()
        config = OmegaConf.create(dict(wandb.config))
        config = OmegaConf.to_container(config, resolve=True)
        config = OmegaConf.create(config)
        os.environ['WANDB_SWEEP_CONFIG'] = OmegaConf.to_yaml(config)
    else:
        config = OmegaConf.create(yaml.safe_load(os.environ['WANDB_SWEEP_CONFIG']))

    main(config=config)


@hydra.main(version_base="1.2", config_path="config/training")
def main_hydra(config: DictConfig) -> None:
    main(config=config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-id", type=str, default=None)
    args, _ = parser.parse_known_args()

    if args.sweep_id is None:
        main_hydra()
    else:
        if 'WANDB_SWEEP_CONFIG' not in os.environ:
            wandb.agent(sweep_id=args.sweep_id, function=wandb_sweep_main, count=1)
        else:
            wandb_sweep_main()
