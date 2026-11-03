import os
import logging

import torch
torch.set_float32_matmul_precision("medium")

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from data_loader import DataModule
from trainer import MusicClassifier
from omegaconf import DictConfig
import hydra
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from pytorch_lightning.strategies import DDPStrategy

log = logging.getLogger(__name__)

def get_latest_version(log_dir):
    if not os.path.exists(log_dir):
        return None
    version_dirs = [d for d in os.listdir(log_dir) if d.startswith('version_')]
    if not version_dirs:
        return None
    version_dirs.sort(key=lambda x: int(x.split('_')[-1]))
    return version_dirs[-1]

@hydra.main(version_base=None, config_path="config", config_name="train_config")
def main(config: DictConfig):

    # --- rerun-v2: refuse removed dataset, seed everything ---
    if "globalmood_va" in config.datasets:
        raise ValueError("globalmood_va was removed from the training pool (rerun-v2).")
    seed = int(config.get("seed", 0))
    pl.seed_everything(seed, workers=True)
    print(f"[train] seed={seed} run_name={config.get('run_name')} datasets={list(config.datasets)}")

    log_base_dir = 'tb_logs/train_audio_classification'

    # Create log directory if not exists
    os.makedirs(log_base_dir, exist_ok=True)

    is_mt = False
    if "mt" in config.model.classifier:
        is_mt = True

    logger = TensorBoardLogger("tb_logs", name="train_audio_classification",
                               version=config.get("run_name") or None)
    logger.log_hyperparams(config)
    train_log_dir = logger.log_dir
    print(f"Logging to {train_log_dir}")
    log.info("Training starts")

    data_module = DataModule(config)
    data_module.setup()

    # Get the list of dataloaders for both train and validation
    trainloaders = {dataset_name: loader for dataset_name, loader in zip(config.datasets, data_module.train_dataloader())}
    vallowers = {dataset_name: loader for dataset_name, loader in zip(config.datasets, data_module.val_dataloader())}

    # Combine multiple loaders using CombinedLoader
    combined_train_loader = CombinedLoader(trainloaders, mode="max_size")
    combined_val_loader = CombinedLoader(vallowers, mode="max_size")

    # val_epoch.txt goes in THIS run's log dir (the old "latest version" lookup
    # can point at another job's folder when several jobs start together)
    val_epoch_file = os.path.join(train_log_dir, 'val_epoch.txt')

    model = MusicClassifier(config, output_file=val_epoch_file)

    # Always define early stopping
    early_stop_callback = EarlyStopping(**config.earlystopping)

    if is_mt:
        checkpoint_callback_mood = ModelCheckpoint(**config.checkpoint_mood)
        checkpoint_callback_va = ModelCheckpoint(**config.checkpoint_va)

        if config.model.kd == True:
            trainer = pl.Trainer(
                **config.trainer,
                strategy=DDPStrategy(find_unused_parameters=True),
                callbacks=[checkpoint_callback_mood, checkpoint_callback_va, early_stop_callback],
                logger=logger,
                num_sanity_val_steps=0
            )
        else:
            trainer = pl.Trainer(
                **config.trainer,
                strategy=DDPStrategy(find_unused_parameters=True),
                callbacks=[checkpoint_callback_mood, checkpoint_callback_va, early_stop_callback],
                logger=logger,
                num_sanity_val_steps=0
            )
    else:
        checkpoint_callback = ModelCheckpoint(**config.checkpoint)
        trainer = pl.Trainer(
            **config.trainer,
            callbacks=[checkpoint_callback, early_stop_callback],
            logger=logger,
            num_sanity_val_steps=0
        )

    trainer.fit(model, combined_train_loader, combined_val_loader)

    if trainer.global_rank == 0:
        best_checkpoint_file = os.path.join(train_log_dir, 'best_checkpoint.txt')
        with open(best_checkpoint_file, 'w') as f:
            if is_mt:
                f.write(f"Best checkpoint (mood): {checkpoint_callback_mood.best_model_path}\n")
                f.write(f"Best checkpoint (va): {checkpoint_callback_va.best_model_path}\n")
            else:
                f.write(f"Best checkpoint: {checkpoint_callback.best_model_path}\n")
            f.write(f"Version: {logger.version}\n")

if __name__ == '__main__':
    main()

