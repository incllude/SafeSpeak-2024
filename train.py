from utils import load_config, seed_everything, get_data_for_dataset, SpoofDataset, pad
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader, WeightedRandomSampler
from lightning.pytorch.loggers import WandbLogger
from argparse import ArgumentParser
from modules import SpoofModule
from dotenv import load_dotenv
from models import AASIST2
import lightning as l
import pandas as pd
import numpy as np
import wandb
import torch
import os


parser = ArgumentParser()
parser.add_argument('--config_path', type=str, required=True)
args = parser.parse_args()

load_dotenv()
WANDB_KEY = os.getenv("WANDB_KEY")
if not WANDB_KEY:
    raise ValueError("WANDB_KEY not found in .env file")


config = load_config(args.config_path)
seed_everything()
wandb.login(key=WANDB_KEY)


train_ids, train_labels = get_data_for_dataset(
    config.asvspoof_2019_data_path.joinpath("ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt")
    )
val_ids, val_labels = get_data_for_dataset(
    config.asvspoof_2019_data_path.joinpath("ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt")
    )
test_ids, test_labels = get_data_for_dataset(
    config.asvspoof_2019_data_path.joinpath("ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt")
    )
new_val_ids, new_val_labels = get_data_for_dataset(
    config.asvspoof_2021_data_path.joinpath("LA-keys-full/keys/LA/CM/trial_metadata.txt"),
    label_idx=-3
    )

error_idxs = np.load(config.path_to_problematic_idxs)
new_val_ids, new_val_labels = new_val_ids[error_idxs], new_val_labels[error_idxs]


ratio = len(val_labels) / sum(val_labels)

pos_test_ids = np.where(test_labels > 0.5)[0]
neg_test_ids = np.where(test_labels < 0.5)[0]

pos_val_ids = np.random.choice(pos_test_ids, sum(val_labels), replace=False)
neg_val_ids = np.random.choice(neg_test_ids, len(val_labels) - sum(val_labels), replace=False)

pos_train_ids = np.setdiff1d(pos_test_ids, pos_val_ids)
neg_train_ids = np.setdiff1d(neg_test_ids, neg_val_ids)

test_ids_for_train = test_ids[np.concatenate((pos_train_ids, neg_train_ids))]
test_ids_for_val = test_ids[np.concatenate((pos_val_ids, neg_val_ids))]
test_labels_for_train = [1] * len(pos_train_ids) + [0] * len(neg_train_ids)
test_labels_for_val = [1] * len(pos_val_ids) + [0] * len(neg_val_ids)


train_path = config.asvspoof_2019_data_path.joinpath("ASVspoof2019_LA_train")
val_path = config.asvspoof_2019_data_path.joinpath("ASVspoof2019_LA_dev")
test_path = config.asvspoof_2019_data_path.joinpath("ASVspoof2019_LA_eval")
new_val_path = config.asvspoof_2021_data_path.joinpath("ASVspoof2021_LA_eval")

train_files = [train_path.joinpath(f"flac/{x}.flac") for x in train_ids]
val_files = [test_path.joinpath(f"flac/{x}.flac") for x in test_ids_for_val]
test_files = [test_path.joinpath(f"flac/{x}.flac") for x in test_ids_for_train]
new_val_files = [new_val_path.joinpath(f"flac/{x}.flac") for x in new_val_ids]

train_files = np.concatenate((train_files, test_files, new_val_files))
train_labels = np.concatenate((train_labels, test_labels_for_train, new_val_labels))
val_labels = test_labels_for_val


train_ds = SpoofDataset(files=train_files, labels=train_labels, is_train=True, p_aug=config.aug_prob)
val_ds = SpoofDataset(files=val_files, labels=val_labels, pad_fn=pad, is_train=False)

vc = pd.Series(train_labels).value_counts()
config.epoch_size = int(vc.min()) * 2
weights = 1 / vc
weights = vc[train_labels].values
sampler = WeightedRandomSampler(weights=weights,
                                num_samples=config.epoch_size,
                                replacement=False)

train_dl = DataLoader(dataset=train_ds,
                      batch_size=config.batch_size,
                      sampler=sampler,
                      num_workers=4,
                      drop_last=True,
                      pin_memory=True)
val_dl = DataLoader(dataset=val_ds,
                    batch_size=config.batch_size,
                    shuffle=False,
                    num_workers=4,
                    drop_last=False,
                    pin_memory=True)


wandb_logger = WandbLogger(project="SafeSpeak-2024", name=config.run_name)
checkpoint_callback = ModelCheckpoint(dirpath="./",
                                      filename="model_{epoch:02}_{EER:.3f}",
                                      verbose=False,
                                      every_n_epochs=1,
                                      save_top_k=1,
                                      monitor="EER",
                                      mode="min")
lr_monitor = LearningRateMonitor(logging_interval='epoch')


model = AASIST2(device="cuda", path_to_pretrained_wav2vec=config.path_to_pretrained_wav2vec)

checkpoint = torch.load(config.path_to_pretrained_aasist, weights_only=True)
model_state_dict = model.state_dict()
filtered_state_dict = {k: v for k, v in checkpoint.items() if k in model_state_dict and v.shape == model_state_dict[k].shape}
model.load_state_dict(filtered_state_dict, strict=False)

module = SpoofModule(model=model, scale=config.scale, margin=config.margin)
module.set_training_settings(optimizer_cfg={"type"  : "Adam",
                                            "params": {"lr": 0.001,
                                                       "weight_decay": 0.0001}},
                             scheduler_cfg={"type"  : "OneCycleLR",
                                            "params": {"max_lr": 0.0001,
                                                       "epochs": config.epochs,
                                                       "steps_per_epoch": config.epoch_size//config.batch_size,
                                                       "pct_start": 0.00,
                                                       "cycle_momentum": False,
                                                       "div_factor": 100,
                                                       "final_div_factor": 100}})

trainer = l.Trainer(accelerator="cuda",
                    max_epochs=config.epoch_stop,
                    enable_progress_bar=True,
                    log_every_n_steps=config.epoch_size//config.batch_size,
                    logger=wandb_logger,
                    num_sanity_val_steps=0,
                    callbacks=[checkpoint_callback, lr_monitor])

trainer.fit(model=module, train_dataloaders=train_dl, val_dataloaders=val_dl)
wandb.finish()
