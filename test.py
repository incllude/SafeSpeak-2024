from utils import load_config, seed_everything, get_data_for_dataset, SpoofDataset, CompDataset, pad
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader, WeightedRandomSampler
from lightning.pytorch.loggers import WandbLogger
from argparse import ArgumentParser
from modules import SpoofModule
from dotenv import load_dotenv
from models import AASIST2
from tqdm import tqdm
import lightning as l
import pandas as pd
import numpy as np
import wandb
import torch
import os


parser = ArgumentParser()
parser.add_argument('--ckpt_path', type=str, required=True)
parser.add_argument('--config_path', type=str, required=True)
args = parser.parse_args()

config = load_config(args.config_path)


model = AASIST2(device="cuda", path_to_pretrained_wav2vec=config.path_to_pretrained_wav2vec)
module = SpoofModule(model=model, scale=15, margin=0.3)

module = SpoofModule.load_from_checkpoint(args.ckpt_path, model=model, scale=15, margin=0.3).eval()


comp_files = list(config.test_data_path.iterdir())
comp_ds = CompDataset(files=comp_files, pad_fn=pad)
comp_dl = DataLoader(dataset=comp_ds,
                     batch_size=config.batch_size * 4,
                     shuffle=False,
                     num_workers=4,
                     drop_last=False)


idxs, probs = [], []

with torch.no_grad():
    for batch in tqdm(comp_dl):
    
        x, idx = batch
        idxs.extend(idx.tolist())

        x = x.to(module.device)
        prob = module.forward(x)
        probs.extend(prob.flatten().tolist())


submission = pd.DataFrame()
submission["ID"] = idxs
submission["score"] = probs
submission.to_csv("submission.csv", index=False)
