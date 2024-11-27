from augmentations import ISD_additive_noise, LnL_convolutive_noise
from torch.utils.data import Dataset
from pathlib import Path
import soundfile as sf
import numpy as np
import random
import torch
import yaml


class Config:
    pass


def seed_everything(random_state):

    torch.backends.cudnn.deterministic = False
    torch.cuda.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    random.seed(random_state)


def load_config(config_path: str) -> Config:

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = Config()
    for key, value in config_dict.items():
        if "path" in key:
            value = Path(value)
        setattr(config, key, value)
        
    return config


def get_data_for_dataset(path, label_idx=-1):
    
    ids_list = []
    label_list = []
    with open(path, "r") as file:
        for line in file:
            line = line.split()
            id, label = line[1], line[label_idx]
            ids_list.append(id)
            label = 1 if label == "bonafide" else 0
            label_list.append(label)
            
    return np.array(ids_list), np.array(label_list)


def pad_random(x, max_len=64600):
    x_len = x.shape[0]

    if x_len > max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt : stt + max_len]

    num_repeats = max_len // x_len + 1
    padded_x = np.tile(x, num_repeats)[:max_len]
    return padded_x


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]

    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


class SpoofDataset(Dataset):
    def __init__(self, files, labels, pad_fn=pad_random, is_train=True, p_aug=0.):
        
        self.files = files
        self.labels = labels
        self.cut = 64600
        self.is_train = is_train
        self.pad_fn = pad_fn
        self.p_aug = p_aug

    def __getitem__(self, index):

        path_to_flac = self.files[index]
        audio, rate = sf.read(path_to_flac)
        if self.is_train and np.random.random() < self.p_aug:
            x = LnL_convolutive_noise(audio, 5, 5, 20, 8000, 100, 1000, 10, 100, 0, 0, 5, 20, rate)
            x = ISD_additive_noise(x, 10, 2)
        else:
            x = audio
        x_pad = self.pad_fn(x, self.cut)
        x_inp = torch.from_numpy(x_pad).float()
        
        return x_inp, torch.tensor(self.labels[index])

    def __len__(self):
        return len(self.files)
    

class CompDataset(Dataset):
    def __init__(self, files, pad_fn=pad_random, cut=64600):
        
        self.files = files
        self.cut = cut
        self.pad_fn = pad_fn

    def __getitem__(self, index):
        path_to_wav = self.files[index]
        idx = int(Path(path_to_wav).stem)
        audio, rate = sf.read(path_to_wav)
        x_pad = self.pad_fn(audio, self.cut)
        x_inp = torch.from_numpy(x_pad).float()
        return x_inp, idx
    
    def __len__(self):
        return len(self.files)


def compute_det_curve(bonafide_scores, spoof_scores):
    """
    function, that comuputes FRR and FAR with their thresholds

    args:
        bonafide_scores: score for bonafide speech
        spoof_scores: score for spoofed speech
    output:
        frr: false rejection rate
        far: false acceptance rate
        threshlods: thresholds for frr and far
    todo:
        rewrite to torch
        create tests
    """
    # number of scores
    n_scores = bonafide_scores.size + spoof_scores.size

    # bona fide scores and spoof scores
    all_scores = np.concatenate((bonafide_scores, spoof_scores))

    # label of bona fide score is 1
    # label of spoof score is 0
    labels = np.concatenate((np.ones(bonafide_scores.size), np.zeros(spoof_scores.size)))

    # indexes of sorted scores in all scores
    indices = np.argsort(all_scores, kind='mergesort')
    # sort labels based on scores
    labels = labels[indices]

    # Compute false rejection and false acceptance rates

    # tar cumulative value
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = spoof_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    # false rejection rates
    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / bonafide_scores.size))

    # false acceptance rates
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / spoof_scores.size))

    # Thresholds are the sorted scores
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))

    return frr, far, thresholds


def compute_eer(bonafide_scores, spoof_scores):
    """
    Returns equal error rate (EER) and the corresponding threshold.
    args:
        bonafide_scores: score for bonafide speech
        spoof_scores: score for spoofed speech
    output:
        eer: equal error rate
        threshold: index, where frr=far
    todo:
        rewrite to torch
        create tests
    """
    frr, far, thresholds = compute_det_curve(bonafide_scores, spoof_scores)

    # absolute differense between frr and far
    abs_diffs = np.abs(frr - far)

    # index of minimal absolute difference
    min_index = np.argmin(abs_diffs)

    # equal error rate
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]
