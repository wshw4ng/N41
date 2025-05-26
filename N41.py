# Neighbors-for-one project
# 24.7.2. for sharing the code

import sys
import argparse

from pathlib import Path
from datetime import timedelta

import dateutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

import time
import os

from torch.utils.data import Dataset, DataLoader
from tqdm import trange

# HYPERPRAMETERS
N_HIDDENS = 200
N_LAYERS = 2
BATCH_SIZE = 1024
EPOCH = 32
DEFAULT_TIME_LAG = 1
DEFAULT_WINDOW_SIZE = 60 
EWM = 0.9
GPU_ID = 0

def dataframe_from_csv(target):
    return pd.read_csv(target, sep=',').rename(columns=lambda x: x.strip())

def dataframe_from_csvs(targets):
    return pd.concat([dataframe_from_csv(x) for x in targets])

def normalize(df):
    ndf = df.copy()
    for c in df.columns:
        if TAG_MIN[c] == TAG_MAX[c]:
            ndf[c] = df[c] - TAG_MIN[c]
        else:
            ndf[c] = (df[c] - TAG_MIN[c]) / (TAG_MAX[c] - TAG_MIN[c])
    return ndf

def boundary_check(df):
    x = np.array(df, dtype=np.float32)
    return np.any(x > 1.0), np.any(x < 0), np.any(np.isnan(x))

class StackedGRU(torch.nn.Module):
    def __init__(self, n_tags):
        super().__init__()
        self.rnn = torch.nn.GRU(
            input_size=n_tags,
            hidden_size=N_HIDDENS,
            num_layers=N_LAYERS,
            bidirectional=True,
            dropout=0,
        )
        self.fc = torch.nn.Linear(N_HIDDENS * 2, 1) # Finally, to get a single value

    def forward(self, x):
        x = x.transpose(0, 1)  # (batch, seq, params) -> (seq, batch, params)
        self.rnn.flatten_parameters()
        outs, _ = self.rnn(x)
        out = self.fc(outs[-1])        
        return out
    
class N41_Dataset(Dataset):
    def __init__(self, timestamps, df, window_size, stride=1, predict_size=1):
        self._window_size = window_size
        self._predict_size = predict_size
        
        self.ts = np.array(timestamps)
        self.df = df
        self.input_values = None 
        self.output_values = None
                
        self.valid_idxs = []        
        # Save the first index of every window
        for L in trange(len(self.ts) - window_size + 1):
            R = L + window_size - 1            
            if dateutil.parser.parse(self.ts[R]) - dateutil.parser.parse(self.ts[L])\
            == timedelta(seconds=(window_size-1)):
                self.valid_idxs.append(L)
                
        self.valid_idxs = np.array(self.valid_idxs, dtype=np.int32)[::stride]
        self.n_idxs = len(self.valid_idxs)
        print(f"# of valid windows: {self.n_idxs}", flush=True)
        
    def __len__(self):
        return self.n_idxs

    def __getitem__(self, idx):
        i = self.valid_idxs[idx]
        last = i + self._window_size - 1
        item = {"ts": self.ts[i + self._window_size - 1] }
        item["given"] = torch.from_numpy(self.input_values[i : i + self._window_size-self._predict_size])
        item["answer"] = torch.from_numpy(np.array([self.output_values[last]], dtype=np.float32))
        return item
    
    def set_inout(self, input_tags, output_tags):
        self.input_values  = np.array(self.df[input_tags], dtype=np.float32)
        self.output_values = np.array(self.df[output_tags], dtype=np.float32)
    

def train(dataset, model, batch_size, n_epochs):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters())
    loss_fn = torch.nn.MSELoss()
    epochs = trange(n_epochs, desc="training")
    best = {"loss": sys.float_info.max}
    loss_history = []
    for e in epochs: #range(n_epochs):
        epoch_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            given = batch["given"].to(device)
            guess = model(given)
            answer = batch["answer"].to(device)
            loss = loss_fn(answer, guess)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
        loss_history.append(epoch_loss)
        epochs.set_postfix_str(f"loss: {epoch_loss:.6f}")
        if epoch_loss < best["loss"]:
            best["state"] = model.state_dict()
            best["loss"] = epoch_loss
            best["epoch"] = e + 1
    return best, loss_history

def inference(dataset, model, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    ts, dist = [], []
    with torch.no_grad():
        for batch in dataloader:
            given = batch["given"].to(device)
            answer = batch["answer"].to(device)
            guess = model(given)
            ts.append(np.array(batch["ts"]))
            dist.append(torch.abs(answer - guess).cpu().numpy())            
    return (
        np.concatenate(ts),
        np.concatenate(dist),
    )

if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--gamma", default="0.3", help="Parameter gamma for neighboring features", required=True)
    argument_parser.add_argument("--data", help="Support \"swat\" or \"hai22\"", required=True)

    arguments = argument_parser.parse_args()
    DATASET = arguments.data
    NEIGHBOR_GAMMA = float(arguments.gamma)
    
    print(f"Stacked GRU with N41...")
    print(f"Dataset:{DATASET}")
    print(f"Gamma for Neighboring Features:{NEIGHBOR_GAMMA}")    

    if DATASET == "swat":
        TRAIN_DATASET = ['~/NAS/Data/SWaT_myver/SWaT_Dataset_Normal_v3.csv']
        TEST_DATASET = ['~/NAS/Data/SWaT_myver/SWaT_Dataset_Attack_v31.csv']
        ATTACK_FIELD = "label"
        COR_FILE  = './similarity/swat.npy'
        RESULT_DIR = "./swat_result/"
    elif DATASET == "hai22":
        TRAIN_DATASET = []
        for i in range(1, 7):
            TRAIN_DATASET.append('~/NAS/Data/hai-master/hai-22.04/train{}.csv'.format(i))
        TEST_DATASET = []
        for i in range(1, 5):
            TEST_DATASET.append('~/NAS/Data/hai-master/hai-22.04/test{}.csv'.format(i))
        ATTACK_FIELD = "Attack"
        COR_FILE  = './similarity/hai22.npy'
        RESULT_DIR = "./hai22_result/"

    TRAIN_DF_RAW = dataframe_from_csvs(TRAIN_DATASET)

    TIMESTAMP_FIELD = "timestamp"    
    VALID_COLUMNS_IN_TRAIN_DATASET = TRAIN_DF_RAW.columns.drop(
        [TIMESTAMP_FIELD, ATTACK_FIELD]
    )
    VALID_COLUMNS_IN_TRAIN_DATASET

    TAG_MIN = TRAIN_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET].min()
    TAG_MAX = TRAIN_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET].max()

    TRAIN_DF = normalize(TRAIN_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET]).ewm(alpha=EWM).mean()
    boundary_check(TRAIN_DF)
            
    TEST_DF_RAW = dataframe_from_csvs(TEST_DATASET)
    TEST_DF = normalize(TEST_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET]).ewm(alpha=EWM).mean()

    print(f"Load training and test datasets of {DATASET}")
    DATASET_TRAIN = N41_Dataset(TRAIN_DF_RAW[TIMESTAMP_FIELD], TRAIN_DF, DEFAULT_WINDOW_SIZE)        
    DATASET_TEST = N41_Dataset(TEST_DF_RAW[TIMESTAMP_FIELD], TEST_DF, DEFAULT_WINDOW_SIZE)

    # The below part is related to the neighbors
    COR_VAL = np.load(COR_FILE)
    for i in range(len(COR_VAL)):        
        for j in range(len(COR_VAL)):
            if COR_VAL[i][j] == -1.1:
                COR_VAL[i][j] = 0.0
            elif COR_VAL[i][j] < 0.0:
                COR_VAL[i][j] *= -1  

    device = torch.device("cuda:{}".format(GPU_ID) if torch.cuda.is_available() else "cpu")
    print(f"Running with {device}")

    for target_tid in range(len(VALID_COLUMNS_IN_TRAIN_DATASET)):
        input_tagnames = np.array(VALID_COLUMNS_IN_TRAIN_DATASET[np.where(COR_VAL[target_tid] >= NEIGHBOR_GAMMA)])
        output_tagnames = VALID_COLUMNS_IN_TRAIN_DATASET[target_tid]
        print(f"Training Model {target_tid}/{len(VALID_COLUMNS_IN_TRAIN_DATASET)}...")
        print(f"  * Output: {output_tagnames}")
        print(f"  * Input: {input_tagnames}")

        if len(input_tagnames) <= 1:
            print(f"Skip because of no neighbor")
            continue

        DATASET_TRAIN.set_inout(input_tagnames, output_tagnames)
        DATASET_TEST.set_inout(input_tagnames, output_tagnames)
               
        torch.cuda.empty_cache()

        MODEL = StackedGRU(n_tags=len(input_tagnames))
        MODEL.to(device)

        MODEL.train()
        BEST_MODEL, LOSS_HISTORY = train(DATASET_TRAIN, MODEL, BATCH_SIZE, EPOCH)

        MODEL.load_state_dict(BEST_MODEL["state"])
        
        MODEL.eval()
        CHECK_TS, CHECK_DIST = inference(DATASET_TEST, MODEL, BATCH_SIZE)

        if os.path.isdir(RESULT_DIR) == False:
            os.mkdir(RESULT_DIR)
        np.save('{}/{}'.format(RESULT_DIR, target_tid), np.mean(CHECK_DIST, axis=1))


