import os
import numpy as np
import random
from tqdm import tqdm

import sklearn.datasets as sk_datasets
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append("../")
from utils import *

# 全手法で共通のハイパーパラメータ
EPOCHS = 100
BATCH_SIZE = 16
NUM_SEED = 10
TEST_SIZE = 0.1
NUM_WORKERS = 1
METRICS = ["train_loss", "train_acc", "test_loss", "test_acc"]

# データ読み込み
def load_data(batch_size = BATCH_SIZE, seed = 0, test_size = TEST_SIZE, num_workers = NUM_WORKERS):

    class MyDataset(Dataset):
        def __init__(self, inputs, teacher_signals):
            if len(inputs) != len(teacher_signals):
                raise

            self.inputs = inputs
            self.teacher_signals = teacher_signals

        def __len__(self):
            return len(self.inputs)

        def __getitem__(self, idx):
            input = torch.tensor( self.inputs[idx], dtype=torch.float32 )
            teacher_signal = torch.tensor(self.teacher_signals[idx], dtype=torch.int64)

            return input, teacher_signal

    def collate_fn(batch):
        inputs = torch.stack( [B[0] for B in batch] )
        teacher_signals = torch.stack( [B[1] for B in batch] )

        return inputs, teacher_signals

    generator = torch.Generator()
    generator.manual_seed(seed)

    iris = sk_datasets.load_iris()
    inputs = iris.data
    teacher_signals = iris.target

    tmp = list(zip(inputs, teacher_signals))
    train_tmp, test_tmp = train_test_split(tmp, test_size = test_size, random_state = seed)

    train_inputs, train_teacher_signals = zip(*train_tmp)
    train_dataset = MyDataset(train_inputs, train_teacher_signals)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        collate_fn = collate_fn,
        num_workers = num_workers,
        shuffle = True,
        generator = generator
    )

    test_inputs, test_teacher_signals = zip(*test_tmp)
    test_dataset = MyDataset(test_inputs, test_teacher_signals)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size = batch_size,
        collate_fn = collate_fn,
        num_workers = num_workers,
        shuffle = False,
    )

    return train_dataloader, test_dataloader

# モデル読み込み
def load_model(seed = 0):

    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()

            self.fc1 = nn.Linear(4, 3)

        def forward(self, inputs):
            Y = self.fc1(inputs)

            return Y

    def set_seed(seed = 0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed(seed = seed)
    model = MyModel()

    return model

def train(target_path, optimizer, optimizer_params, scheduler = None,
          epochs = EPOCHS, batch_size = BATCH_SIZE,
          seed = 0, test_size = TEST_SIZE, num_workers = NUM_WORKERS,
          verbose = False):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def loss_func(input, target):
        E = nn.CrossEntropyLoss()(input, target)
        return E

    def accuracy_func(Y, T):
        Y = Y.argmax(dim = 1)
        A = (Y == T).float().mean()
        return A

    def iteration(X_batch, T_batch, model, optimizer = None):
        X_batch = X_batch.to(device)
        T_batch = T_batch.to(device)

        if optimizer is not None:
            optimizer.zero_grad()
            Y_batch = model(X_batch)
            E_batch = loss_func(Y_batch, T_batch)
            E_batch.backward()
            optimizer.step()

        else:
            with torch.no_grad():
                Y_batch = model(X_batch)
                E_batch = loss_func(Y_batch, T_batch)

        A_batch = accuracy_func(Y_batch, T_batch)

        return E_batch.item(), A_batch.item()

    def epoch(dataloader, model, optimizer = None):

        E = 0
        A = 0
        total_data = 0

        if verbose: pb = tqdm(dataloader)
        else: pb = dataloader

        for X_batch, T_batch in pb:

            E_batch, A_batch = iteration(X_batch, T_batch, model, optimizer)

            E += E_batch * len(X_batch)
            A += A_batch * len(X_batch)
            total_data += len(X_batch)

            if verbose: pb.set_postfix({"loss": E / total_data,
                                        "accuracy": A / total_data})

        E /= total_data
        A /= total_data

        return E, A

    train_dataloader, test_dataloader = load_data(batch_size = batch_size,
                                                  seed = seed,
                                                  test_size = test_size,
                                                  num_workers = num_workers)
    model = load_model(seed = seed).to(device)
    optimizer = optimizer(model.parameters(), **optimizer_params)

    logger = ResultLogger()
    logger.set_names(*METRICS)

    for i in range(epochs):
        if verbose: print(f"epoch; {i}")

        # 0エポック目は学習しない
        if i == 0:
            train_loss, train_acc = epoch(train_dataloader, model)
        else:
            train_loss, train_acc = epoch(train_dataloader, model, optimizer)
        test_loss, test_acc = epoch(test_dataloader, model)

        logger(train_loss, train_acc, test_loss, test_acc)

        if scheduler:
            scheduler.step()

    logger.save(target_path)

def train_all(target_dir, optimizer, search_space, fixed_params,
              epochs = EPOCHS, batch_size = BATCH_SIZE,
              num_seed = NUM_SEED, test_size = TEST_SIZE,
              num_workers = NUM_WORKERS, verbose = False):

    for PARAMS in product(*search_space.values()):

        # 探索空間のパラメータ
        params = {
            K: P for K, P in zip(search_space.keys(), PARAMS)
        }
        
        for seed in range(num_seed):
            
            tmp = ""
            for V in params.values():
                tmp += f"{V}_"
            tmp = tmp[:-1]
            target_path = f"{target_dir}/{tmp}_{seed}.json"

            if os.path.exists(target_path):
                continue

            # 最適化手法のパラメータ
            optimizer_params = {
                **params,
                **fixed_params
            }
            
            print(target_path)
            train(target_path = target_path,
                  optimizer = optimizer,
                  optimizer_params = optimizer_params,
                  scheduler = None,
                  seed = seed,

                  epochs = epochs,
                  batch_size = batch_size,
                  test_size = test_size,
                  num_workers = num_workers,
                  verbose = verbose
                )