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

# データ読み込み
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def load_data(batch_size = 16, num_workers = 1, test_size = None, seed = 0):

    def collate_fn(batch):
        inputs = torch.stack([B[0] for B in batch])
        teacher_signals = torch.tensor([B[1] for B in batch], dtype = torch.int64)

        return inputs, teacher_signals

    generator = torch.Generator()
    generator.manual_seed(seed)

    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root = "./data",
        train = True,
        download = True,
        transform = transform
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        collate_fn = collate_fn,
        num_workers = num_workers,
        shuffle = True,
        generator = generator
    )

    test_dataset = datasets.MNIST(
        root = "./data",
        train = False,
        download = True,
        transform = transform
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size = batch_size,
        collate_fn = collate_fn,
        num_workers = num_workers,
        shuffle = False
    )

    return train_dataloader, test_dataloader

# モデル読み込み
def load_model(seed = 0):

    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()

            self.fc1 = nn.Linear(784, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, inputs):
            inputs = inputs.view(inputs.size(0), -1)
            Y = self.fc1(inputs)
            Y = torch.relu(Y)
            Y = self.fc2(Y)

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

def train(target_path, optimizer, optimizer_params,
          epochs = 100, batch_size = 16, test_size = 0.1, num_workers = 1,
          seed = 0, verbose = False):

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
            model.train()
            if hasattr(optimizer, "train"): optimizer.train()

            optimizer.zero_grad()
            Y_batch = model(X_batch)
            E_batch = loss_func(Y_batch, T_batch)
            E_batch.backward()
            optimizer.step()

        else:
            model.eval()
            if hasattr(optimizer, "eval"): optimizer.eval()

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
                                                  test_size = test_size,
                                                  num_workers = num_workers,
                                                  seed = seed,)
    model = load_model(seed = seed).to(device)
    optimizer = optimizer(model.parameters(), **optimizer_params)


    logger = ResultLogger()
    logger.set_names(*["train_loss", "train_acc", "test_loss", "test_acc"])

    for i in range(epochs):
        if verbose: print(f"epoch; {i}")

        # 0エポック目は学習しない
        if i == 0:
            train_loss, train_acc = epoch(train_dataloader, model)
        else:
            train_loss, train_acc = epoch(train_dataloader, model, optimizer)
        test_loss, test_acc = epoch(test_dataloader, model)

        logger(train_loss, train_acc, test_loss, test_acc)

    logger.save(target_path)

def train_all(target_dir, optimizer, search_space, fixed_params,
              epochs = 100, batch_size = 16, test_size = 0.1, num_workers = 1, 
              num_seed = 10, verbose = False):

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
                  seed = seed,
                  epochs = epochs,
                  batch_size = batch_size,
                  test_size = test_size,
                  num_workers = num_workers,
                  verbose = verbose
                )