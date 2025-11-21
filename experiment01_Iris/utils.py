
import json
import numpy as np
import random
from tqdm import tqdm

import sklearn.datasets as sk_datasets
from sklearn.model_selection import train_test_split

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

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
            """
                inputs: (batch_size, 4)
            """
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

# 結果ロガー
class ResultLogger:
    def __init__(self, target_path=None):
        self.names = None
        self.history = {}

        if target_path:
            self.load(target_path)

    def set_names(self, *names):
        if self.names:
            raise

        self.names = list(names)
        for name in self.names:
            if name not in self.history:
                self.history[name] = []

    def __call__(self, *values):
        if self.names is None:
            raise
        if len(values) != len(self.names):
            raise

        for name, value in zip(self.names, values):
            self.history[name].append(value)

    def save(self, target_path):
        with open(target_path, "w") as f:
            json.dump(self.history, f, indent=4)

    def load(self, target_path):
        with open(target_path, "r") as f:
            data = json.load(f)
            self.history = data
            self.names = list(data.keys())

    def __getitem__(self, key):
        return self.history.get(key, [])

def train(target_path, optimizer, optimizer_params, scheduler = None,
          epochs = EPOCHS, batch_size = BATCH_SIZE,
          seed = 0, test_size = TEST_SIZE, num_workers = NUM_WORKERS,
          verbose = False):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def loss_func(input, target):
        """
            input: (batch_size, 4) モデルの出力
            target: (batch_size, 1) 教師信号
        """
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