import matplotlib.pyplot as plt
from itertools import product

from torch import optim

from Wine_utils import *
import sys
sys.path.append("../")
from utils import *

###
OPTIMIZER = optim.SGD
TARGET_DIR = "results/SGD"

# 探索するパラメータと探索空間
SEARCH_SPACE = {
    "lr": [1.0, 0.1, 0.01, 0.001, 0.0001],
    "weight_decay": [0.1, 0.01, 0.001, 0.0001, 0.0],
}
# 固定のパラメータ
FIXED_PARAMS ={
    "momentum": 0.0,
    "dampening": 0.0,
    "nesterov": False,
}

os.makedirs(TARGET_DIR, exist_ok = True)

###
train_all(target_dir=TARGET_DIR,
          optimizer=OPTIMIZER,
          search_space=SEARCH_SPACE,
          fixed_params=FIXED_PARAMS,
          epochs=EPOCHS,
          batch_size=BATCH_SIZE,
          test_size=TEST_SIZE,
          num_workers=NUM_WORKERS,
          num_seed=NUM_SEED,
          verbose=False)

###
best_histories = get_best_histories(target_dir=TARGET_DIR,
                                    search_space=SEARCH_SPACE,
                                    num_seed=NUM_SEED,
                                    metric="test_acc",
                                    mode="max")

plot_training_results(target_dir=TARGET_DIR,
                      best_histories=best_histories,
                      search_space=SEARCH_SPACE,
                      metrics=METRICS)

tmp = [
    [K, V["test_acc"][-1]] for K, V in best_histories.items()
]
tmp = sorted(tmp, key=lambda x: x[1], reverse=True)
for K, V in tmp:
    print(f"{K}: {V:.4f}")