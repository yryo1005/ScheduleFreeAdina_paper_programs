import json
import matplotlib.pyplot as plt
from itertools import product

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
    
# 全seedの中でベストの結果を抽出
def get_best_histories(target_dir, search_space, num_seed, metric = "test_acc", mode = "min"):
    
    best_histories = dict()
    for PARAMS in product(*search_space.values()):

        # 探索空間のパラメータ
        params = {
            K: P for K, P in zip(search_space.keys(), PARAMS)
        }

        best_metric = None
        for seed in range(num_seed):
            tmp = ""
            for V in params.values():
                tmp += f"{V}_"
            tmp = tmp[:-1]
            target_path = f"{target_dir}/{tmp}_{seed}.json"
            logger = ResultLogger(target_path = target_path)

            if best_metric is None:
                best_metric = logger[metric][-1]
            else:
                if mode == "min":
                    best_metric = min(best_metric, logger[metric][-1])
                elif mode == "max":
                    best_metric = max(best_metric, logger[metric][-1])
            if best_metric == logger[metric][-1]:
                best_histories[tmp] = logger

    return best_histories

def plot_training_results(target_dir, best_histories, search_space, metrics, ):

    fig = plt.figure(figsize = (10, 10))
    for i, M in enumerate(metrics):
        # TODO
        ax = fig.add_subplot(2, 2, i + 1)
        for K, V in best_histories.items():
            ax.plot(V[M], label = f"{K}")
        ax.set_xlabel("epoch")
        ax.set_ylabel(M)
        ax.set_title(M)
        ax.grid()

    tmp = ""
    for K in search_space.keys():
        tmp += f"{K}_"
    tmp = tmp[:-1]
    fig.suptitle(tmp)

    fig.legend(
        labels=[K for K in best_histories.keys()],
        loc='upper center',
        ncol=5,  # 横に並べる列数
        bbox_to_anchor=(0.5, 1.05) # figure の外に配置
    )

    fig.savefig(f"{target_dir}/training_results.png", bbox_inches = "tight")
    plt.show()