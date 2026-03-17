import torch

class InterpolatedScheduleFreeSGD(torch.optim.Optimizer):
    def __init__(
            self,
            params,
            lr = 0.01,
            gamma = 0.9,
            adaptive_gamma = False
    ):
        """
            0 < lr; 学習率
            0 <= gamme <= 1; 真のパラメータと平均パラメータの内分の割合
            adaptive_gamma; Trueの場合gammaを適応的に変更する
                            Falseの場合gammaは固定する
        """

        defaults = dict(
            lr = lr,
            gamma = gamma,
            adaptive_gamma = adaptive_gamma,
            k = 0,
            train_mode = True,
        )

        super().__init__(params, defaults)

    @torch.no_grad()
    def eval(self):
        """
            真のパラメータから平均パラメータへ切り替える関数
            推論の直前で使用する
        """
        for group in self.param_groups:
            train_mode = group['train_mode']
            if train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'r' in state:
                        p.copy_(state['r'])
                group['train_mode'] = False

    @torch.no_grad()
    def train(self):
        """
            平均パラメータから真のパラメータへ切り替える関数
            学習の直前で使用する
        """
        for group in self.param_groups:
            train_mode = group['train_mode']
            if not train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'w' in state:
                        p.copy_(state['w'])
                group['train_mode'] = True

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        for group in self.param_groups:
            lr = group['lr']
            gamma = group['gamma']
            adaptive_gamma = group['adaptive_gamma']
            k = group['k']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if 'w' not in state:
                    state['w'] = torch.clone(p, memory_format=torch.preserve_format) # 真のパラメータ
                    state['r'] = torch.clone(p, memory_format=torch.preserve_format) # 平均パラメータ
                    state['grad_r'] = torch.zeros_like(p, memory_format=torch.preserve_format) # 平均パラメータでの勾配

                w = state['w']
                r = state['r']
                grad_r = state['grad_r']

                # 平均パラメータでの勾配を更新
                tmp = 1 / k if k != 0 else 1
                grad_r.mul_(1 - tmp).add_(grad, alpha=tmp)

                # gammaを適応的に更新
                if adaptive_gamma:
                    gamma = 1 / (1 + ((r - p) ** 2).sum())

                # 真のパラメータを更新
                tmp = (1 - gamma) * grad_r + gamma * grad
                p.sub_(tmp, alpha=lr)
                w.copy_(p)

                # 平均パラメータを更新
                tmp = 1 / (k + 1)
                r.mul_(1 - tmp).add_(w, alpha = tmp)

            group['k'] = k + 1

        return loss