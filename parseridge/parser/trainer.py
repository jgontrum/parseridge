import torch

from torch.optim import Adam

from parseridge.parser.modules.data_parallel import Module
from parseridge.utils.logger import LoggerMixin


class Trainer(LoggerMixin):
    def __init__(
        self,
        model: Module,
        learning_rate=1e-3,
        weight_decay=0.00,
        loss_factor=0.75,
        update_size=50,
        gradient_clipping=10,
        mode="avg",
    ):

        self.supported_modes = ["avg", "sum"]
        self.optimizer = Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        self.model = model
        self.loss_factor = loss_factor
        self.update_size = update_size
        self.mode = mode
        self.gradient_clipping = gradient_clipping

        assert self.mode in self.supported_modes

    def learn(self, loss, metric):
        if len(loss) >= self.update_size:
            if self.mode == "avg":
                batch_loss = sum(loss) / len(loss) * self.loss_factor
            elif self.mode == "sum":
                batch_loss = sum(loss) * self.loss_factor
            else:
                raise NotImplementedError(f"Only supporting {self.supported_modes}.")

            batch_loss.backward()

            if self.gradient_clipping > 0:
                torch.nn.utils.clip_grad_value_(
                    self.model.parameters(), self.gradient_clipping
                )

            self.optimizer.step()
            self.optimizer.zero_grad()

            metric.loss += batch_loss.item() + len(loss)
            metric.num_backprop += 1

            loss = []

        return loss, metric
