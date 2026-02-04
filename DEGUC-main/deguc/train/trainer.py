import torch
import torch.optim as optim
from deguc.core.logging import SimpleLogger
from deguc.core.stats import global_stats
from deguc.clustering.online_clustering import OnlineExpertClustering

class DEGUCTrainer:
    def __init__(self, model, lr=1e-3, device=None, logger: SimpleLogger=None,
                 clustering_interval=200, quant_interval=400, offload_interval=300,
                 balance_loss_weight=0.01):
        self.model = model
        self.device = device or torch.device("cpu")
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.logger = logger or SimpleLogger()
        self.iteration = 0
        self.clustering = OnlineExpertClustering()
        self.clustering_interval = clustering_interval
        self.quant_interval = quant_interval
        self.offload_interval = offload_interval
        self.balance_loss_weight = balance_loss_weight
        self.last_quant_report = None

    def dummy_task_loss(self, outputs):
        return (outputs ** 2).mean()

    def training_step(self, batch):
        self.iteration += 1
        self.model.train()
        inp = batch.to(self.device)
        out, balance_loss, aux = self.model(inp)
        task_loss = self.dummy_task_loss(out)
        total_loss = task_loss + self.balance_loss_weight * balance_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        if self.iteration % self.clustering_interval == 0:
            new_map = self.clustering.cluster(self.model.compression)
            self.model.update_groups(new_map)
            global_stats.clustering_round += 1

        if self.iteration % self.quant_interval == 0:
            self.last_quant_report = self.model.apply_quantization()

        if self.iteration % self.offload_interval == 0:
            self.model.offload_inactive()

        if self.iteration % 50 == 0:
            metrics = {
                "iter": self.iteration,
                "task_loss": task_loss.item(),
                "balance_loss": balance_loss.item(),
                "total_loss": total_loss.item(),
                "groups": len(self.model.router.group_expert_map),
                "offloaded": global_stats.offloaded_experts,
                "reloaded": global_stats.reloaded_experts,
                "clustering_round": global_stats.clustering_round,
                "group_stability": global_stats.group_stability_score
            }
            if self.last_quant_report:
                metrics.update({
                    "quant_ratio": self.last_quant_report["ratio"],
                    "quant_MB": self.last_quant_report["quant_MB"]
                })
            self.logger.log(**metrics)

    def finish(self):
        self.logger.dump()