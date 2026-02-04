import torch
from deguc.core.stats import global_stats

class OnlineExpertClustering:
    def __init__(self, target_group_size=4, min_activation_rate=0.001, max_groups=64):
        self.target_group_size = target_group_size
        self.min_activation_rate = min_activation_rate
        self.max_groups = max_groups

    def cluster(self, compression_module):
        active_experts = []
        feat_list = []
        for k in list(compression_module.expert_residuals.keys()):
            if k.endswith("_A"):
                e = int(k[:-2])
                rate = global_stats.activation_rate(e)
                if rate < self.min_activation_rate:
                    continue
                A = compression_module.expert_residuals[k].data
                feat = A.mean(dim=1)
                feat = feat / (feat.norm() + 1e-9)
                active_experts.append(e)
                feat_list.append(feat)
        if not active_experts:
            return compression_module.group_expert_map

        feats = torch.stack(feat_list)
        N = feats.shape[0]
        est_groups = max(1, min(self.max_groups, N // self.target_group_size))
        centroids = feats[:est_groups].clone()
        assign = torch.zeros(N, dtype=torch.long)
        for _ in range(5):
            sims = torch.matmul(feats, centroids.t())
            assign = sims.argmax(dim=-1)
            for g in range(est_groups):
                idx = (assign == g)
                if idx.sum() == 0: continue
                cent = feats[idx].mean(dim=0)
                centroids[g] = cent / (cent.norm() + 1e-9)
        new_map = {}
        for g in range(est_groups):
            exps = [active_experts[i] for i in range(N) if assign[i].item() == g]
            new_map[g] = exps
        all_experts_current = set(
            e for exps in compression_module.group_expert_map.values() for e in exps
        )
        inactive = all_experts_current - set(active_experts)
        if inactive:
            if est_groups - 1 in new_map:
                new_map[est_groups - 1].extend(list(inactive))
            else:
                new_map[est_groups - 1] = list(inactive)
        global_stats.compute_group_stability(new_map)
        return new_map