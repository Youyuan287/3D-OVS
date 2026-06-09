"""
This module defines the SemanticNAG class, which constructs a hierarchical superpoint graph with semantic features.
It builds upon the Nested Adjacency Graph (NAG) structure from the SPT (Superpoint Transformer) library (https://arxiv.org/abs/2306.08045).
"""
import torch
from typing import List, Optional
import sys
sys.path.append('ext/')
from spt.data import Data, NAG, Cluster
import torch.nn.functional as F
# from ext.spt.data import Data, NAG, Cluster

class SemanticNAG():
    def __init__(self, labels, feat):
        labels = [label.cuda() for label in labels]
        self.labels = labels # [4, N]
        self.nag = self.build_nag_from_multilevel_labels(labels)
        self.feat = feat
        self.gaussian_num = labels[0].shape[0]
        self.child_to_parent, self.parent_to_children = self.build_parent_child_maps(labels)

    # sim in, gaussian_num * 1 out, indicating the 0-1 mask
    def get_related_gaussian(self, sim: List[torch.Tensor], topk: int = 1, level: int = -1) -> torch.Tensor:
        """
        sim: List of Tensors, each shape like superpoint
        topk: int, number of top similar points to consider
        level: int, choose certain level to get the related gaussian, or -1 to get all levels
        """
        assert len(sim) == len(self.labels) - 1, "Number of similarity matrices must match number of levels"
        if level == -1:
            lvls = [i for i in range(1, len(sim))]
        elif isinstance(level, list):
            lvls = [i - 1 for i in level]
        else:
            lvls = [level - 1]
        
        # 1. find the topk similar points in sim list, get tuple (level, sim_val, index)
        related_sp_lvl = []
        for i in lvls:
            sim_array = sim[i]
            sim_val, indices = torch.topk(sim_array, topk)
            related_sp_lvl.extend(list(zip([i+1] * topk, sim_val, indices)))
        # sort the list based on sim_val, descending order
        related_sp_lvl.sort(key=lambda x: x[1], reverse=True)
        # get the topk similar points
        related_sp_lvl = related_sp_lvl[:topk]
        
        # 2. get the related gaussian
        rel_gaussians = torch.zeros(self.gaussian_num, 1, dtype=torch.float32)
        for i, tup in enumerate(related_sp_lvl):
            level, _, index = tup
            lowest_idx = torch.where(self.labels[level] == index)[0]
            rel_gaussians[lowest_idx, 0] = 1
        return rel_gaussians

    def get_hierarchy_features(
        self,
        parent_weight: float = 0.25,
        residual_weight: float = 0.15,
        gate_center: float = 0.2,
        gate_temp: float = 0.07,
    ) -> List[torch.Tensor]:
        """
        Return hierarchy-aware features for levels 1..3. The top level is kept
        unchanged; lower levels receive gated parent context plus residual detail.
        """
        enhanced = [F.normalize(feat, p=2, dim=-1) for feat in self.feat]
        for level in range(len(enhanced) - 1, 0, -1):
            child_feat = enhanced[level - 1]
            parent_feat = enhanced[level]
            child_to_parent = self.child_to_parent.get(level)
            if child_to_parent is None:
                continue

            child_to_parent = child_to_parent.to(child_feat.device)
            parent_for_child = parent_feat[child_to_parent.long()]
            cos = (child_feat * parent_for_child).sum(dim=-1, keepdim=True)
            gate = torch.sigmoid((cos - gate_center) / max(gate_temp, 1e-6))
            residual = child_feat - parent_for_child
            enhanced[level - 1] = F.normalize(
                child_feat
                + parent_weight * gate * parent_for_child
                + residual_weight * residual,
                p=2,
                dim=-1,
            )
        return enhanced

    def get_soft_gaussian_scores(
        self,
        sim_full: List[torch.Tensor],
        sim_terms: Optional[List[List[torch.Tensor]]] = None,
        levels: List[int] | int = (1, 2, 3),
        topm_per_level: int = 24,
        temperature: float = 0.07,
        phrase_weight: float = 1.0,
        term_weight: float = 0.35,
        delta_weight: float = 0.5,
    ):
        """
        Convert per-superpoint similarities into a continuous Gaussian score field.
        Returns (scores [N, 1], debug dict). Levels are 1-based feature/label levels.
        """
        assert len(sim_full) == len(self.labels) - 1, "Similarity levels must match features"
        if isinstance(levels, int):
            levels = [levels]

        base_scores = []
        term_scores = []
        for idx, full in enumerate(sim_full):
            full = full.flatten()
            if sim_terms:
                terms_at_level = torch.stack([term[idx].flatten().to(full.device) for term in sim_terms], dim=0)
                term_score = terms_at_level.max(dim=0).values
            else:
                term_score = torch.zeros_like(full)
            term_scores.append(term_score)
            base_scores.append(phrase_weight * full + term_weight * term_score)

        candidates = []
        debug_candidates = []
        for level in levels:
            sim_idx = level - 1
            if sim_idx < 0 or sim_idx >= len(base_scores):
                continue
            score = base_scores[sim_idx]
            delta = torch.zeros_like(score)
            child_to_parent = self.child_to_parent.get(level)
            if child_to_parent is not None and sim_idx + 1 < len(base_scores):
                parent_ids = child_to_parent.to(score.device).long()
                parent_score = base_scores[sim_idx + 1][parent_ids]
                delta = torch.relu(score - parent_score)
                score = score + delta_weight * delta

            k = min(topm_per_level, int(score.numel()))
            vals, indices = torch.topk(score, k)
            for val, index in zip(vals, indices):
                candidates.append((level, index, val))
                debug_candidates.append({
                    "level": int(level),
                    "superpoint_id": int(index.detach().cpu()),
                    "score": float(val.detach().cpu()),
                    "full": float(sim_full[sim_idx].flatten()[index].detach().cpu()),
                    "term": float(term_scores[sim_idx][index].detach().cpu()),
                    "delta": float(delta[index].detach().cpu()),
                })

        if not candidates:
            scores = torch.zeros(self.gaussian_num, 1, dtype=torch.float32, device=self.labels[0].device)
            return scores, {"routes": [], "confidence": 0.0, "top_prob": 0.0, "second_prob": 0.0, "entropy": 0.0}

        route_scores = torch.stack([cand[2] for cand in candidates]).float()
        probs = torch.softmax(route_scores / max(temperature, 1e-6), dim=0)
        probs_cpu = probs.detach().cpu()
        sorted_probs, _ = torch.sort(probs_cpu, descending=True)
        top_prob = float(sorted_probs[0])
        second_prob = float(sorted_probs[1]) if sorted_probs.numel() > 1 else 0.0
        entropy = float(-(probs_cpu * torch.log(probs_cpu + 1e-8)).sum())
        norm_entropy = entropy / max(float(torch.log(torch.tensor(float(probs_cpu.numel())))), 1e-6)
        confidence = 0.5 * (top_prob - second_prob) + 0.5 * (1.0 - norm_entropy)

        gaussian_scores = torch.zeros(self.gaussian_num, 1, dtype=torch.float32, device=self.labels[0].device)
        for route_prob, (level, index, _) in zip(probs, candidates):
            mask = self.labels[level] == index.to(self.labels[level].device)
            gaussian_scores[mask, 0] += route_prob.to(gaussian_scores.device)

        max_score = gaussian_scores.max()
        if max_score > 0:
            gaussian_scores = gaussian_scores / max_score

        for route, prob in zip(debug_candidates, probs_cpu):
            route["prob"] = float(prob)
        debug_candidates.sort(key=lambda x: x["prob"], reverse=True)
        debug = {
            "routes": debug_candidates[: min(20, len(debug_candidates))],
            "confidence": float(confidence),
            "top_prob": top_prob,
            "second_prob": second_prob,
            "entropy": entropy,
            "normalized_entropy": float(norm_entropy),
            "max_gaussian_score": float(max_score.detach().cpu()) if torch.is_tensor(max_score) else float(max_score),
        }
        return gaussian_scores, debug
    
    def get_gs_by_mask(self, mask, level):
        """
        mask: N, boolean mask
        level: int, choose certain level to get the related gaussian
        """
        assert level > 0 and level <= len(self.labels), "Level out of range"
        # get the sub from NAG, iterative down to level 0
        label = self.labels[level]
        sps = label[mask]
        # get gaussian in the sps
        rel_gaussians = torch.zeros(self.gaussian_num, 1, dtype=torch.float32)
        for sp in sps:
            lowest_idx = torch.where(label == sp)[0]
            rel_gaussians[lowest_idx, 0] = 1
        return rel_gaussians

    @staticmethod
    def build_parent_child_maps(labels: List[torch.Tensor]):
        child_to_parent = {}
        parent_to_children = {}
        for child_level in range(1, len(labels) - 1):
            parent_level = child_level + 1
            child = labels[child_level].long()
            parent = labels[parent_level].long()
            num_children = int(child.max().item()) + 1
            mapping = torch.full((num_children,), -1, dtype=torch.long, device=child.device)
            for child_id in torch.unique(child):
                parent_ids = parent[child == child_id].unique()
                if parent_ids.numel() != 1:
                    raise ValueError(
                        f"Ambiguous parent mapping for level {child_level} child {int(child_id)}"
                    )
                mapping[child_id.long()] = parent_ids[0].long()
            child_to_parent[child_level] = mapping

            parents = {}
            for child_id, parent_id in enumerate(mapping.detach().cpu().tolist()):
                parents.setdefault(parent_id, []).append(child_id)
            parent_to_children[child_level] = parents
        return child_to_parent, parent_to_children

    
    @staticmethod
    def build_nag_from_multilevel_labels(labels: List[torch.Tensor]) -> NAG:
        """
        labels: List of Tensors, each shape = (N,), level-0 point to higher level cluster IDs
            e.g. [label_lvl0, label_lvl1, label_lvl2]
        pos / feat: Optional, only used at level 0
        """
        assert len(labels) >= 2, "At least two levels required to construct NAG"
        device = labels[0].device
        N = labels[0].shape[0]
        num_levels = len(labels)

        data_list = [Data(num_nodes=N, super_index=labels[0])]
        prev_sub = None

        for i in range(num_levels):
            data = Data()
            # 求每层的sub和上一层的super_index
            if i == 0:
                upper_labels = labels[i]
                lower_labels = torch.arange(N, device=device)
                sorted_upper_labels, perm = torch.sort(upper_labels)
                sorted_lower_labels = lower_labels[perm]
                num_clusters = int(sorted_upper_labels.max()) + 1
                cluster_sizes = torch.bincount(sorted_upper_labels, minlength=num_clusters)
                pointers = torch.zeros(num_clusters + 1, dtype=torch.long, device=device)
                pointers[1:] = torch.cumsum(cluster_sizes, dim=0)
                prev_sub = Cluster(pointers=pointers, points=sorted_lower_labels, dense=False)
            else:
                data = Data(num_nodes=labels[i-1].max().item() + 1)
                data.sub = prev_sub
                upper_labels = labels[i].long()
                lower_labels = labels[i-1].long()

                # 1. remove duplicates based on lower labels
                unique_lower_labels, inv = torch.unique(lower_labels, sorted=True, return_inverse=True)
                unique_upper_labels = torch.zeros_like(unique_lower_labels)
                unique_upper_labels[inv] = upper_labels
                # print('uni:', lower_labels, unique_lower_labels, unique_upper_labels, inv)

                data.super_index = unique_upper_labels
                # 2. sort based on upper labels, construct next sub
                sorted_upper_labels, perm = torch.sort(unique_upper_labels)
                sorted_lower_labels = unique_lower_labels[perm]
                # print("sort:", sorted_lower_labels, sorted_upper_labels)
                # upper label take down the change
                num_clusters = int(sorted_upper_labels.max()) + 1
                cluster_sizes = torch.bincount(sorted_upper_labels, minlength=num_clusters)
                pointers = torch.zeros(num_clusters + 1, dtype=torch.long, device=device)
                pointers[1:] = torch.cumsum(cluster_sizes, dim=0)
                # print('csr:', pointers, sorted_lower_labels)
                prev_sub = Cluster(pointers=pointers, points=sorted_lower_labels, dense=False)
                data_list.append(data)
        data_list.append(Data(sub=prev_sub, num_nodes=labels[-1].max().item() + 1))
        return NAG(data_list)

if __name__ == '__main__':
    lvl1 = torch.tensor([4, 5, 2, 3, 1, 5, 0, 3])
    lvl2 = torch.tensor([3, 4, 3, 2, 1, 4, 0, 2])
    lvl3 = torch.tensor([1, 2, 1, 1, 0, 2, 0, 1])
    nag = SemanticNAG.build_nag_from_multilevel_labels([lvl1, lvl2, lvl3])
    print(nag, nag[1].sub, 777)
    print(nag[3].sub[1].points)
    pt = 1
    for i in range(3, 0, -1):
        pt = nag[i].sub[pt].points
        print(55, pt)
    print(nag.get_super_index(1, 0))
    print(nag.get_super_index(2, 0))
    print(nag.get_super_index(3, 0))
    print(nag.get_super_index(2, 1))
    # print(from_super_index(lvl2, lvl1))
    # cls = Cluster(pointers=torch.tensor([0, 3, 5, 8]), points=torch.tensor([2, 0, 1, 3, 4, 5, 6, 7]))
    # print(cls, cls.pointers)
