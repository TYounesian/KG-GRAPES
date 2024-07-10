import torch

from models import RGCN2, Mini_Batch_RGCN, LADIES_Mini_Batch_ERGCN
from utils import *


class MRGCN_Batch(nn.Module):
    def __init__(self, n, feat_size, embed_size, modality, num_classes, num_rels, num_bases, sampler, depth):
        super().__init__()

        self.num_nodes = n
        self.num_classes = num_classes
        self.embed_size = embed_size
        self.modality = modality
        self.depth = depth
        self.sampler = sampler
        self.num_rels = num_rels
        self.batch_rgcn = LADIES_Mini_Batch_ERGCN(n, feat_size, embed_size, num_classes, num_rels, num_bases, sampler)

    def forward(self, embed_X, A_en_sliced, after_nodes_list, idx_per_rel_list, nonzero_rel_list, device):
        out, nodes_in_rels, = self.sampler_forward(embed_X, A_en_sliced, after_nodes_list,
                                                    idx_per_rel_list, nonzero_rel_list, device)
        return out, nodes_in_rels

    def sampler_forward(self, embed_X, A_en_sliced, after_nodes, idx_per_rel_list,
                        nonzero_rel_list,
                        device):
        if type(after_nodes) == list:
            if len(after_nodes) > 0:
                em_X = embed_X[after_nodes[0]]
            else:
                em_X = embed_X
        else:
            em_X = embed_X[after_nodes]

        em_X_dev = None if em_X is None else em_X.to(device)
        # A_en_sliced = [i.to(device) for i in A_en_sliced]

        self.batch_rgcn.to(device)

        if self.sampler == "LDRN" or self.sampler == "LDRE" or self.sampler =='LRUN':
            out = self.batch_rgcn(em_X_dev, A_en_sliced, 'full', idx_per_rel_list, nonzero_rel_list)
        else:
            out = self.batch_rgcn(em_X_dev, A_en_sliced, 'full', [], [])

        return out, 0


class MRGCN_Full(nn.Module):
    def __init__(self, n, edges, feat_size, embed_size, modality, num_classes, num_rels, num_bases, self_loop_dropout):
        super().__init__()

        self.num_nodes = n
        self.num_classes = num_classes
        self.embed_size = embed_size
        self.feat_size = feat_size
        self.modality = modality
        self.num_rels = num_rels

        self.rgcn = RGCN2(n, edges, feat_size, embed_size, num_classes, num_rels, num_bases, self_loop_dropout)

    def forward(self, embed_X):
        # Sum up the init embeddings with cnn embeddings
        out = self.rgcn(embed_X)  # A is calculated inside

        return out
