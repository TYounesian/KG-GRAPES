import numpy as np
import torch
import pdb
import torch.nn as Nnn
import torch.nn.functional as F
from utils import *
from PIL import ImageOps
import transformers as tf
from torchvision import transforms
import time

class RGCN(nn.Module):  # classic rgcn
    def __init__(self, n, edges, feat_size, embed_size, num_classes, num_rels, num_bases):
        super().__init__()
        # pdb.set_trace()
        self.embed_size = embed_size
        self.num_classes = num_classes
        self.num_bases = num_bases
        self.num_rels = num_rels
        # Add self-loops and inverses
        edges = enrich(edges, n, int((self.num_rels-1)/2))

        # horizontal and vertical indices and sizes
        hor_ind, hor_size = adj(edges, n, self.num_rels, vertical=False)
        ver_ind, ver_size = adj(edges, n, self.num_rels, vertical=True)

        _, rn = hor_size
        num_rels = rn // n

        vals = torch.ones(ver_ind.size(0), dtype=torch.float)
        vals = vals / sum_sparse(ver_ind, vals, ver_size, True)

        # constructing the horizontal and vertical Adj matrices
        horizontalA = torch.sparse.FloatTensor(indices=hor_ind.t(), values=vals, size=hor_size)
        self.register_buffer('horizontalA', horizontalA)

        verticalA = torch.sparse.FloatTensor(indices=ver_ind.t(), values=vals, size=ver_size)
        self.register_buffer('verticalA', verticalA)

        # layer 1 init weights
        if num_bases is None:
            self.w1 = nn.Parameter(torch.FloatTensor(num_rels, feat_size, embed_size))
            nn.init.xavier_uniform_(self.w1, gain=nn.init.calculate_gain('relu'))
            self.bases1 = None

        else:
            self.comps1 = nn.Parameter(torch.FloatTensor(num_rels, num_bases))
            nn.init.xavier_uniform_(self.comps1, gain=nn.init.calculate_gain('relu'))

            self.bases1 = nn.Parameter(torch.FloatTensor(num_bases, feat_size, embed_size))
            nn.init.xavier_uniform_(self.bases1, gain=nn.init.calculate_gain('relu'))

        # layer 2 init weights
        if num_bases is None:

            self.w2 = nn.Parameter(torch.FloatTensor(num_rels, embed_size, num_classes))
            nn.init.xavier_uniform_(self.w2, gain=nn.init.calculate_gain('relu'))
            self.bases2 = None

        else:
            self.comps2 = nn.Parameter(torch.FloatTensor(num_rels, num_bases))
            nn.init.xavier_uniform_(self.comps2, gain=nn.init.calculate_gain('relu'))

            self.bases2 = nn.Parameter(torch.FloatTensor(num_bases, embed_size, num_classes))
            nn.init.xavier_uniform_(self.bases2, gain=nn.init.calculate_gain('relu'))

        self.bias1 = nn.Parameter(torch.FloatTensor(embed_size).zero_())
        self.bias2 = nn.Parameter(torch.FloatTensor(num_classes).zero_())

    def forward(self, features):
        ## Layer 1
        n, rn = self.horizontalA.size()
        num_rels = rn // n
        e = self.embed_size
        b, c = self.num_bases, self.num_classes
        _, f = features.size()

        h = torch.mm(self.verticalA, features)
        h = h.view(num_rels, n, f)
        if self.bases1 is None:
            w = self.w1
        else:
            w = torch.einsum('rb, bij -> rij', self.comps1, self.bases1)

        # Apply weights and sum over relations
        h = torch.einsum('rnf, rfe -> ne', h, w)
        # Add bias and activation
        h = F.relu(h + self.bias1)

        # Layer 2
        h = torch.mm(self.verticalA, h)
        h = h.view(num_rels, n, e)

        if self.bases2 is None:
            w = self.w2
        else:
            w = torch.einsum('rb, bho -> rho', self.comps2, self.bases2)

        # Multiplying the weights with hidden via einsum
        h = torch.einsum('rho, rnh -> no', w, h)

        return h + self.bias2

    def penalty(self, p=2):

        if self.num_bases is None:
            return self.w1.pow(p).sum()

        return self.comps1.pow(p).sum() + self.bases1.pow(p).sum()


class RGCN2(nn.Module):
    """
    We use a classic RGCN, with embeddings as inputs (instead of the one-hot inputs of rgcn.py)

    """

    def __init__(self, n, edges, feat_size, embed_size, numcls, num_rels, bases, self_loop_dropout):

        super().__init__()

        self.feat_size = feat_size
        self.embed_size = embed_size
        self.bases = bases
        self.numcls = numcls
        self.edges = enrich_with_drop(edges, self_loop_dropout, n, int((num_rels - 1) / 2))

        # horizontally and vertically stacked versions of the adjacency graph
        hor_ind, hor_size = adj(self.edges, n, num_rels, vertical=False)
        ver_ind, ver_size = adj(self.edges, n, num_rels, vertical=True)

        _, rn = hor_size
        r = rn // n

        vals = torch.ones(ver_ind.size(0), dtype=torch.float)
        vals = vals / sum_sparse(ver_ind, vals, ver_size, True)

        hor_graph = torch.sparse.FloatTensor(indices=hor_ind.t(), values=vals, size=hor_size)
        self.register_buffer('hor_graph', hor_graph)

        ver_graph = torch.sparse.FloatTensor(indices=ver_ind.t(), values=vals, size=ver_size)
        self.register_buffer('ver_graph', ver_graph)

        # layer 1 weights
        if bases is None:
            self.weights1 = nn.Parameter(torch.FloatTensor(r, feat_size, embed_size))
            nn.init.xavier_uniform_(self.weights1, gain=nn.init.calculate_gain('relu'))

            self.bases1 = None
        else:
            self.comps1 = nn.Parameter(torch.FloatTensor(r, bases))
            nn.init.xavier_uniform_(self.comps1, gain=nn.init.calculate_gain('relu'))

            self.bases1 = nn.Parameter(torch.FloatTensor(bases, feat_size, embed_size))
            nn.init.xavier_uniform_(self.bases1, gain=nn.init.calculate_gain('relu'))

        # layer 2 weights
        if bases is None:

            self.weights2 = nn.Parameter(torch.FloatTensor(r, embed_size, numcls))
            nn.init.xavier_uniform_(self.weights2, gain=nn.init.calculate_gain('relu'))

            self.bases2 = None
        else:
            self.comps2 = nn.Parameter(torch.FloatTensor(r, bases))
            nn.init.xavier_uniform_(self.comps2, gain=nn.init.calculate_gain('relu'))

            self.bases2 = nn.Parameter(torch.FloatTensor(bases, embed_size, numcls))
            nn.init.xavier_uniform_(self.bases2, gain=nn.init.calculate_gain('relu'))

        self.bias1 = nn.Parameter(torch.FloatTensor(embed_size).zero_())
        self.bias2 = nn.Parameter(torch.FloatTensor(numcls).zero_())

    def forward(self, features):
        # size of node representation per layer: f -> e -> c
        n, rn = self.hor_graph.size()
        r = rn // n
        e = self.embed_size
        b, c = self.bases, self.numcls

        n, f = features.size()
        # Layer 1
        h = torch.mm(self.ver_graph, features) # sparse mm
        h = h.view(r, n, f) # new dim for the relations

        if self.bases1 is not None:
            weights = torch.einsum('rb, bij -> rij', self.comps1, self.bases1)
            # weights = torch.mm(self.comps1, self.bases1.view(b, n*e)).view(r, n, e)
        else:
            weights = self.weights1

        assert weights.size() == (r, f, e)
        # Apply weights and sum over relations
        h = torch.bmm(h, weights).sum(dim=0)

        assert h.size() == (n, e)

        h = F.relu(h + self.bias1)
        # Layer 2
        # Multiply adjacencies by hidden
        h = torch.mm(self.ver_graph, h) # sparse mm
        h = h.view(r, n, e) # new dim for the relations

        if self.bases2 is not None:
            weights = torch.einsum('rb, bij -> rij', self.comps2, self.bases2)
            # weights = torch.mm(self.comps2, self.bases2.view(b, e * c)).view(r, e, c)
        else:
            weights = self.weights2

        # Apply weights, sum over relations
        # h = torch.einsum('rhc, rnh -> nc', weights, h)
        h = torch.bmm(h, weights).sum(dim=0)

        assert h.size() == (n, c)
        h = h + self.bias2 # -- softmax is applied in the loss

        return h

    def penalty(self, p=2):

        assert p==2

        if self.bases is None:
            return self.weights1.pow(2).sum()

        return self.comps1.pow(p).sum() + self.bases1.pow(p).sum()


class Mini_Batch_ERGCN(nn.Module):  # ergcn - rgcn with node embeddings
    def __init__(self, n, edges, embed_size, num_classes, num_rels, num_bases):
        super().__init__()

        self.num_classes = num_classes
        self.embed_size = embed_size
        self.num_rels = num_rels
        self.n = n
        self.num_bases = num_bases

        # layer 1 init weights
        if num_bases is None:
            self.w1 = nn.Parameter(torch.FloatTensor(num_rels, embed_size))
            nn.init.xavier_uniform_(self.w1, gain=nn.init.calculate_gain('relu'))

            self.bases1 = None
        else:
            self.comp1 = nn.Parameter(torch.FloatTensor(num_rels, num_bases))
            nn.init.xavier_uniform_(self.comp1, gain=nn.init.calculate_gain('relu'))

            self.bases1 = nn.Parameter(torch.FloatTensor(num_bases, embed_size, embed_size))
            nn.init.xavier_uniform_(self.bases1, gain=nn.init.calculate_gain('relu'))

        # layer 2 init weights
        if num_bases is None:

            self.w2 = nn.Parameter(torch.FloatTensor(num_rels, embed_size, num_classes))
            nn.init.xavier_uniform_(self.w2, gain=nn.init.calculate_gain('relu'))

            self.bases2 = None

        else:
            self.comp2 = nn.Parameter(torch.FloatTensor(num_rels, num_bases))
            nn.init.xavier_uniform_(self.comp2, gain=nn.init.calculate_gain('relu'))

            self.bases2 = nn.Parameter(torch.FloatTensor(num_bases, embed_size, num_classes))
            nn.init.xavier_uniform_(self.bases2, gain=nn.init.calculate_gain('relu'))

        self.bias1 = nn.Parameter(torch.FloatTensor(embed_size).zero_())
        self.bias2 = nn.Parameter(torch.FloatTensor(num_classes).zero_())

    def forward(self, X_batch, A_batch, batch_idx, A_neighbours_unseen, A,  neighbours, test_state):
        if self.training:
            neighbours_idx = neighbours[0]
            depth2neighbours_idx = neighbours[1]
            H_idx = neighbours[2]
            H_node_idx = neighbours[3]
            # if not featureless:
            # 	self.embedding = nn.Parameter(self.embedding + X)
            n, rn = A_batch.size()

            e = self.embed_size
            c = self.num_classes

            ## Layer 1
            X_batch_neighbours_idx = [i for i in range(len(batch_idx))
                                      if batch_idx[i] in neighbours_idx]
            X_batch_depth2neighbours_idx = [i for i in range(len(batch_idx))
                                            if batch_idx[i] in depth2neighbours_idx]

            # consider only the embeddings of connected nodes
            A_idx = getAdjacencyNodeColumnIdx(neighbours_idx, self.n, self.num_rels)

            A_idx = sliceSparseCOO(A_batch, self.num_rels, A_idx)  # slice sparse COO tensor

            if A_neighbours_unseen.shape[0] > 0:
                # only needed if not all nodes have been computed yet
                A_idx_depth2 = getAdjacencyNodeColumnIdx(depth2neighbours_idx, self.n, self.num_rels)
                A_idx_depth2 = sliceSparseCOO(A_neighbours_unseen, self.num_rels, A_idx_depth2)

            # print("first layer weights applying...")
            if self.bases1 is None:
                w = self.w1

                xw_dp1 = torch.einsum('ne, re -> rne', X_batch[X_batch_neighbours_idx], w).contiguous()
                xw_dp2 = torch.einsum('ne, re -> rne', X_batch[X_batch_depth2neighbours_idx], w).contiguous()

            else:
                w = torch.einsum('rb, beh -> reh', self.comp1, self.bases1)
                xw_dp1 = torch.einsum('ne, reh -> rnh', X_batch[X_batch_neighbours_idx].float(), w.float()).contiguous()
                xw_dp2 = torch.einsum('ne, reh -> rnh', X_batch[X_batch_depth2neighbours_idx].float(),
                                      w.float()).contiguous()

            # Apply weights and sum over relations
            h1_dp1 = torch.mm(A_idx.float(), xw_dp1.view(self.num_rels * len(X_batch_neighbours_idx), e).float())
            h1_dp1 = F.relu(h1_dp1 + self.bias1)

            h1_dp2 = torch.mm(A_idx_depth2.float(),
                              xw_dp2.view(self.num_rels * len(X_batch_depth2neighbours_idx), e).float())
            h1_dp2 = F.relu(h1_dp2 + self.bias1)

            h1 = torch.vstack([h1_dp1, h1_dp2]) if h1_dp2 is not None else h1_dp2

            ############### Layer 2
            if self.bases2 is None:
                w = self.w2
            else:
                w = torch.einsum('rb, bho -> rho', self.comp2, self.bases2)

            A_idx = getAdjacencyNodeColumnIdx(H_node_idx, self.n, self.num_rels)
            A_idx = sliceSparseCOO(A_batch, self.num_rels, A_idx)
            h2 = torch.einsum('nh, rhc -> rnc', h1[H_idx], w).contiguous()
            h2 = h2.view(self.num_rels * len(H_idx), c)
            h2 = torch.mm(A_idx.float(), h2.float())

            return h2 + self.bias2


    def penalty(self, p=2):

        if self.num_bases is None:
            return self.w1.pow(p).sum()

        return self.comp1.pow(p).sum() + self.bases1.pow(p).sum()


class LADIES_Mini_Batch_ERGCN(nn.Module):  # ergcn - rgcn with node embeddings
    def __init__(self, n, feat_size, embed_size, num_classes, num_rels, num_bases, sampler):
        super().__init__()

        self.num_classes = num_classes
        self.embed_size = embed_size
        self.num_rels = num_rels
        self.n = n
        self.num_bases = num_bases
        self.sampler = sampler

        # layer 1 init weights
        if num_bases is None:
            self.w1 = nn.Parameter(torch.FloatTensor(num_rels, embed_size))
            nn.init.xavier_uniform_(self.w1, gain=nn.init.calculate_gain('relu'))

            self.bases1 = None
        else:
            self.comp1 = nn.Parameter(torch.FloatTensor(num_rels, num_bases))
            nn.init.xavier_uniform_(self.comp1, gain=nn.init.calculate_gain('relu'))

            self.bases1 = nn.Parameter(torch.FloatTensor(num_bases, feat_size, embed_size))
            nn.init.xavier_uniform_(self.bases1, gain=nn.init.calculate_gain('relu'))

        # layer 2 init weights
        if num_bases is None:

            self.w2 = nn.Parameter(torch.FloatTensor(num_rels, embed_size, num_classes))
            nn.init.xavier_uniform_(self.w2, gain=nn.init.calculate_gain('relu'))

            self.bases2 = None

        else:
            self.comp2 = nn.Parameter(torch.FloatTensor(num_rels, num_bases))
            nn.init.xavier_uniform_(self.comp2, gain=nn.init.calculate_gain('relu'))

            self.bases2 = nn.Parameter(torch.FloatTensor(num_bases, embed_size, num_classes))
            nn.init.xavier_uniform_(self.bases2, gain=nn.init.calculate_gain('relu'))

        self.bias1 = nn.Parameter(torch.FloatTensor(embed_size).zero_())
        self.bias2 = nn.Parameter(torch.FloatTensor(num_classes).zero_())

    def forward(self, X_batch, after_nodes_list, nodes_needed, A_en_sliced, A, test_state,
                idx_per_rel_list=None, nonzero_rel_list=None):
        if test_state == 'LDRN' or test_state == 'full-mini' or (self.training and (self.sampler == 'LDRN'
                                                                or self.sampler == 'LDUN'
                                                                or self.sampler == 'LDRE'
                                                                or self.sampler == 'IARN'
                                                                or self.sampler == 'IDRN'
                                                                or self.sampler == 'full-mini-batch')):
            dp2neighbours_idx = after_nodes_list[0]
            dp1neighbours_idx = after_nodes_list[1]

            A_0 = A_en_sliced[0]
            A_1 = A_en_sliced[1]

            e = self.embed_size
            c = self.num_classes

            ## Layer 1
            X_batch_dp2neighbours_idx = [i for i in range(len(nodes_needed))
                                      if nodes_needed[i] in dp2neighbours_idx]

            if self.bases1 is None:
                w = self.w1

                xw_dp1 = torch.einsum('ne, re -> rne', X_batch[X_batch_dp2neighbours_idx], w).contiguous()

            else:
                w = torch.einsum('rb, beh -> reh', self.comp1, self.bases1)
                xw_dp1 = torch.einsum('ne, reh -> rnh', X_batch[X_batch_dp2neighbours_idx].float(), w.float()).contiguous()

            h1_dp1 = torch.mm(A_0.float(), (xw_dp1.view(self.num_rels * len(dp2neighbours_idx), e).float()))
            h1 = F.relu(h1_dp1 + self.bias1)

            # Layer 2
            if self.bases2 is None:
                w = self.w2
            else:
                w = torch.einsum('rb, bho -> rho', self.comp2, self.bases2)

            h2 = torch.einsum('nh, rhc -> rnc', h1, w).contiguous()
            h2 = h2.view(self.num_rels * len(dp1neighbours_idx), c)
            h2 = torch.mm(A_1.float(), h2.float())
            h2 = h2 + self.bias2

            return h2

        if test_state == 'LDRN' or test_state == 'full-mini':
            dp2neighbours_idx = after_nodes_list[0]
            dp1neighbours_idx = after_nodes_list[1]

            idx_per_rel_dp2 = idx_per_rel_list[0]
            idx_per_rel_dp1 = idx_per_rel_list[1]
            A_0 = A_en_sliced[0]
            A_1 = A_en_sliced[1]

            e = self.embed_size
            c = self.num_classes

            ## Layer 1
            X_batch_dp2neighbours_idx = [i for i in range(len(nodes_needed))
                                      if nodes_needed[i] in dp2neighbours_idx]

            if self.bases1 is None:
                w = self.w1
                xw_dp1 = torch.einsum('ne, re -> rne', X_batch[X_batch_dp2neighbours_idx], w).contiguous()

            else:
                w = torch.einsum('rb, beh -> reh', self.comp1, self.bases1)
                xw_dp1 = torch.einsum('ne, reh -> rnh', X_batch[X_batch_dp2neighbours_idx].float(), w.float()).contiguous()

            if self.sampler == 'LDRE':
                xw_dp1_flat = xw_dp1.view(self.num_rels * len(dp2neighbours_idx), e)
                xw_dp1_samp_cat = torch.empty(A_0.size()[1],e)
                offset = 0
                for r in range(self.num_rels):
                    ri = (nonzero_rel_list[-1] == r).nonzero(as_tuple=True)[0]
                    if len(ri) > 0:
                        local_idx_in_dp2neigh, _ = (dp2neighbours_idx.unsqueeze(1) == idx_per_rel_dp2[ri]).nonzero(as_tuple=True)
                        xw_dp1_samp = xw_dp1_flat[r*len(dp2neighbours_idx)+local_idx_in_dp2neigh, :]
                        xw_dp1_samp_cat[offset:offset+len(idx_per_rel_dp2[ri])] = xw_dp1_samp
                        offset += len(idx_per_rel_dp2[ri])
                h1_dp1 = torch.mm(A_0.float(), xw_dp1_samp_cat)

            h1 = F.relu(h1_dp1 + self.bias1)
            ############### Layer 2

            if self.bases2 is None:
                w = self.w2
            else:
                w = torch.einsum('rb, bho -> rho', self.comp2, self.bases2)

            h2 = torch.einsum('nh, rhc -> rnc', h1, w).contiguous()
            h2 = h2.view(self.num_rels * len(dp1neighbours_idx), c)
            h2 = torch.mm(A_1.float(), h2.float())
            h2 = h2 + self.bias2

            return h2

        elif test_state == 'full' and not self.training:
            n, rn = A.size()
            num_rels = rn // n
            e = self.embed_size
            c = self.num_classes

            # Layer 1
            if self.bases1 is None:
                w = self.w1
                xw = torch.einsum('ne, re -> rne', X_batch, w).contiguous()
            else:
                w = torch.einsum('rb, beh -> reh', self.comp1, self.bases1)
                xw = torch.einsum('ne, reh -> rnh', X_batch, w).contiguous()

            # Apply weights and sum over relations
            h1 = torch.mm(A, xw.view(num_rels * n, e))
            h1 = F.relu(h1 + self.bias1)

            # Layer 2
            if self.bases2 is None:
                w = self.w2
            else:
                w = torch.einsum('rb, bho -> rho', self.comp2, self.bases2)

            h2 = torch.einsum('nh, rhc -> rnc', h1, w).contiguous()
            h2 = h2.view(self.num_rels * n, c)
            h2 = torch.mm(A, h2)

            return h2 + self.bias2

    def penalty(self, p=2):

        if self.num_bases is None:
            return self.w1.pow(p).sum()

        return self.comp1.pow(p).sum() + self.bases1.pow(p).sum()


class Mini_Batch_RGCN(nn.Module):  # ergcn - rgcn with node embeddings
    def __init__(self, n, feat_size, embed_size, num_classes, num_rels, num_bases):
        super().__init__()

        self.num_classes = num_classes
        self.embed_size = embed_size
        self.num_rels = num_rels
        self.n = n
        self.num_bases = num_bases

        # layer 1 init weights
        if num_bases is None:
            self.w1 = nn.Parameter(torch.FloatTensor(num_rels, embed_size))
            nn.init.xavier_uniform_(self.w1, gain=nn.init.calculate_gain('relu'))

            self.bases1 = None
        else:
            self.comp1 = nn.Parameter(torch.FloatTensor(num_rels, num_bases))
            nn.init.xavier_uniform_(self.comp1, gain=nn.init.calculate_gain('relu'))

            self.bases1 = nn.Parameter(torch.FloatTensor(num_bases, feat_size, embed_size))
            nn.init.xavier_uniform_(self.bases1, gain=nn.init.calculate_gain('relu'))

        # layer 2 init weights
        if num_bases is None:

            self.w2 = nn.Parameter(torch.FloatTensor(num_rels, embed_size, num_classes))
            nn.init.xavier_uniform_(self.w2, gain=nn.init.calculate_gain('relu'))

            self.bases2 = None

        else:
            self.comp2 = nn.Parameter(torch.FloatTensor(num_rels, num_bases))
            nn.init.xavier_uniform_(self.comp2, gain=nn.init.calculate_gain('relu'))

            self.bases2 = nn.Parameter(torch.FloatTensor(num_bases, embed_size, num_classes))
            nn.init.xavier_uniform_(self.bases2, gain=nn.init.calculate_gain('relu'))

        self.bias1 = nn.Parameter(torch.FloatTensor(embed_size).zero_())
        self.bias2 = nn.Parameter(torch.FloatTensor(num_classes).zero_())

    def forward(self, X_batch, A_batch, batch_idx, A_neighbours_unseen, A, neighbours, test_state):
        if self.training:
            neighbours_idx = neighbours[0]
            depth2neighbours_idx = neighbours[1]
            H_idx = neighbours[2]
            H_node_idx = neighbours[3]
            # if not featureless:
            # 	self.embedding = nn.Parameter(self.embedding + X)
            e = self.embed_size
            c = self.num_classes

            ## Layer 1
            print("slicing depth1 first layer")
            X_batch_neighbours_idx = [i for i in range(len(batch_idx))
                                      if batch_idx[i] in neighbours_idx]
            X_batch_depth2neighbours_idx = [i for i in range(len(batch_idx))
                                            if batch_idx[i] in depth2neighbours_idx]

            # consider only the embeddings of connected nodes
            A_idx = getAdjacencyNodeColumnIdx(neighbours_idx, self.n, self.num_rels)

            A_idx = sliceSparseCOO(A_batch, self.num_rels, A_idx)  # slice sparse COO tensor

            if A_neighbours_unseen.shape[0] > 0:
                # only needed if not all nodes have been computed yet
                A_idx_depth2 = getAdjacencyNodeColumnIdx(depth2neighbours_idx, self.n, self.num_rels)
                A_idx_depth2 = sliceSparseCOO(A_neighbours_unseen, self.num_rels, A_idx_depth2)

            print("first layer weights applying...")
            if self.bases1 is None:
                w = self.w1

                xw_dp1 = torch.einsum('ne, re -> rne', X_batch[X_batch_neighbours_idx], w).contiguous()
                xw_dp2 = torch.einsum('ne, re -> rne', X_batch[X_batch_depth2neighbours_idx], w).contiguous()

            else:
                w = torch.einsum('rb, beh -> reh', self.comp1, self.bases1)
                xw_dp1 = torch.einsum('ne, reh -> rnh', X_batch[X_batch_neighbours_idx].float(), w.float()).contiguous()
                xw_dp2 = torch.einsum('ne, reh -> rnh', X_batch[X_batch_depth2neighbours_idx].float(),
                                      w.float()).contiguous()


            # Apply weights and sum over relations
            h1_dp1 = torch.mm(A_idx.float(), xw_dp1.view(self.num_rels * len(X_batch_neighbours_idx), e).float())
            h1_dp1 = F.relu(h1_dp1)

            h1_dp2 = torch.mm(A_idx_depth2.float(),
                              xw_dp2.view(self.num_rels * len(X_batch_depth2neighbours_idx), e).float())
            h1_dp2 = F.relu(h1_dp2)

            h1 = torch.vstack([h1_dp1, h1_dp2]) if h1_dp2 is not None else h1_dp2

            ############### Layer 2
            if self.bases2 is None:
                w = self.w2
            else:
                w = torch.einsum('rb, bho -> rho', self.comp2, self.bases2)

            A_idx = getAdjacencyNodeColumnIdx(H_node_idx, self.n, self.num_rels)
            A_idx = sliceSparseCOO(A_batch, self.num_rels, A_idx)  # slice sparse COO tensor

            h2 = torch.einsum('nh, rhc -> rnc', h1[H_idx], w).contiguous()
            h2 = h2.view(self.num_rels * len(H_idx), c)
            h2 = torch.mm(A_idx.float(), h2.float())

            return h2 + self.bias2

        elif test_state == 'full' and not self.training:
            n, rn = A.size()
            num_rels = rn // n
            e = self.embed_size
            c = self.num_classes

            ## Layer 1
            if self.bases1 is None:
                w = self.w1
                xw = torch.einsum('ne, re -> rne', X_batch, w).contiguous()
            else:
                w = torch.einsum('rb, beh -> reh', self.comp1, self.bases1)
                xw = torch.einsum('ne, reh -> rnh', X_batch, w).contiguous()

            # Apply weights and sum over relations
            h1 = torch.mm(A, xw.view(num_rels * n, e))
            h1 = F.relu(h1 + self.bias1)

            ## Layer 2
            if self.bases2 is None:
                w = self.w2
            else:
                w = torch.einsum('rb, bho -> rho', self.comp2, self.bases2)

            h2 = torch.einsum('nh, rhc -> rnc', h1, w).contiguous()
            h2 = h2.view(self.num_rels * n, c)
            h2 = torch.mm(A, h2)

            return h2 + self.bias2

    def penalty(self, p=2):

        if self.num_bases is None:
            return self.w1.pow(p).sum()

        return self.comp1.pow(p).sum() + self.bases1.pow(p).sum()






