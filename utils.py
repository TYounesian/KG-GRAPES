from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import scipy.sparse as sp
import numpy as np
import os, pdb
import torchvision as tv
from PIL import ImageOps
import time
from torch.distributions import Bernoulli, Gumbel
import networkx as nx
import matplotlib.pyplot as plt
import math


def d(tensor=None):
    """
    Returns a device string either for the best available device,
    or for the device corresponding to the argument
    :param tensor:
    :return:
    """
    if tensor is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if type(tensor) == bool:
        return 'cuda'if tensor else 'cpu'
    return 'cuda' if tensor.is_cuda else 'cpu'


def sum_sparse(indices, values, size, row_normalisation, device='cpu'):
    """
    Sum the rows or columns of a sparse matrix, and redistribute the
    results back to the non-sparse row/column entries
    Arguments are interpreted as defining sparse matrix.
    Source: https://github.com/pbloem/gated-rgcn/blob/1bde7f28af8028f468349b2d760c17d5c908b58b/kgmodels/util/util.py#L304
    """
    assert len(indices.size()) == len(values.size()) + 1
    k, r = indices.size()

    if not row_normalisation:
        # Transpose the matrix for column-wise normalisation
        indices = torch.cat([indices[:, 1:2], indices[:, 0:1]], dim=1)
        size = size[1], size[0]

    ones = torch.ones((size[1], 1), device=device)
    if device == 'cuda':
        values = torch.cuda.sparse.FloatTensor(indices.t(), values, torch.Size(size))
    else:
        values = torch.sparse.FloatTensor(indices.t(), values, torch.Size(size))
    sums = torch.spmm(values, ones)
    sums = sums[indices[:, 0], 0]

    return sums.view(k)


def here(subpath=None):
    """
    :return: the path in which the package resides (the directory containing the 'kgmodels' dir)
    """
    if subpath is None:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', subpath))


def get_neighbours_sparse(A, idx):
    """
    Given the sparce Adjacency matrix, return the indices of the neighbouring nodes 
    of idx irrespective of the relations.
    """
    n, rn = A.size()

    isin = torch.isin(A._indices()[0], idx.to(A.device))
    col = A._indices()[1][torch.where(isin)[0]]
    neighbours_global = torch.unique(col % n)

    return neighbours_global


def slice_rows_tensor(A, idx):
    n, _ = A.size()
    idx_onehot = F.one_hot(idx, n)
    A_transpose = torch.transpose(A, 0 , 1)
    A_sliced = torch.transpose(torch.mm(A_transpose, idx_onehot.float().t()), 0 , 1)
    return A_sliced


def slice_rows_tensor2(A, idx):
    """
    Given the sparce Adjacency matrix, return the indices of the neighbouring nodes
    of idx irrespective of the relations.
    """
    n, rn = A.size()
    r = rn // n
    # idx_set = set(idx.tolist())
    # isin = torch.tensor([(id.item() in idx_set) for id in A._indices()[0]])
    isin = torch.isin(A._indices()[0], idx.to(A.device))
    # diff = torch.eq(isin, isin2).sum()
    # assert diff == len(isin)
    # row, col = A._indices()[:, torch.where(torch.isin(A._indices([0], idx))[0]] #use to work!)
    row, col = A._indices()[:, torch.where(isin)[0]]
    idx_set = idx
    row_index_map = {int(j): i for i, j in enumerate(idx_set)}
    row = torch.LongTensor([row_index_map[int(i)] for i in row])

    # val_ind = torch.where(torch.isin(A._indices()[0], idx))[0]
    val_ind = torch.where(isin)[0]
    val_ind = torch.where(isin)[0]
    vals = A._values()
    vals = vals[val_ind]

    return torch.sparse_coo_tensor(torch.vstack([row, col]), vals,
                                   size=[len(idx), A.size(1)])


def getAdjacencyNodeColumnIdx(idx, num_nodes, num_relations):
    """
    Return column idx for all nodes in idx for all relations
    """

    r = torch.arange(0, num_relations)
    r = torch.mul(r, num_nodes)
    cols = r[:, None] + idx[None, :]
    cols = torch.reshape(cols, (-1,))
    return cols


def sliceSparseCOO(t, num_rels, idx):
    #TODO:vectorize possible?
    row, col = t._indices()[:, torch.where(torch.isin(t._indices()[1],
                                                      idx))[0]]

    mx = torch.max(idx)+1
    map = torch.empty((int(mx),), dtype=torch.long, device= t.device)
    map1 = map.scatter_(0, idx, torch.arange(len(idx), device = t.device))
    col = torch.gather(map1, 0, col)

    indices = torch.vstack([row, col])
    vals = torch.ones(len(row), device=t.device)
    size = [t.shape[0], len(idx)]
    # ones = torch.ones(r, len(idx), 1, device=t.device)
    values = torch.sparse.FloatTensor(indices, vals, torch.Size(size))
    # block_mat = torch.block_diag(*ones)
    ones_rows = torch.tensor(range(len(idx)), device= t.device)
    ones_cols = torch.tensor([item for item in range(num_rels) for i in range(len(idx)//num_rels)], device= t.device)
    ones_indices = torch.vstack([ones_rows, ones_cols])
    block_mat = torch.sparse.FloatTensor(ones_indices, torch.ones(len(idx), device= t.device), torch.Size([len(idx), num_rels]))
    sums = (torch.sparse.mm(values, block_mat)).to_dense()
    sums_a = sums[indices[0], torch.div(indices[1], len(idx)//num_rels, rounding_mode='floor')]
    vals_norm = torch.div(vals, sums_a)

    return torch.sparse_coo_tensor(indices, vals_norm, size)


def slice_whole_tensor(A, idx, n, r):
    idx_extended = torch.LongTensor([int((rr*n) + i) for rr in range(r) for i in idx]).cuda()
    idx_onehot = F.one_hot(idx_extended, n*r)
    A_sliced = torch.mm(A, idx_onehot.float().t())
    return A_sliced


def to_tvbatches(images, batch_size=16,  min_size=0, dtype=None, prep=tv.transforms.ToTensor()):
    """
    Returns a generator over torch batches of tensors, using torchvision transforms to translate from
    PIL images to tensors.

    :param images:
    :param torch:
    :return:
    """
    batches = []
    for fr in range(0, len(images), batch_size):
        batch = images[fr:fr+batch_size]

        yield to_tvbatch(batch, min_size=min_size, dtype=dtype, prep=prep)


def to_tvbatch(images, min_size=0, dtype=None, prep=tv.transforms.ToTensor()):

    maxw = max(max([img.size[0] for img in images]), min_size)
    maxh = max(max([img.size[1] for img in images]), min_size)

    res = []
    for img in images:
        img = pad(img, (maxw, maxh))
        res.append(prep(img)[None, :, :, :])

    res = torch.cat(res, dim=0)
    if dtype is not None:
        res = res.to(dtype)

    return res


def pad(im, desired_size):

    dw = desired_size[0] - im.size[0]
    dh = desired_size[1] - im.size[1]
    padding = (dw // 2, dh // 2, dw - (dw // 2), dh - (dh // 2))

    return ImageOps.expand(im, padding)


def enrich(triples : torch.Tensor, n : int, r: int):
    cuda = triples.is_cuda

    inverses = torch.cat([
        triples[:, 2:],
        triples[:, 1:2] + r,
        triples[:, :1]
    ], dim=1)

    selfloops = torch.cat([
        torch.arange(n, dtype=torch.long,  device=d(cuda))[:, None],
        torch.full((n, 1), fill_value=2*r),
        torch.arange(n, dtype=torch.long, device=d(cuda))[:, None],
    ], dim=1)

    return torch.cat([triples, inverses, selfloops], dim=0)


def enrich_with_drop(triples, self_loop_dropout, n, r):
    cuda = triples.is_cuda

    inverses = torch.cat([
        triples[:, 2:],
        triples[:, 1:2] + r,
        triples[:, :1]
    ], dim=1)

    keep_rate = 1- self_loop_dropout

    self_loops = torch.arange(n, dtype=torch.long,  device=d(cuda))[:, None]
    mask = torch.bernoulli(torch.empty(size=(n,), dtype=torch.float).fill_(
        keep_rate)).to(torch.bool)
    self_loops = self_loops[mask, :]

    selfloops = torch.cat([
        self_loops,
        torch.full((len(self_loops), 1), fill_value=2*r),
        self_loops,
    ], dim=1)

    return torch.cat([triples, inverses, selfloops], dim=0)


def adj(edges, num_nodes, num_rels, cuda=False, vertical=True):
    """
    Computes a sparse adjacency matrix for the given graph (the adjacency matrices of all
    relations are stacked vertically).
    :param edges: Dictionary representing the edges
    :param i2r: list of relations
    :param i2n: list of nodes
    :return: sparse tensor
    """

    r, n = num_rels, num_nodes
    size = (r*n, n) if vertical else (n, r*n)

    from_indices = []
    upto_indices = []

    for fr, rel, to in edges:
        offset = rel.item() * n

        if vertical:
            fr = offset + fr.item()
        else:
            to = offset + to.item()

        from_indices.append(fr)
        upto_indices.append(to)

    indices = torch.tensor([from_indices, upto_indices], dtype=torch.long, device=d(cuda))
    # pdb.set_trace()
    assert indices.size(1) == len(edges)
    assert indices[0, :].max() < size[0], f'{indices[0, :].max()}, {size}, {r}'
    assert indices[1, :].max() < size[1], f'{indices[1, :].max()}, {size}, {r}'

    return indices.t(), size


def adj_norel(edges, num_nodes, cuda=False):
    """
    Computes a sparse adjacency matrix for the given graph (the adjacency matrices of all
    relations are stacked vertically).
    :param edges: Dictionary representing the edges
    :param i2r: list of relations
    :param i2n: list of nodes
    :return: sparse tensor
    """
    size = (num_nodes, num_nodes)

    from_indices = []
    upto_indices = []

    for fr, rel, to in edges:
        from_indices.append(fr.item())
        upto_indices.append(to.item())

    indices = torch.tensor([from_indices, upto_indices], dtype=torch.long, device=d(cuda))

    assert indices.size(1) == len(edges)

    return indices.t(), size


def adj_separate(edges, num_nodes, num_rels, cuda=False):
    """
    Computes a list of sparse adjacency matrices for the given graph.
    :param edges: Dictionary representing the edges
    :return: sparse tensor
    """

    r, n = num_rels, num_nodes

    from_indices = [[]] * r
    upto_indices = [[]] * r
    indices = []

    for fr, rel, to in edges:
        from_indices[rel].append(fr)
        upto_indices[rel].append(to)

    for rel in range(r):
        indices.append(torch.tensor([from_indices[rel], upto_indices[rel]], dtype=torch.long, device=d(cuda)))

    assert len(indices) == r

    return indices


def adj_with_neighbours(edges, num_nodes, num_rels, cuda=False, vertical=True):
    """
    Computes a sparse adjacency matrix for the given graph (the adjacency matrices of all
    relations are stacked vertically).
    :param edges: Dictionary representing the edges
    :param i2r: list of relations
    :param i2n: list of nodes
    :return: sparse tensor
    """
    neighbours = {}
    r, n = num_rels, num_nodes
    size = (r*n, n) if vertical else (n, r*n)

    from_indices = []
    upto_indices = []

    for fr, rel, to in edges:

        # get neighbours
        if fr.item() not in neighbours:
            neighbours[fr.item()] = list()
        if to not in neighbours[fr.item()]:
            neighbours[fr.item()].append(to.item())


        offset = rel.item() * n

        if vertical:
            fr = offset + fr.item()
        else:
            to = offset + to.item()

        from_indices.append(fr)
        upto_indices.append(to)


    indices = torch.tensor([from_indices, upto_indices], dtype=torch.long, device=d(cuda))

    assert indices.size(1) == len(edges)
    assert indices[0, :].max() < size[0], f'{indices[0, :].max()}, {size}, {r}'
    assert indices[1, :].max() < size[1], f'{indices[1, :].max()}, {size}, {r}'

    return indices.t(), size


def adj_triples(triples, num_nodes, num_rels, cuda=False, vertical=True):
    """
    Computes a sparse adjacency matrix for the given graph (the adjacency matrices of all
    relations are stacked vertically).
    :param edges: List representing the triples
    :param i2r: list of relations
    :param i2n: list of nodes
    :return: sparse tensor
    """
    r, n = num_rels, num_nodes
    size = (r*n, n) if vertical else (n, r*n)

    from_indices = []
    upto_indices = []

    for fr, rel, to in triples:

        offset = rel.item() * n

        if vertical:
            fr = offset + fr.item()
        else:
            to = offset + to.item()

        from_indices.append(fr)
        upto_indices.append(to)

    tic()
    indices = torch.tensor([from_indices, upto_indices], dtype=torch.long, device=d(cuda))

    assert indices.size(1) == len(triples)
    assert indices[0, :].max() < size[0], f'{indices[0, :].max()}, {size}, {r}'
    assert indices[1, :].max() < size[1], f'{indices[1, :].max()}, {size}, {r}'

    return indices.t(), size


def get_splits(y, train_idx, test_idx, validation=True):
    # Make dataset splits
    # np.random.shuffle(train_idx)
    if validation:
        idx_train = train_idx[len(train_idx) / 5:]
        idx_val = train_idx[:len(train_idx) / 5]
        idx_test = idx_val  # report final score on validation set for hyperparameter optimization
    else:
        idx_train = train_idx
        idx_val = train_idx  # no validation
        idx_test = test_idx

    y_train = np.zeros(y.shape)
    y_val = np.zeros(y.shape)
    y_test = np.zeros(y.shape)

    y_train[idx_train] = np.array(y[idx_train].todense())
    y_val[idx_val] = np.array(y[idx_val].todense())
    y_test[idx_test] = np.array(y[idx_test].todense())

    return y_train, y_val, y_test, idx_train, idx_val, idx_test


def sampler_func(sampler, batch_id, num_nodes, num_rels, horizontal_en_A_tr, norel_A_tr, A_sep, samp_num_list, depth,
            model_g, model_z, embed_X, indicator_features, current_e, start_e, end_e, pert, pert_ratio, device):
    if sampler == 'full-mini-batch':
        A_en_sliced, after_nodes_list, rels_more = full_mini_sampler(batch_id, num_nodes,
                                                                     num_rels, horizontal_en_A_tr, depth, device)
        idx_per_rel_list = []
        nonzero_rel_list = []
        log_probs = 0
        log_z = 0
        A_en_sliced_pert = []
        after_nodes_list_pert = []
    elif sampler == 'grapes':
        A_en_sliced, after_nodes_list, idx_per_rel_list, nonzero_rel_list, rels_more, log_probs, log_z, statistics, \
        A_en_sliced_pert, after_nodes_list_pert = \
            grapes_sampler(
                batch_id,
                samp_num_list,
                num_nodes,
                num_rels,
                horizontal_en_A_tr,
                depth,
                sampler,
                model_g,
                model_z,
                embed_X,
                indicator_features,
                current_e,
                start_e,
                end_e,
                pert,
                pert_ratio,
                device)
    elif sampler == 'LDUN':
        A_en_sliced, after_nodes_list, idx_per_rel_list, nonzero_rel_list, rels_more = ladies_norel_sampler(batch_id,
                                                                                                            samp_num_list,
                                                                                                            num_nodes,
                                                                                                            num_rels,
                                                                                                            norel_A_tr,
                                                                                                            horizontal_en_A_tr,
                                                                                                            depth,
                                                                                                            sampler,
                                                                                                            device)
        log_probs = 0.
        log_z = 0.
        statistics = []
        A_en_sliced_pert = []
        after_nodes_list_pert = []
    elif sampler == "LDRN" or sampler == "LDRE":
        A_en_sliced, after_nodes_list, idx_per_rel_list, nonzero_rel_list, rels_more = ladies_sampler(batch_id,
                                                                                                      samp_num_list,
                                                                                                      num_nodes,
                                                                                                      num_rels,
                                                                                                      horizontal_en_A_tr,
                                                                                                      depth,
                                                                                                      sampler,
                                                                                                      device)
        log_probs = 0.
        log_z = 0.
        A_en_sliced_pert = []
        after_nodes_list_pert = []
    elif sampler == 'hetero':
        A_en_sliced, after_nodes_list = hetero_sampler(batch_id, samp_num_list, num_nodes,
                                                       num_rels, A_sep, depth, device)
        idx_per_rel_list = []
        nonzero_rel_list = []
        rels_mode = []
        log_probs = 0.
        log_z = 0.
        A_en_sliced_pert = []
        after_nodes_list_pert = []
    elif sampler == 'IARN':
        A_en_sliced, after_nodes_list, rels_more = random_sampler(batch_id, samp_num_list, num_nodes,
                                                                  num_rels,
                                                                  horizontal_en_A_tr, depth, device)
        idx_per_rel_list = []
        nonzero_rel_list = []
        log_probs = 0.
        log_z = 0.
        A_en_sliced_pert = []
        after_nodes_list_pert = []
    elif sampler == 'IDRN':
        A_en_sliced, after_nodes_list, rels_more = fastgcn_plus_sampler(batch_id, samp_num_list, num_nodes,
                                                                        num_rels,
                                                                        horizontal_en_A_tr, depth, device)
        idx_per_rel_list = []
        nonzero_rel_list = []
        log_probs = 0.
        log_z = 0.
        A_en_sliced_pert = []
        after_nodes_list_pert = []
    return A_en_sliced, after_nodes_list, idx_per_rel_list, nonzero_rel_list, rels_more, log_probs, log_z, statistics,\
           A_en_sliced_pert, after_nodes_list_pert


def adj_r_creator(edges, self_loop_dropout, num_nodes, num_rels, sampler):
    edges_en_tr = enrich_with_drop(edges, self_loop_dropout, num_nodes,
                                   int((num_rels - 1) / 2))
    edges_en_ts = enrich_with_drop(edges, 0, num_nodes, int((num_rels - 1) / 2))

    # horizontal indices and sizes
    hor_en_ind_tr, hor_en_size_tr = adj(edges_en_tr, num_nodes, num_rels, vertical=False)
    ver_en_ind_tr, ver_en_size_tr = adj(edges_en_tr, num_nodes, num_rels, vertical=True)

    hor_en_ind_ts, hor_en_size_ts = adj(edges_en_ts, num_nodes, num_rels, vertical=False)
    ver_en_ind_ts, ver_en_size_ts = adj(edges_en_ts, num_nodes, num_rels, vertical=True)

    vals_en_tr = torch.ones(ver_en_ind_tr.size(0), dtype=torch.float)
    vals_en_tr = vals_en_tr / sum_sparse(ver_en_ind_tr, vals_en_tr, ver_en_size_tr, True)

    vals_en_ts = torch.ones(ver_en_ind_ts.size(0), dtype=torch.float)
    vals_en_ts = vals_en_ts / sum_sparse(ver_en_ind_ts, vals_en_ts, ver_en_size_ts, True)

    # constructing the horizontal and vertical Adj matrices
    hor_en_adj_tr = torch.sparse.FloatTensor(indices=hor_en_ind_tr.t(), values=vals_en_tr, size=hor_en_size_tr)
    hor_en_adj_ts = torch.sparse.FloatTensor(indices=hor_en_ind_ts.t(), values=vals_en_ts, size=hor_en_size_ts)
    norel_A_tr = []
    norel_A_ts = []
    if sampler == 'LDUN':
        norel_ind_tr, norel_size_tr = adj_norel(edges_en_tr, num_nodes)

        norel_vals_en_tr = torch.ones(norel_ind_tr.size(0), dtype=torch.float)
        norel_vals_en_tr = norel_vals_en_tr / sum_sparse(norel_ind_tr, norel_vals_en_tr, norel_size_tr, True)

        norel_A_tr = torch.sparse.FloatTensor(indices=norel_ind_tr.t(), values=norel_vals_en_tr,
                                              size=norel_size_tr)

        norel_ind_ts, norel_size_ts = adj_norel(edges_en_ts, num_nodes)

        norel_vals_en_ts = torch.ones(norel_ind_ts.size(0), dtype=torch.float)
        norel_vals_en_ts = norel_vals_en_ts / sum_sparse(norel_ind_ts, norel_vals_en_ts, norel_size_ts, True)

        norel_A_ts = torch.sparse.FloatTensor(indices=norel_ind_ts.t(), values=norel_vals_en_ts,
                                              size=norel_size_ts)

    return hor_en_adj_tr, hor_en_adj_ts, norel_A_tr, norel_A_ts


def sample_neighborhoods_from_probs(logits, num_samples, test, current_e, start_e, end_e):
    """Remove edges from an edge index, by removing nodes according to some
    probability.

    Uses Gumbel-max trick to sample from Bernoulli distribution. This is off-policy, since the original input
    distribution is a regular Bernoulli distribution.
    Args:
        logits: tensor of shape (N,), where N all the number of unique
            nodes in a batch, containing the probability of dropping the node.
        neighbor_nodes: tensor containing global node identifiers of the neighbors nodes
        num_samples: the number of samples to keep. If None, all edges are kept.
    """

    k = num_samples
    n = logits.size(0)
    if k >= n:
        logprobs = torch.nn.functional.logsigmoid(logits.squeeze(-1))
        return torch.arange(0, len(logits)), logprobs, {}
    assert k < n
    assert k > 0

    # shuffle logits
    # shuffle_idx = torch.randperm(logits.size(0))
    # logits = logits[shuffle_idx]
    test = True
    b = Bernoulli(logits=logits.squeeze())
    # Gumbel-sort trick https://timvieira.github.io/blog/post/2014/08/01/gumbel-max-trick-and-weighted-reservoir-sampling/
    gumbel = Gumbel(torch.tensor(0., device=logits.device), torch.tensor(1., device=logits.device))
    gumbel_noise = gumbel.sample((n,))
    if current_e < start_e:
        perturbed_log_probs = b.probs.log() + gumbel_noise
    else:
        # decay = math.log(1 / 1e-6) / (end_e - start_e)
        # gumbel_noise = gumbel_noise * math.exp(-decay * (current_e - start_e + 1)) #* b.probs.log().std()
        # tau = 0.1 #1 - (current_e - start_e) / (end_e - start_e)
        # if current_e > end_e/4:
        #     tau = 0.01
        # if current_e > end_e/2:
        #     tau = 0.005
        # min_noise = 1e-3
        # gumbel_noise = gumbel_noise * tau + min_noise
        # print(f'gumbel noise: {gumbel_noise} at epoch {current_e}')
        perturbed_log_probs = b.probs.log() + torch.rand(n, device=logits.device)*1e-2 #+ gumbel_noise
    if test:
        samples = torch.topk(b.probs, k=k, dim=0, sorted=False)[1].to('cpu')#torch.topk(b.probs+torch.rand(n)*1e-2, k=k, dim=0, sorted=False)[1].to('cpu')
        # samples = torch.arange(k)
    else:
        samples = torch.topk(perturbed_log_probs, k=k, dim=0, sorted=False)[1].to('cpu')

    print("probs:", " ".join(f"{p:.5f}" for p in b.probs[torch.sort(samples).values].tolist()))
    print("samples:", torch.sort(samples).values)

    # shuffle back
    # samples = shuffle_idx[samples]
    # print("samples:", samples)

    print("shared between probs and perturbed:",
          len(set(torch.topk(b.probs, k=k, dim=0, sorted=False)[1].to('cpu').tolist()) &
              set(torch.topk(perturbed_log_probs, k=k, dim=0, sorted=False)[1].to('cpu').tolist())))
    print("shared between noise and perturbed:",
          len(set(torch.topk(gumbel_noise, k=k, dim=0, sorted=False)[1].to('cpu').tolist()) &
              set(torch.topk(perturbed_log_probs, k=k, dim=0, sorted=False)[1].to('cpu').tolist())))
    # calculate the entropy in bits
    entropy = -(b.probs * (b.probs).log2() + (1 - b.probs) * (1 - b.probs).log2())

    min_prob = b.probs.min(-1)[0]
    max_prob = b.probs.max(-1)[0]

    if torch.isnan(entropy).any():
        nan_ind = torch.isnan(entropy)
        entropy[nan_ind] = 0.0

    std_entropy, mean_entropy = torch.std_mean(entropy)
    mask = torch.zeros_like(logits.squeeze(), dtype=torch.float)
    mask[samples] = 1

    # neighbor_nodes = neighbor_nodes[mask.bool().cpu()]

    stats_dict = {"min_prob": min_prob,
                  "max_prob": max_prob,
                  "mean_entropy": mean_entropy,
                  "std_entropy": std_entropy}

    if torch.isinf(b.log_prob(mask)).any():
        inf_ind = torch.isinf(b.log_prob(mask))
        b.log_prob(mask)[inf_ind] = -1e-9

    return samples, b.log_prob(mask), stats_dict


def random_sampler_layerwise(batch_idx, samp_num_list, num_nodes, num_rels, A_en, depth, device):
    previous_nodes = batch_idx
    col_ind = []
    A_en_sliced = []
    after_nodes_list = []

    for d in range(depth):
        A_row = slice_rows_tensor2(A_en, previous_nodes)
        size = [len(previous_nodes), num_nodes * num_rels]
        pi = calc_prob(A_row, size, num_rels, device)
        num_prev_nodes = len(previous_nodes)
        p, nonzero_rels = sum_prob_per_rel(pi, num_nodes, num_rels)
        s_num = samp_num_list[d]
        if s_num > 0:
            start = time.time()
            idx_local = rand_sel_idx(p, s_num, num_nodes, num_rels)
            print(f'4 ({time.time() - start:.4}s).')
            after_nodes = idx_local
        else:
            after_nodes = batch_idx
        after_nodes = torch.unique(torch.cat((after_nodes, previous_nodes.to('cpu')))).to(device=device)
        cols = getAdjacencyNodeColumnIdx(after_nodes, num_nodes, num_rels)
        col_ind.append(cols)
        previous_nodes = after_nodes
        # sample A
        A_en_sliced.append(slice_adj_col(A_row, col_ind, num_rels, num_prev_nodes, 'IARN', after_nodes,
                                         len(after_nodes), [], []))
        after_nodes_list.append(after_nodes)

    A_en_sliced.reverse()
    after_nodes_list.reverse()

    return A_en_sliced, after_nodes_list


def random_sampler(batch_idx, samp_num_list, num_nodes, num_rels, A_en, depth, device):
    previous_nodes = batch_idx
    col_ind = []
    A_en_sliced = []
    after_nodes_list = []

    for d in range(depth):
        s_num = samp_num_list[d]
        num_prev_nodes = len(previous_nodes)
        aftermat = [torch.randperm(num_nodes)[:s_num] for r in range(num_rels-1)]
        after_nodes = torch.tensor([i for l in aftermat for i in l])
        after_nodes = torch.unique(torch.cat((after_nodes, previous_nodes.to('cpu')))).to(device=device)
        cols = getAdjacencyNodeColumnIdx(after_nodes, num_nodes, num_rels)
        col_ind.append(cols)
        # sample A
        A_row = slice_rows_tensor2(A_en, previous_nodes)
        A_en_sliced.append(slice_adj_col(A_row, col_ind, num_rels, num_prev_nodes, 'IARN', after_nodes,
                                         len(after_nodes), [], []))
        after_nodes_list.append(after_nodes)
        previous_nodes = after_nodes

    A_en_sliced.reverse()
    after_nodes_list.reverse()

    return A_en_sliced, after_nodes_list, 0


def fastgcn_plus_sampler(batch_idx, samp_num_list, num_nodes, num_rels, A_en, depth, device):
    previous_nodes = batch_idx
    col_ind = []
    A_en_sliced = []
    after_nodes_list = []

    for d in range(depth):
        size = A_en.size()
        pi = calc_prob(A_en, size, num_rels, device)
        num_prev_nodes = len(previous_nodes)
        p, nonzero_rels = sum_prob_per_rel(pi, num_nodes, num_rels)
        s_num = samp_num_list[d]
        if s_num > 0:
            idx_local, non_zero_rels, _, _, _ = sel_idx_node(p, s_num, num_nodes, num_rels)
            after_nodes = idx_local

        after_nodes = torch.unique(torch.cat((after_nodes, previous_nodes.to('cpu')))).to(device=device)
        cols = getAdjacencyNodeColumnIdx(after_nodes, num_nodes, num_rels)
        col_ind.append(cols)
        # sample A
        A_row = slice_rows_tensor2(A_en, previous_nodes)
        A_en_sliced.append(slice_adj_col(A_row, col_ind, num_rels, num_prev_nodes, 'IARN', after_nodes,
                                         len(after_nodes), [], []))
        after_nodes_list.append(after_nodes)
        previous_nodes = after_nodes

    A_en_sliced.reverse()
    after_nodes_list.reverse()

    return A_en_sliced, after_nodes_list, 0


def grapes_sampler(batch_idx, samp_num_list, num_nodes, num_rels, A_en, depth, sampler, model_g, model_z, embed_X,
                   indicator_features, current_e, start_e, end_e, pert, pert_ratio, device):
    previous_nodes = batch_idx
    previous_nodes_pert = batch_idx
    col_ind = []
    A_en_sliced = []
    A_en_sliced_pert = []
    after_nodes_list = []
    after_nodes_pert = []
    after_nodes_list_pert = []
    idx_per_rel_list = []
    non_zero_rel_list = []
    log_probs = []
    stats = []

    for d in range(depth):
        neighbors = get_neighbours_sparse(A_en, previous_nodes)
        mask = ~torch.isin(neighbors, previous_nodes)
        only_neighbors = neighbors[mask]
        cols = getAdjacencyNodeColumnIdx(previous_nodes, num_nodes, 2*num_rels+1)
        # A_en_row = slice_rows_tensor2(A_en, previous_nodes)
        A_gf = slice_adj_row_col(A_en, neighbors, cols, len(neighbors), len(previous_nodes), 'prob')
        # calculate the importance of each neighbor
        indicator_features[only_neighbors, d] = 1.0
        # batch_nodes = neighbors[~torch.isin(neighbors, previous_nodes)]
        x = torch.cat([embed_X[neighbors],
                    indicator_features[neighbors]],
                    dim=1)
        node_logits, _ = model_g(x, A_gf.to(device), neighbors, [], [], 'full', device)
        if d == 0:
            pred_z, _ = model_z(embed_X[neighbors], A_gf.to(device), neighbors, [], [], 'full', device)
            log_z = pred_z.mean()
        num_prev_nodes = len(previous_nodes)
        # calculate the probability of sampling each neighbor in each relation
        # output the relations that appear in at least one neighbor

        node_logits = node_logits[mask]
        s_num = samp_num_list[d]
        if s_num > 0:
            idx_local, log_prob, statistics = sample_neighborhoods_from_probs(
                node_logits,
                s_num,
                not(model_g.training),
                current_e,
                start_e,
                end_e)
            log_probs.append(log_prob)
            # sample nodes given their probablity.
            # output the local and global idx of the neighbors, the probability of the nodes sampled
            # idx_local, nonzero_rels, global_idx, prob, rels_more = sel_idx_node(p, s_num, num_nodes, num_rels)
            idx_list_per_rel = []
            after_nodes = only_neighbors[idx_local]  # unique node idx
            print("after nodes:", torch.unique(after_nodes))
        else:
            after_nodes = batch_idx

        # A_row = slice_rows_tensor2(nore_A, previous_nodes)
        # A_en_row = slice_rows_tensor2(A_en, previous_nodes)
        # size = [len(previous_nodes), num_nodes]
        # pi = calc_prob(A_row, size, 1, device)
        # num_prev_nodes = len(previous_nodes)
        # sum_pi = pi.sum()
        # p = pi / sum_pi
        # s_num = samp_num_list[d]
        # if s_num > 0:
        #     idx_local, nonzero_rels, global_idx, prob, rels_more = sel_idx_node(p, s_num, num_nodes, 1)
        #     idx_list_per_rel = []
        #     after_nodes = idx_local  # unique node idx

        # unique node idx with aggregation
        after_nodes = torch.unique(torch.cat((after_nodes, batch_idx.to('cpu'))))
        if pert:
            after_nodes_pert = neighbors[torch.randperm(s_num)[:int(s_num * (1 - pert_ratio))]]
            after_nodes_pert = torch.unique(torch.cat((after_nodes_pert, batch_idx.to('cpu'))))
            cols = getAdjacencyNodeColumnIdx(after_nodes_pert, num_nodes, 2 * num_rels + 1)
            col_ind.append(cols)
            # sample A
            A_en_sliced_pert.append(
                slice_adj_row_col(A_en, previous_nodes_pert, cols, len(previous_nodes_pert), len(after_nodes_pert), 'cl').to(device))
            previous_nodes_pert = after_nodes_pert

        cols = getAdjacencyNodeColumnIdx(after_nodes, num_nodes, 2*num_rels+1)
        col_ind.append(cols)
        # sample A
        # A_en_sliced.append(slice_adj_col(A_en_row, col_ind, 2 * num_rels + 1, num_prev_nodes, sampler, after_nodes,
        #                                  len(after_nodes), [], []).to(device))
        A_en_sliced.append(slice_adj_row_col(A_en, previous_nodes, cols, num_prev_nodes, len(after_nodes), 'cl').to(device))
        # A_en_sliced.append(slice_adj_col(A_en_row, col_ind, 2 * num_rels + 1, num_prev_nodes, sampler, after_nodes,
        #                                  len(after_nodes), global_idx, prob).to(device))

        previous_nodes = after_nodes

        after_nodes_list.append(after_nodes)
        after_nodes_list_pert.append(after_nodes_pert)
        idx_list_per_rel_dev = [i.to(device) for i in idx_list_per_rel]
        idx_per_rel_list.append(idx_list_per_rel_dev)
        stats.append(statistics)
        # non_zero_rel_list.append(nonzero_rels)
    A_en_sliced.reverse()
    A_en_sliced_pert.reverse()
    after_nodes_list.reverse()
    after_nodes_list_pert.reverse()
    idx_per_rel_list.reverse()
    stats.reverse()
    return A_en_sliced, after_nodes_list, idx_per_rel_list, non_zero_rel_list, 0, log_probs, log_z, stats, \
           A_en_sliced_pert, after_nodes_list_pert


def ladies_sampler(batch_idx, samp_num_list, num_nodes, num_rels, A_en, depth, sampler, device):
    previous_nodes = batch_idx
    col_ind = []
    A_en_sliced = []
    after_nodes_list = []
    idx_per_rel_list = []
    non_zero_rel_list = []

    for d in range(depth):
        # row slice the adjacency by the batch nodes to find their neighbors
        A_row = slice_rows_tensor2(A_en, previous_nodes)
        size = [len(previous_nodes), num_nodes * (2*num_rels+1)]
        # calculate the importance of each neighbor
        pi = calc_prob(A_row, size, num_rels, device)
        num_prev_nodes = len(previous_nodes)
        # calculate the probability of sampling each neighbor in each relation
        # output the relations that appear in at least one neighbor
        p, nonzero_rels = sum_prob_per_rel(pi, num_nodes, (2*num_rels+1))
        s_num = samp_num_list[d]
        if s_num > 0:
            if sampler == 'LDRN': # node sampling by ReWise: including all the relations of a node
                #sample nodes given their probablity.
                # output the local and global idx of the neighbors, the probability of the nodes sampled
                idx_local, nonzero_rels, global_idx, prob, rels_more = sel_idx_node(p, s_num, num_nodes, (2*num_rels+1))
                idx_list_per_rel = []

            elif sampler == 'LDRE': # edge sampling by ReWise: sampling nodes in different relations
                idx_extend, idx_local, idx_list_per_rel, s_n, rels_more = sel_idx_edge(p, s_num, num_nodes, (2*num_rels+1), nonzero_rels)
                idx_extend = torch.unique(torch.cat((idx_extend, num_nodes*(2*num_rels)+previous_nodes)))
            after_nodes = idx_local  # unique node idx
        else:
            after_nodes = batch_idx
        # unique node idx with aggregation
        after_nodes = torch.unique(torch.cat((after_nodes, previous_nodes.to('cpu'))))
        if sampler == 'LDRN':
            cols = getAdjacencyNodeColumnIdx(after_nodes, num_nodes, (2*num_rels+1))
            col_ind.append(cols)
            # sample A
            A_en_sliced.append(slice_adj_col(A_row, col_ind, (2*num_rels+1), num_prev_nodes, sampler, after_nodes,
                                             len(after_nodes), global_idx, prob).to(device))
        elif sampler == 'LDRE':
            col_ind.append(idx_extend)
            A_en_sliced.append(slice_adj_col(A_row, col_ind, (2*num_rels+1), num_prev_nodes, sampler, after_nodes,
                                             len(idx_extend), [], []).to(device))
        previous_nodes = after_nodes
        after_nodes_list.append(after_nodes)
        idx_list_per_rel_dev = [i.to(device) for i in idx_list_per_rel]
        idx_per_rel_list.append(idx_list_per_rel_dev)
        non_zero_rel_list.append(nonzero_rels)
    A_en_sliced.reverse()
    after_nodes_list.reverse()
    idx_per_rel_list.reverse()

    return A_en_sliced, after_nodes_list, idx_per_rel_list, non_zero_rel_list, rels_more


def ladies_norel_sampler(batch_idx, samp_num_list, num_nodes, num_rels, nore_A, A_en, depth, sampler, device):
    previous_nodes = batch_idx
    col_ind = []
    A_en_sliced = []
    after_nodes_list = []
    idx_per_rel_list = []
    non_zero_rel_list = []

    for d in range(depth):
        # row slice the adjacency by the batch nodes to find their neighbors
        A_row = slice_rows_tensor2(nore_A, previous_nodes)
        A_en_row = slice_rows_tensor2(A_en, previous_nodes)
        size = [len(previous_nodes), num_nodes]
        # calculate the importance of each neighbor
        pi = calc_prob(A_row, size, 1, device)
        num_prev_nodes = len(previous_nodes)
        # calculate the probability of sampling each neighbor in each relation
        # output the relations that appear in at least one neighbor
        sum_pi = pi.sum()
        p = pi/sum_pi
        s_num = samp_num_list[d]
        if s_num > 0:
            if sampler == 'LDRN' or sampler == 'LDUN':
                # node sampling by ReWise: including all the relations of a node
                # sample nodes given their probablity.
                # output the local and global idx of the neighbors, the probability of the nodes sampled
                idx_local, nonzero_rels, global_idx, prob, rels_more = sel_idx_node(p, s_num, num_nodes, 1)
                idx_list_per_rel = []
            after_nodes = idx_local  # unique node idx

        else:
            after_nodes = batch_idx
        # unique node idx with aggregation
        after_nodes = torch.unique(torch.cat((after_nodes, batch_idx.to('cpu'))))
        if sampler == 'LDRN' or sampler == 'LDUN':
            cols = getAdjacencyNodeColumnIdx(after_nodes, num_nodes, 2*num_rels+1)
            col_ind.append(cols)
            # sample A
            A_en_sliced.append(slice_adj_col(A_en_row, col_ind, 2*num_rels+1, num_prev_nodes, sampler, after_nodes,
                                             len(after_nodes), global_idx, prob).to(device))

        previous_nodes = after_nodes
        after_nodes_list.append(after_nodes)
        idx_list_per_rel_dev = [i.to(device) for i in idx_list_per_rel]
        idx_per_rel_list.append(idx_list_per_rel_dev)
        non_zero_rel_list.append(nonzero_rels)
    A_en_sliced.reverse()
    after_nodes_list.reverse()
    idx_per_rel_list.reverse()

    return A_en_sliced, after_nodes_list, idx_per_rel_list, non_zero_rel_list, rels_more


def full_mini_sampler(batch_idx, num_nodes, num_rels, A_en, depth, device):
    previous_nodes = batch_idx
    row_ind = []
    col_ind = []
    A_en_sliced = []
    after_nodes_list = []

    for d in range(depth):
        previous_nodes = previous_nodes.to('cpu')
        after_nodes = get_neighbours_sparse(A_en, previous_nodes).to('cpu')
        A_row = slice_rows_tensor2(A_en, previous_nodes)
        num_prev_nodes = len(previous_nodes)
        cols = getAdjacencyNodeColumnIdx(after_nodes, num_nodes, 2 * num_rels + 1)
        row_ind.append(previous_nodes)
        col_ind.append(cols)
        previous_nodes = after_nodes
        # sample A
        A_en_sliced.append(slice_adj_col(A_row, col_ind, 2*num_rels+1, num_prev_nodes, 'LDRN', after_nodes,
                                         len(after_nodes), [], []).to(device))
        after_nodes_list.append(after_nodes)

    A_en_sliced.reverse()
    after_nodes_list.reverse()

    return A_en_sliced, after_nodes_list, 0


def ladies_sampler_allr(batch_idx, samp_num_list, num_nodes, num_rels, A, A_en, depth, device):
    previous_nodes = batch_idx
    row_ind = []
    col_ind = []
    A_en_sliced = []
    after_nodes_list = []

    for d in range(depth):
        A_row = slice_rows_tensor2(A, previous_nodes)
        size = [len(previous_nodes), num_nodes]
        pi = calc_prob(A_row, size, num_rels, device)
        num_prev_nodes = len(previous_nodes)
        p, nonzero_rels = sum_prob_per_rel(pi, num_nodes, 1)
        s_num = samp_num_list[d]
        if s_num > 0:
            idx, nonzero_rels, _, _, _ = sel_idx_node(p, s_num, num_nodes, 1)
            after_nodes = idx
        else:
            after_nodes = batch_idx
        after_nodes = torch.unique(torch.cat((after_nodes, batch_idx))).to(device=device)
        cols = after_nodes
        row_ind.append(previous_nodes)
        col_ind.append(cols)
        previous_nodes = after_nodes
        # sample A
        A_en_sliced.append(slice_adj_row_col(A_en, row_ind, col_ind, 2 * num_rels + 1, num_prev_nodes, len(after_nodes)))
        after_nodes_list.append(after_nodes)

    A_en_sliced.reverse()
    after_nodes_list.reverse()

    return A_en_sliced, after_nodes_list, 0


def hetero_sampler(batch_idx, samp_num_list, num_nodes, num_rels, A, A_en, depth, device):
    previous_nodes = batch_idx
    row_ind = []
    col_ind = []
    A_en_sliced = []
    after_nodes_list = []

    for d in range(depth):
        A_row = slice_rows_tensor2(A, previous_nodes)
        size = [len(previous_nodes), num_nodes * num_rels]
        pi = calc_prob(A_row, size, num_rels, device)
        num_prev_nodes = len(previous_nodes)
        p, nonzero_rels = sum_prob_per_rel(pi, num_nodes, num_rels)
        s_num = samp_num_list[d]
        if s_num > 0:
            idx, nonzero_rels, _, _ = sel_idx_node(p, s_num, num_nodes, num_rels)
            after_nodes = idx
        else:
            after_nodes = batch_idx
        after_nodes = torch.unique(torch.cat((after_nodes, batch_idx))).to(device=device)
        cols = getAdjacencyNodeColumnIdx(after_nodes, num_nodes, 2 * num_rels + 1)
        row_ind.append(previous_nodes)
        col_ind.append(cols)
        previous_nodes = after_nodes
        # sample A
        A_en_sliced.append(slice_adj_row_col(A_en, row_ind, col_ind, 2 * num_rels + 1, num_prev_nodes, len(after_nodes)))
        after_nodes_list.append(after_nodes)
    A_en_sliced.reverse()
    after_nodes_list.reverse()

    return A_en_sliced, after_nodes_list


def calc_prob(A, size, num_rels, device):
    vals = A._values()
    indices = A._indices()
    ones = torch.ones((size[0], 1))
    vals_sq = torch.mul(vals, vals)
    if device == 'cuda':
        values = torch.cuda.sparse.FloatTensor(indices, vals_sq, torch.Size(size))
    else:
        values = torch.sparse.FloatTensor(indices, vals_sq, torch.Size(size))
    sums = torch.spmm(values.t(), ones)
    return sums.view(A.size()[1], 1)


def sum_prob_per_rel(pi, num_nodes, num_rels):
    sums = torch.sum(pi.view(num_rels, num_nodes), 1)
    nonzero_rels = torch.squeeze(torch.nonzero(sums))
    pp = pi.t() / sums.repeat_interleave(num_nodes)

    return torch.squeeze(pp, 0), nonzero_rels


def sel_idx_node(pp, s_num, num_nodes, num_rels): #TODO: vectorize?
    total_idx = []
    global_idx = []
    prob_list = []
    # get the nodes and relations that appear in the neighbourhood of the target nodes
    non_zero_node_idx = (pp.isnan() == False).nonzero(as_tuple=True)[0]
    non_zero_rels = torch.unique((non_zero_node_idx - non_zero_node_idx % num_nodes)/num_nodes)

    filtered_p = pp[~pp.isnan()]
    r = len(filtered_p)//num_nodes
    p_2d = torch.reshape(filtered_p, (r, num_nodes))
    print(torch.count_nonzero(p_2d, dim=1))
    # Turning LDRN to LARN
    # if torch.count_nonzero(p_2d, dim=1) > 0:
    #     p_2d[p_2d != 0.0] = 1 / torch.count_nonzero(p_2d, dim=1)
    # percentage of the rels that have more nonzero degree nodes than the sample size
    rels_more_s_num = sum(torch.count_nonzero(p_2d, dim=1) > s_num)/len(non_zero_rels)*100
    s_n = torch.minimum(torch.mul(torch.ones(r), s_num), torch.count_nonzero(p_2d, dim=1))
    for ri in range(r):
        idx = p_2d[ri].multinomial(int(s_n[ri]), replacement=False)
        total_idx.append(idx)
        global_idx.append(idx+non_zero_rels[ri]*num_nodes)
        prob_list.append(p_2d[ri][idx])

    total_idx = torch.unique(torch.cat(total_idx))

    return total_idx, non_zero_rels, torch.cat(global_idx), torch.cat(prob_list), rels_more_s_num


def sel_idx_edge(pp, s_num, num_nodes, num_rels, nonzero_rels): #TODO: implement, r matters
    total_idx_ext = []  # global idx
    total_idx = []  # local to the relation
    filtered_p = pp[~pp.isnan()]
    r = len(filtered_p)//num_nodes
    p_2d = torch.reshape(filtered_p, (r, num_nodes))
    print(torch.count_nonzero(p_2d, dim=1))
    # percentage of the rels that have more nonzero degree nodes than the sample size
    rels_more_s_num = sum(torch.count_nonzero(p_2d, dim=1) > s_num) / num_rels * 100
    s_n = torch.minimum(torch.mul(torch.ones(r), s_num), torch.count_nonzero(p_2d, dim=1))
    s_n[-1] = torch.count_nonzero(p_2d[r-1])  # Sample all edges for the self-loops (the last relation)

    for ri in range(r):
        idx = p_2d[ri].multinomial(int(s_n[ri]), replacement=False)
        sorted_idx, _ = torch.sort(idx)
        total_idx_ext.append(nonzero_rels[ri]*num_nodes+sorted_idx)
        total_idx.append(sorted_idx)

    total_idx_ext = torch.cat(total_idx_ext)
    total_idx_all = torch.unique(torch.cat(total_idx))  # concatenated local idx

    return total_idx_ext, total_idx_all, total_idx, s_n, rels_more_s_num


# Was trying to implement per relation separately. To continue later
def sel_idx_per_rel(pp, s_num, num_nodes, num_rels):
    # Separately sampling nodes per relations and not aggregating them
    total_idx = []
    filtered_p = pp[~pp.isnan()]
    r = len(filtered_p)//num_nodes
    p_2d = torch.reshape(filtered_p, (r, num_nodes))
    s_n = torch.minimum(torch.mul(torch.ones(r), s_num), torch.count_nonzero(p_2d, dim=1))
    for ri in range(r):
        idx = p_2d[ri].multinomial(int(s_n[ri]), replacement=False)
        rel_idx = ri*num_nodes + idx
        total_idx.append(rel_idx)

    return total_idx


def slice_adj_row_col(A, row_ind, col_ind, num_prev_nodes, num_after_nodes, mode):
    n, rn = A.size()
    r = rn // n
    row_isin = torch.isin(A._indices()[0], row_ind)
    col_isin = torch.isin(A._indices()[1], col_ind)
    # row, col = A._indices()[:, torch.where(torch.isin(A._indices()[0], idx))[0]] #use to work!
    row_and_col = torch.logical_and(col_isin, row_isin)
    row, col = A._indices()[:, torch.where(row_and_col)[0]]
    if mode == 'prob':
        # add self-loops to make A_r square
        self_loops = torch.unique(row)
        self_loops_r = self_loops + (r-1) * n
        row = torch.cat((row[0:-num_after_nodes], self_loops))
        col = torch.cat((col[0:-num_after_nodes], self_loops_r))
        col_ind = torch.unique(torch.cat((col, col_ind)))
        # size is now square because we added the self-loops
        col_size = num_prev_nodes

        col_mx = torch.max(row_ind) + 1
        map = torch.empty((int(col_mx),), dtype=torch.long)
        map1 = map.scatter_(0, row_ind, torch.arange(len(row_ind)))
        idx_m = torch.remainder(col, n).long()
        idx_d = torch.floor_divide(col, n).long()
        col = map1[idx_m] + idx_d * num_prev_nodes

        # this doesn't work because we want square matrix with self-loops
        # col_mx = torch.max(col_ind) + 1
        # map = torch.empty((int(col_mx),), dtype=torch.long)
        # map1 = map.scatter_(0, col_ind, torch.arange(len(col_ind)))
        # col = torch.gather(map1, 0, col)

    else:
        col_size = num_after_nodes

        col_mx = torch.max(col_ind[0:num_after_nodes]) + 1
        map = torch.empty((int(col_mx),), dtype=torch.long)
        map1 = map.scatter_(0, col_ind[0:num_after_nodes], torch.arange(num_after_nodes))
        idx_m = torch.remainder(col, n).long()
        idx_d = torch.floor_divide(col, n).long()
        col = map1[idx_m] + idx_d * num_after_nodes

        # equivalent to this:
        # mx2 = torch.max(col_ind) + 1
        # map2 = torch.empty((int(mx2),), dtype=torch.long)
        # map12 = map2.scatter_(0, col_ind, torch.arange(len(col_ind)))
        # col2 = torch.gather(map12, 0, col)

    row_mx = torch.max(row_ind) + 1
    map = torch.empty((int(row_mx),), dtype=torch.long)
    map1 = map.scatter_(0, row_ind, torch.arange(len(row_ind)))
    row = torch.gather(map1, 0, row)

    indices = torch.vstack([row, col])
    size = [num_prev_nodes, col_size * r]
    vals = torch.ones(len(row))
    values = torch.sparse.FloatTensor(indices, vals, torch.Size(size))
    ones = torch.ones(r, col_size, 1)
    block_mat = torch.block_diag(*ones)
    sums = torch.squeeze(torch.spmm(values, block_mat))
    sums_a = sums[indices[0], torch.div(indices[1], col_size, rounding_mode='floor')]
    vals_norm = torch.div(vals, sums_a)

    return torch.sparse_coo_tensor(indices, vals_norm, size)


def slice_adj_col(A, col_ind, num_rels, num_prev_nodes, sampler, after_nodes, num_after_nodes, global_idx, probs):
    idx = col_ind[-1]
    indices_sliced = torch.where(torch.isin(A._indices()[1], idx))[0]
    row, col = A._indices()[:, indices_sliced]
    vals = torch.ones(len(row))
    if sampler == 'LDRE':
        mx = torch.max(after_nodes) + 1
        map = torch.empty((int(mx),), dtype=torch.long)
        map1 = map.scatter_(0, after_nodes, torch.arange(len(idx)))
        n = int(A.size(1) / num_rels)
        idx_m = torch.remainder(col, n).long()
        idx_d = torch.floor_divide(col, n).long()
        col = map1[idx_m] + idx_d * (len(after_nodes))
    else:
        mx = torch.max(idx) + 1
        map = torch.empty((int(mx),), dtype=torch.long)
        map1 = map.scatter_(0, idx, torch.arange(len(idx)))
        col = torch.gather(map1, 0, col)

    indices = torch.vstack([row, col])
    size = [num_prev_nodes, len(after_nodes) * num_rels]
    if num_rels > 1:
        ones = torch.ones(num_rels, len(after_nodes), 1)

    unbiased = False
    if unbiased:
        probss = torch.ones(max(A._indices()[1].max().item(), int(global_idx.max())) + 1)
        probss[global_idx.type(torch.int64)] = 1 / probs
        prob_vals = probss[A._indices()[1]]
        vals_unbiased = torch.mul(torch.ones(len(A._values())), prob_vals)

        values = torch.sparse.FloatTensor(indices, vals_unbiased[indices_sliced], torch.Size(size))
        # row normalize the values of Adj
        block_mat = torch.block_diag(*ones)
        sums = torch.squeeze(torch.spmm(values, block_mat))
        sums_a = sums[indices[0], torch.div(indices[1], num_after_nodes, rounding_mode='floor')]
        vals_norm = torch.div(vals_unbiased[indices_sliced], sums_a)
    else:
        values = torch.sparse.FloatTensor(indices, vals, torch.Size(size))
        # row normalize the values of Adj
        block_mat = torch.block_diag(*ones)
        sums = torch.squeeze(torch.spmm(values, block_mat))
        sums_a = sums[indices[0], torch.div(indices[1], len(after_nodes), rounding_mode='floor')]
        vals_norm = torch.div(vals, sums_a)

    return torch.sparse_coo_tensor(indices, vals_norm, size)


def rand_sel_idx(pp, s_num, num_nodes, num_rels):
    total_idx = []
    filtered_p = pp[~pp.isnan()]
    r = len(filtered_p) // num_nodes
    p_2d = torch.reshape(filtered_p, (r, num_nodes))
    s_n = torch.minimum(torch.mul(torch.ones(r), s_num), torch.count_nonzero(p_2d, dim=1))
    p_2d = torch.ones_like(p_2d)

    for ri in range(r):
        idx = p_2d[ri].multinomial(int(s_n[ri]), replacement=False)
        total_idx.append(idx)

    total_idx = torch.unique(torch.cat(total_idx))

    return total_idx


def get_sampled_rels(A_en_sliced, num_rels):
    A_0 = A_en_sliced[0]
    A_1 = A_en_sliced[1]
    num_nodes0 = A_0.size(1)/num_rels
    num_nodes1 = A_1.size(1) / num_rels
    nodes_in_rels0 = torch.zeros(num_rels, dtype=torch.long)
    nodes_in_rels1 = torch.zeros(num_rels, dtype=torch.long)
    # how many nodes per layer per relation are sampled
    col0 = torch.unique(A_0._indices()[1])
    unique0, counts0 = torch.unique(torch.floor_divide(col0, num_nodes0).long(), return_counts=True)
    nodes_in_rels0[unique0] = counts0

    col1 = torch.unique(A_1._indices()[1])
    unique1, counts1 = torch.unique(torch.floor_divide(col1, num_nodes1).long(), return_counts=True)
    nodes_in_rels1[unique1] = counts1

    return [nodes_in_rels0, nodes_in_rels1]


def plot_graph(batch_node_idx_s, data, after_nodes_list, batch_out_train, batch_y_train_s, y_train):

    # Initialize graph
    graph = nx.DiGraph()
    correct_target = []
    label_dict = {0: 'accommodation', 1: 'facility', 2: 'location', 3: 'room', 4: 'no'}
    for i, target in enumerate(batch_node_idx_s):
        if batch_out_train[i].argmax() == batch_y_train_s[i]:
            correct_target.append(target)
            target_name = data.i2e[target][0].split('/')[-1].split('#')[-1]
            target_label = data.training[torch.nonzero(data.training[:,0] == target).squeeze(), 1]
            graph.add_node(target_name, label=label_dict[target_label.item()], color='red')
        else:
            target_name = data.i2e[target][0].split('/')[-1].split('#')[-1]
            target_label = data.training[torch.nonzero(data.training[:, 0] == target).squeeze(), 1]
            graph.add_node(target_name, label=label_dict[target_label.item()], color='pink')
        # Add first-hop neighbors
    for neighbor in after_nodes_list[1]:
        if neighbor not in batch_node_idx_s:
            neighbor_name = data.i2e[neighbor][0].split('/')[-1].split('#')[-1]
            if neighbor in data.training[:, 0]:
                neighbor_label = data.training[
                torch.nonzero(data.training[:, 0] == neighbor).squeeze(), 1]
            elif neighbor in data.withheld[:, 0]:
                neighbor_label = data.withheld[
                    torch.nonzero(data.withheld[:, 0] == neighbor).squeeze(), 1]
            else:
                neighbor_label = y_train.max() + 1
            graph.add_node(neighbor_name, label=label_dict[neighbor_label.item()], color='blue')
    # Add second-hop neighbors
    for neighbor in after_nodes_list[0]:
        if neighbor not in after_nodes_list[1] and neighbor not in batch_node_idx_s:
            neighbor_name = data.i2e[neighbor.item()][0].split('/')[-1].split('#')[-1]
            if neighbor in data.training[:, 0]:
                neighbor_label = data.training[
                    torch.nonzero(data.training[:, 0] == neighbor).squeeze(), 1]
            elif neighbor in data.withheld[:, 0]:
                neighbor_label = data.withheld[
                    torch.nonzero(data.withheld[:, 0] == neighbor).squeeze(), 1]
            else:
                neighbor_label = y_train.max() + 1
            graph.add_node(neighbor_name, label=label_dict[neighbor_label.item()], color='green')

    for s, r, o in data.triples:
        if s in batch_node_idx_s or s in after_nodes_list[0] or s in after_nodes_list[1]:
            neighbor_name = data.i2e[o][0].split('/')[-1].split('#')[-1]
            relation_name = data.i2r[r].split('/')[-1].split('#')[-1]
            if o in data.training[:, 0]:
                neighbor_label = data.training[
                    torch.nonzero(data.training[:, 0] == o).squeeze(), 1]
            elif o in data.withheld[:, 0]:
                neighbor_label = data.withheld[
                    torch.nonzero(data.withheld[:, 0] == o).squeeze(), 1]
            else:
                neighbor_label = y_train.max() + 1
            if o in batch_node_idx_s or o in after_nodes_list[1] or o in after_nodes_list[0]:
                graph.add_edge(data.i2e[s][0].split('/')[-1].split('#')[-1], neighbor_name,
                            label=relation_name, color='black')
            else:
                graph.add_node(neighbor_name, label=label_dict[neighbor_label.item()], color='grey')
                graph.add_edge(data.i2e[s][0].split('/')[-1].split('#')[-1], neighbor_name,
                               label=relation_name, color='grey')
        # if o in batch_node_idx_s or o in after_nodes_list[0] or o in after_nodes_list[1]:
        #     neighbor_name = data.i2e[o][0].split('/')[-1].split('#')[-1]
        #     relation_name = data.i2r[r].split('/')[-1].split('#')[-1]
        #     if o in data.training[:, 0]:
        #         neighbor_label = data.training[
        #             torch.nonzero(data.training[:, 0] == o).squeeze(), 1]
        #     elif o in data.withheld[:, 0]:
        #         neighbor_label = data.withheld[
        #             torch.nonzero(data.withheld[:, 0] == o).squeeze(), 1]
        #     else:
        #         neighbor_label = y_train.max() + 1
        #     if s in after_nodes_list[1] or s in after_nodes_list[0] or s in batch_node_idx_s:
        #         graph.add_edge(data.i2e[s][0].split('/')[-1].split('#')[-1], neighbor_name,
        #                        label=relation_name, color='black')
        #     else:
        #         graph.add_node(neighbor_name, label=label_dict[neighbor_label.item()], color='grey')
        #         graph.add_edge(data.i2e[s][0].split('/')[-1].split('#')[-1], neighbor_name,
        #                        label=relation_name, color='grey')

    # Visualization
    pos = nx.spring_layout(graph, k=1.5, iterations=50)  # or other layout algorithms like nx.circular_layout

    # Extract node colors
    node_colors = [data['color'] for _, data in graph.nodes(data=True)]
    edge_colors = [data['color'] for _, _, data in graph.edges(data=True)]
    plt.figure()
    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=500)

    # Draw edges
    nx.draw_networkx_edges(graph, pos, arrows=True, edge_color=edge_colors)

    # Draw node labels
    node_labels = {node: f"{node} ({data['label']})" for node, data in graph.nodes(data=True)}
    nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=5, font_color='black')

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(graph, 'label')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=5, font_color='grey')
    plt.show()