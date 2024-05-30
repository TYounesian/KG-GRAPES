from models import RGCN2, Mini_Batch_RGCN, LADIES_Mini_Batch_ERGCN
from utils import *


class MRGCN_Batch(nn.Module):
    def __init__(self, n, edges, feat_size, embed_size, modality, num_classes, num_rels, num_bases, sampler, depth,
                 samp_num_list, self_loop_dropout):
        super().__init__()

        self.num_nodes = n
        self.num_classes = num_classes
        self.embed_size = embed_size
        self.modality = modality
        self.depth = depth
        self.samp_num_list = samp_num_list
        self.sampler = sampler
        self.self_loop_dropout = self_loop_dropout
        self.num_rels = num_rels

        edges_en_tr = enrich_with_drop(edges, self.self_loop_dropout, self.num_nodes,
                                    int((self.num_rels - 1) / 2))
        edges_en_ts = enrich_with_drop(edges, 0, self.num_nodes, int((self.num_rels - 1) / 2))

        # horizontal indices and sizes
        hor_en_ind_tr, hor_en_size_tr = adj(edges_en_tr, self.num_nodes, self.num_rels, vertical=False)
        ver_en_ind_tr, ver_en_size_tr = adj(edges_en_tr, self.num_nodes, self.num_rels, vertical=True)

        hor_en_ind_ts, hor_en_size_ts = adj(edges_en_ts, self.num_nodes, self.num_rels, vertical=False)
        ver_en_ind_ts, ver_en_size_ts = adj(edges_en_ts, self.num_nodes, self.num_rels, vertical=True)

        if self.sampler == "hetero":
            indices_tr = adj_separate(edges_en_tr, self.num_nodes, self.num_rels)
            pdb.set_trace()
            A_sep = [[]] * num_rels
            for i in range(num_rels):
                vals = torch.ones(len(indices_tr[i][:,0]), dtype=torch.float)
                ss = sum_sparse(indices_tr[i], vals, (n,n), True)
                values = vals/ss
                A_sep[i] = torch.sparse.FloatTensor(indices=indices_tr[i], values=values, size=(n,n))

            self.register_buffer('A_sep', A_sep)

        # ignoring the relations, making one adjacency matrix
        if self.sampler == 'LDUN':
            norel_ind_tr, norel_size_tr = adj_norel(edges_en_tr, self.num_nodes)

            norel_vals_en_tr = torch.ones(norel_ind_tr.size(0), dtype=torch.float)
            norel_vals_en_tr = norel_vals_en_tr / sum_sparse(norel_ind_tr, norel_vals_en_tr, norel_size_tr, True)

            norel_A_tr = torch.sparse.FloatTensor(indices=norel_ind_tr.t(), values=norel_vals_en_tr,
                                                          size=norel_size_tr)
            self.register_buffer('norel_A_tr', norel_A_tr)

        vals_en_tr = torch.ones(ver_en_ind_tr.size(0), dtype=torch.float)
        vals_en_tr = vals_en_tr / sum_sparse(ver_en_ind_tr, vals_en_tr, ver_en_size_tr, True)

        vals_en_ts = torch.ones(ver_en_ind_ts.size(0), dtype=torch.float)
        vals_en_ts = vals_en_ts / sum_sparse(ver_en_ind_ts, vals_en_ts, ver_en_size_ts, True)

        # constructing the horizontal and vertical Adj matrices
        horizontal_en_A_tr = torch.sparse.FloatTensor(indices=hor_en_ind_tr.t(), values=vals_en_tr, size=hor_en_size_tr)
        self.register_buffer('horizontal_en_A_tr', horizontal_en_A_tr)

        horizontal_en_A_ts = torch.sparse.FloatTensor(indices=hor_en_ind_ts.t(), values=vals_en_ts, size=hor_en_size_ts)
        self.register_buffer('horizontal_en_A_ts', horizontal_en_A_ts)

        if self.sampler == 'rgcn':
            self.batch_rgcn = Mini_Batch_RGCN(n, feat_size, embed_size, num_classes, num_rels, num_bases)
        else:
            self.batch_rgcn = LADIES_Mini_Batch_ERGCN(n, feat_size, embed_size, num_classes, num_rels, num_bases, sampler)

        # self.embedding = nn.Parameter(torch.FloatTensor(n, embed_size))
        # nn.init.xavier_uniform_(self.embedding, gain=nn.init.calculate_gain('relu'))

    def forward(self, data, embed_X, batch_id, test_state, device):
        out, num_node_needed, nodes_in_rels, num_edges, rels_more = self.sampler_forward(data, embed_X, batch_id, test_state, device)
        return out, num_node_needed, nodes_in_rels, num_edges, rels_more

    def sampler_forward(self, data, embed_X, batch_id, test_state, device):
        if test_state == 'full' and not self.training:
            self.batch_rgcn.to(device)
            embed_X_dev = None if embed_X is None else embed_X.to(device)
            horizontal_en_A_ts_dev = self.horizontal_en_A_ts.to(device)
            out = self.batch_rgcn(embed_X_dev, torch.empty(0), 0, torch.empty(0),
                                  horizontal_en_A_ts_dev, test_state)
            return out, None, None, None, None

        elif self.training:
            if self.sampler == 'full-mini-batch':
                A_en_sliced, after_nodes_list, rels_more = full_mini_sampler(batch_id, self.num_nodes,
                                                                  int((self.num_rels - 1) / 2),
                                                                  self.horizontal_en_A_tr, self.depth, device)
            elif self.sampler == 'grapes':
                A_en_sliced, after_nodes_list, idx_per_rel_list, nonzero_rel_list, rels_more = grapes_sampler(
                    batch_id,
                    self.samp_num_list,
                    self.num_nodes,
                    self.num_rels,
                    self.norel_A_tr,
                    self.horizontal_en_A_tr,
                    self.depth,
                    self.sampler,
                    device)
            elif self.sampler == 'LDUN':
                A_en_sliced, after_nodes_list, idx_per_rel_list, nonzero_rel_list, rels_more = ladies_norel_sampler(batch_id,
                                                                                                         self.samp_num_list,
                                                                                                         self.num_nodes,
                                                                                                         self.num_rels,
                                                                                                         self.norel_A_tr,
                                                                                                         self.horizontal_en_A_tr,
                                                                                                         self.depth,
                                                                                                         self.sampler,
                                                                                                         device)
            elif self.sampler == "LDRN" or self.sampler == "LDRE":
                A_en_sliced, after_nodes_list, idx_per_rel_list, nonzero_rel_list, rels_more = ladies_sampler(batch_id,
                                                                                                   self.samp_num_list,
                                                                                                   self.num_nodes,
                                                                                                   self.num_rels,
                                                                                                   self.horizontal_en_A_tr,
                                                                                                   self.depth,
                                                                                                   self.sampler, device)
            elif self.sampler == 'hetero':
                A_en_sliced, after_nodes_list = hetero_sampler(batch_id, self.samp_num_list, self.num_nodes,
                                                               self.num_rels, self.A_sep, self.depth, device)

            elif self.sampler == 'IARN':
                A_en_sliced, after_nodes_list, rels_more = random_sampler(batch_id, self.samp_num_list, self.num_nodes,
                                                               self.num_rels,
                                                               self.horizontal_en_A_tr, self.depth, device)
            elif self.sampler == 'IDRN':
                A_en_sliced, after_nodes_list, rels_more = fastgcn_plus_sampler(batch_id, self.samp_num_list, self.num_nodes,
                                                                     self.num_rels,
                                                                     self.horizontal_en_A_tr, self.depth, device)

        elif test_state == 'LDRN':
            A_en_sliced, after_nodes_list, idx_per_rel_list, nonzero_rel_list, rels_more = ladies_sampler(batch_id,
                                                                                                self.samp_num_list,
                                                                                                self.num_nodes,
                                                                                                self.num_rels,
                                                                                                self.horizontal_en_A_ts,
                                                                                                self.depth, device)

        elif test_state == 'full-mini':
            A_en_sliced, after_nodes_list = full_mini_sampler(batch_id, self.num_nodes,
                                                            int((self.num_rels - 1) / 2),
                                                            self.horizontal_en_A_ts, self.depth, device)

        for i in range(2):
            print(f'Number of sampled edges layer {i+1}: {A_en_sliced[abs(i-1)]._indices().size(1)}'
                f' / {self.horizontal_en_A_tr._indices().size(1)}')
            total_num_edges = A_en_sliced[0]._indices().size(1)+A_en_sliced[1]._indices().size(1)

        nodes_in_rels = get_sampled_rels(A_en_sliced, self.num_rels)
        nodes_in_rels_ratio = [nodes_in_rels[i] / after_nodes_list[i].size(0) for i in range(len(nodes_in_rels))]
        nodes_needed = [i for j in after_nodes_list for i in j]
        nodes_needed = torch.unique(torch.tensor(nodes_needed, dtype=torch.int64, device=device))
        em_X = embed_X[nodes_needed]
        num_nodes_needed = len(nodes_needed)
        print("Computing embeddings for: ", num_nodes_needed, "out of ", self.num_nodes)
        # calculate how many nodes per data type:
        for datatype in data.datatypes():
            n = len(data.get_strings_batch(nodes_needed, dtype=datatype))
            print(f'number of nodes with {datatype} datatype: {n}')

        em_X_dev = None if em_X is None else em_X.to(device)
        A_en_sliced_dev = [a.to(device) for a in A_en_sliced]
        after_nodes_list_dev = [a.to(device) for a in after_nodes_list]

        self.batch_rgcn.to(device)

        if self.sampler == self.sampler == "LDRN" or self.sampler == "LDRE" or self.sampler=='LRUN':
            out = self.batch_rgcn(em_X_dev, after_nodes_list_dev, nodes_needed, A_en_sliced_dev, self.horizontal_en_A_ts
                                  , test_state, idx_per_rel_list, nonzero_rel_list)
        else:
            out = self.batch_rgcn(em_X_dev, after_nodes_list_dev, nodes_needed, A_en_sliced_dev, self.horizontal_en_A_ts
                                  , test_state, [], [])

        return out, num_nodes_needed, nodes_in_rels_ratio, total_num_edges, rels_more


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
