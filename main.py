import toml, fire, sys

from kgbench import load
import kgbench as kg
from encoders import extract_embeddings
import torch
import torch.nn as nn
from mrgcn import MRGCN_Batch, MRGCN_Full
import pickle as pkl
import numpy as np
import os
import time
import pdb
import math
from statistics import stdev
import wandb
from utils import *
# from memory_profiler import profile
#
# @profile
from datetime import datetime


def go(project="kg-g", data_name='amplus', batch_size=2048, feat_size=16, num_epochs=50, modality='no',
       l2=5e-4, lr_c=0.01, lr_d=0, lr_g=0.01, loss_coef=1e4, log_z_init=0., use_indicators=True, prune=True, final=True,
       embed_size=16, bases=40, sampler='LDRN', depth=2, samp0=2048, self_loop_dropout=0, test_state='full',
       testing=True, saving=False, repeat=5, lr_embed=0, log_wandb=False):
    config = {
        "project": project, "data_name": data_name, "batch_size": batch_size, "feat_size": feat_size,
        "num_epochs": num_epochs, "modality": modality, "l2": l2, "lr_c": lr_c, "lr_d": lr_d, "lr_g": lr_g,
        'loss_coef': loss_coef,
        "prune": prune, "final": final, "embed_size": embed_size, "bases": bases, "sampler": sampler, "depth": depth,
        "samp0": samp0, "self_loop_dropout": self_loop_dropout, "test_state": test_state, "testing": testing,
        "saving": saving, "repeat": repeat, "lr_embed": lr_embed
    }

    wandb.init(project=project, entity='tyou', mode='online' if log_wandb else 'disabled', config=config)

    if os.path.isfile(f"data_{data_name}_final_{final}.pkl"):
        print(f'Using cached data {data_name}.')
        with open(f"data_{data_name}_final_{final}.pkl", 'rb') as file:
            data = pkl.load(file)
    else:
        print(f'No cache found (or relation limit is set). Loading data {data_name} from scratch.')
        data = load(data_name, torch=True, prune_dist=2 if prune else None, final=final)
        with open(f'data_{data_name}_final_{final}.pkl', 'wb') as f:
            pkl.dump(data, f)
    data = kg.group(data)

    y_train = data.training[:, 1].long()
    train_idx = data.training[:, 0].long()
    y_test = data.withheld[:, 1].long()
    test_idx = data.withheld[:, 0].long()
    num_rels = data.num_relations

    multilabel = False
    if multilabel:
        y_train = data.training[:, 1:].long()
        y_test = data.withheld[:, 1:].long()
        data.num_classes = y_train.size(1)
        y_train = y_train.float()
        y_test = y_test.float()

    save_model_info = False
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    y_train = y_train.to(device)
    y_test = y_test.to(device)

    if use_indicators:
        num_indicators = depth + 1
    else:
        num_indicators = 0

    num_train = len(y_train)
    num_test = len(y_test)
    train_num_batches = max(math.ceil(num_train / batch_size), 1)
    train_batch_size = min(batch_size, num_train)
    test_num_batches = max(math.ceil(num_test / batch_size), 1)
    test_batch_size = min(batch_size, num_test)
    print("device: ", device)

    results = torch.empty(repeat)
    for i in range(repeat):
        if modality == 'all':  # multimodal (MRGCN)
            embed_X = extract_embeddings(data, feat_size)
        else:  # If RGCN
            embed_X = nn.Parameter(torch.FloatTensor(data.num_entities, feat_size), requires_grad=True)
            # nn.init.xavier_uniform_(embed_X, gain=nn.init.calculate_gain('relu'))
            nn.init.kaiming_normal_(embed_X, mode='fan_in')

        if batch_size > 0:
            model_c = MRGCN_Batch(n=data.num_entities, feat_size=feat_size, embed_size=embed_size,
                                  modality=modality,
                                  num_classes=data.num_classes, num_rels=2 * num_rels + 1, num_bases=bases,
                                  sampler=sampler,
                                  depth=depth)
            if sampler == 'grapes':
                model_g = MRGCN_Batch(n=data.num_entities, feat_size=feat_size + num_indicators,
                                      embed_size=embed_size, modality=modality, num_classes=1,
                                      num_rels=2 * num_rels + 1, num_bases=bases, sampler=sampler, depth=depth)
                model_z = MRGCN_Batch(n=data.num_entities, feat_size=feat_size,
                                      embed_size=embed_size, modality=modality, num_classes=1,
                                      num_rels=2 * num_rels + 1, num_bases=bases, sampler=sampler, depth=depth)
            else:
                model_g = model_z = model_c

        else:
            model_c = MRGCN_Full(n=data.num_entities, edges=data.triples, feat_size=feat_size, embed_size=embed_size,
                                 modality=modality,
                                 num_classes=data.num_classes, num_rels=2 * num_rels + 1, num_bases=bases,
                                 self_loop_dropout=self_loop_dropout)
        if multilabel:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        model_measure = 'comps'
        if modality == 'no':
            optimizer_c = torch.optim.Adam([{'params': model_c.parameters(), 'lr': lr_c},
                                            {'params': embed_X, 'lr': lr_c}], weight_decay=0)
            if sampler == 'grapes':
                # log_z = torch.tensor(log_z_init, requires_grad=True)
                optimizer_g = torch.optim.Adam(list(model_g.parameters()) + list(model_z.parameters()), lr=lr_g,
                                               weight_decay=0)
        else:
            optimizer_c = torch.optim.Adam(model_c.parameters(), lr=lr_c, weight_decay=0)

        # training
        adj_tr, adj_ts, adj_norel_tr, adj_norel_ts = adj_r_creator(data.triples, self_loop_dropout, data.num_entities,
                                                                   2 * num_rels + 1, sampler)

        indicator_features = torch.zeros((data.num_entities, num_indicators))
        print('start training!')
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        folder = f'sampled_{current_time}'
        # if not os.path.exists(folder):
        #     os.makedirs(folder)

        # # shuffle
        # shuffle_idx = torch.randperm(train_idx.size(0))
        # train_idx = train_idx[shuffle_idx]
        # y_train = y_train[shuffle_idx]

        for epoch in range(0, num_epochs):
            samp_num_list = [samp0, samp0]
            start_e = 0
            pert = False
            pert_ratio = 0.25
            loss_c = 0
            loss_g = 0
            acc = 0
            necessity = 0
            batch_acc_train_pert = 0
            num_nodes_list = list()
            total_num_edges = 0
            total_rels_more = 0
            if batch_size > 0:
                model_c.train()
                model_g.train()
                for batch_id in range(0, train_num_batches):
                    acc_batch = 0.
                    loss_batch = 0.
                    batch_loss_train = 0.
                    batch_pert_acc = 0.
                    batch_nec = 0.
                    start = time.time()
                    if batch_id == train_num_batches - 1:
                        batch_node_idx = train_idx[batch_id * train_batch_size:]
                        batch_y_train = y_train[batch_id * train_batch_size:]
                    else:
                        batch_node_idx = train_idx[batch_id * train_batch_size:(batch_id + 1) * train_batch_size]
                        batch_y_train = y_train[batch_id * train_batch_size:(batch_id + 1) * train_batch_size]

                    batch_node_idx_s, id_sorted = batch_node_idx.sort()
                    batch_y_train_s = batch_y_train[id_sorted]

                    indicator_features.zero_()
                    indicator_features[batch_node_idx_s, -1] = 1.0
                    adj_tr_sliced, after_nodes_list, idx_per_rel_list, nonzero_rel_list, rels_more, log_probs, log_z, \
                    statistics, adj_tr_pert, after_nodes_list_pert = \
                        sampler_func(sampler,
                                     batch_node_idx_s,
                                     data.num_entities,
                                     num_rels,
                                     adj_tr,
                                     adj_norel_tr,
                                     [],
                                     samp_num_list,
                                     depth,
                                     model_g,
                                     model_z,
                                     embed_X,
                                     indicator_features,
                                     epoch,
                                     start_e,
                                     num_epochs,
                                     pert,
                                     pert_ratio,
                                     device)

                    nodes_needed = [i for j in after_nodes_list for i in j]
                    nodes_needed = torch.unique(torch.tensor(nodes_needed, dtype=torch.int64, device=device))
                    print("Computing embeddings for: ", len(nodes_needed), "out of ", data.num_entities)

                    batch_out_train, nodes_in_rels = model_c(embed_X, adj_tr_sliced,
                                                             after_nodes_list, idx_per_rel_list,
                                                             nonzero_rel_list, test_state, device)
                    if pert and epoch == num_epochs - 1:
                        batch_out_train_pert, _ = model_c(embed_X, adj_tr_pert,
                                                          after_nodes_list_pert, idx_per_rel_list,
                                                          nonzero_rel_list, test_state, device)
                        with torch.no_grad():
                            batch_acc_train_pert = (batch_out_train_pert.argmax(
                                dim=1) == batch_y_train_s).sum().item() / len(batch_y_train_s) * 100
                            necessity = (torch.abs(F.softmax(batch_out_train)[batch_y_train_s] - F.softmax
                            (batch_out_train_pert)[batch_y_train_s])).sum() / data.num_classes
                        batch_pert_acc += batch_acc_train_pert
                        batch_nec += necessity

                    if save_model_info and epoch == 0:
                        if batch_id == 0:
                            nodes_in_rels_sum = nodes_in_rels
                        else:
                            nodes_in_rels_sum = [nodes_in_rels_sum[i] + nodes_in_rels[i] for i
                                                 in range(len(nodes_in_rels))]
                        if batch_id == train_num_batches - 1:
                            nodes_in_rels_sum = [nodes_in_rels_sum[i] / train_num_batches for i
                                                 in range(len(nodes_in_rels_sum))]

                            with open(f"{data_name}_nodes_in_rels.pkl", 'wb') as f:
                                pkl.dump(nodes_in_rels_sum, f)
                    optimizer_c.zero_grad()
                    batch_loss_train += criterion(batch_out_train, batch_y_train_s)
                    if l2 != 0.0 and modality == 'no':
                        batch_loss_train += batch_loss_train + l2 * embed_X.pow(2).sum()

                    with torch.no_grad():
                        if multilabel:
                            y_pred = batch_out_train > 0
                            y_true = batch_y_train_s > 0.5

                            tp = int((y_true & y_pred).sum())
                            fp = int((~y_true & y_pred).sum())
                            fn = int((y_true & ~y_pred).sum())

                            try:
                                precision = tp / (tp + fp)
                                recall = tp / (tp + fn)
                                batch_acc_train = 2 * (precision * recall) / (precision + recall) * 100
                            except ZeroDivisionError:
                                batch_acc_train = 0.
                        else:
                            batch_acc_train = (batch_out_train.argmax(dim=1) == batch_y_train_s).sum() / len(
                                batch_y_train_s) * 100
                    print(f'train batch time ({time.time() - start:.4}s).')
                    print(f'Repeat: {i}, Training Epoch: {epoch}, batch number: {batch_id}/{train_num_batches}, '
                          f'Accuracy: {batch_acc_train}, necessity: {necessity}, Acc train pert: {batch_acc_train_pert}')
                    acc_batch += batch_acc_train
                    loss_batch += batch_loss_train

                    draw = False
                    if epoch == num_epochs - 1 and draw:
                        file_path = os.path.join(folder,
                                                 f'{data_name}_sampled_train_epoch{epoch}_batch{batch_id}.pkl')
                        sampled_dict = {'targets': batch_node_idx_s, 'after_nodes_list': after_nodes_list,
                                        'out': batch_out_train, 'batch_y': batch_y_train_s}
                        torch.save(sampled_dict, file_path)

                    batch_loss_train = loss_batch
                    batch_loss_train.backward()
                    optimizer_c.step()
                    if sampler == 'grapes':
                        optimizer_g.zero_grad()
                        cost_gfn = batch_loss_train.detach()
                        tot_log_prob = sum(t.sum() for sublist in log_probs for t in sublist)
                        # Trajectory Balance loss
                        loss_g = (log_z + tot_log_prob + loss_coef * cost_gfn) ** 2

                        loss_g.backward()
                        optimizer_g.step()
                        batch_loss_g = loss_g.item()
                    else:
                        batch_loss_g = 0.
                        tot_log_prob = 0.
                        cost_gfn = 0.

                    wandb.log({'epoch': epoch,
                               'train_batch_loss_c': batch_loss_train,
                               'train_batch_loss_g': batch_loss_g,
                               'log_z': log_z,
                               'log_probs': tot_log_prob,
                               'diff': loss_coef * cost_gfn + tot_log_prob,
                               'batch_pert_acc': batch_pert_acc / len(batch_y_train),
                               'batch_necessity': batch_nec / len(batch_y_train)})
                    log_dict = {}
                    for j, stat in enumerate(statistics):
                        for key, value in stat[0].items():
                            log_dict[f"{key}_{j}"] = 0
                    for j, stat in enumerate(statistics):  # Loop through the 256 dictionaries
                        for st in stat:
                            for key, value in st.items():
                                log_dict[f"{key}_{j}"] += value
                    for key, value in log_dict.items():
                        log_dict[key] = value / len(statistics[0])
                    wandb.log(log_dict)

                    loss_c += batch_loss_train
                    loss_g += batch_loss_g
                    acc += acc_batch
                    num_nodes_list.append(len(nodes_needed))
                    total_rels_more += rels_more

                    layers_c = [model_c.batch_rgcn.comp1.to('cpu'), model_c.batch_rgcn.comp2.to('cpu')]
                    with open(f"train_{data_name}_comps.pkl", 'wb') as f:
                        pkl.dump(layers_c, f)

                epoch_tr_loss_c = loss_c / train_num_batches
                epoch_tr_loss_g = loss_g / train_num_batches
                epoch_tr_acc = acc / train_num_batches

                print(f'Repeat: {i}, Epoch: {epoch}, final Train Accuracy: {epoch_tr_acc}, Loss gc: '
                      f'{epoch_tr_loss_c.item()}, Loss gf: {epoch_tr_loss_g},  Average num nodes needed: '
                      f'{np.mean(num_nodes_list)}, Average num edges:{total_num_edges / train_num_batches}, '
                      f'rels_more: {total_rels_more / train_num_batches}')
                if i == 0:
                    log_dict = {'epoch': epoch,
                                'train_epoch_loss_c': epoch_tr_loss_c,
                                'train_epoch_loss_g': epoch_tr_loss_g,
                                'train_epoch_acc': epoch_tr_acc,
                                'num_nodes_needed': np.mean(num_nodes_list),
                                'num_edges': total_num_edges / train_num_batches}
                    wandb.log(log_dict)

                # Saving the model
                if saving and (epoch % (num_epochs // 3) == (num_epochs // 3 - 1) or epoch == (num_epochs - 1)):
                    save_filename = f'model_bsize%s_{data_name}_%s.pth' % (batch_size, epoch)
                    save_path = os.path.join('./savedModels', save_filename)
                    torch.save(model_c.state_dict(), save_path)
                    embed_file_name = f'embeddings_bsize%s_{data_name}_%s.pkl' % (batch_size, epoch)
                    save_path = os.path.join('./savedModels', embed_file_name)
                    with open(save_path, 'wb') as f:
                        pkl.dump(embed_X, f)

                # testing
                model_c.eval()
                model_g.eval()
                loss = 0
                acc = 0
                if (test_state == 'LDRN' or test_state == 'full-mini') and testing:
                    for batch_id in range(0, test_num_batches):
                        if batch_id == test_num_batches - 1:
                            batch_node_idx = test_idx[batch_id * test_batch_size:]
                            batch_y_test = y_test[batch_id * test_batch_size:]
                        else:
                            batch_node_idx = test_idx[batch_id * test_batch_size:(batch_id + 1) * test_batch_size]
                            batch_y_test = y_test[batch_id * test_batch_size:(batch_id + 1) * test_batch_size]

                        batch_node_idx_s, id_sorted = batch_node_idx.sort()
                        batch_y_test_s = batch_y_test[id_sorted]
                        # IMPORTANT: at the moment sampling in test is not done per node, but still per batch_size
                        # samp_num_list = [test_batch_size, test_batch_size]
                        indicator_features.zero_()
                        indicator_features[batch_node_idx_s, -1] = 1.0
                        adj_ts_sliced, after_nodes_list, idx_per_rel_list, nonzero_rel_list, rels_more, log_probs, \
                        log_z, _, _, _ = sampler_func(sampler,
                                                      batch_node_idx_s,
                                                      data.num_entities,
                                                      num_rels,
                                                      adj_ts,
                                                      adj_norel_ts,
                                                      [],
                                                      samp_num_list,
                                                      depth,
                                                      model_g,
                                                      model_z,
                                                      embed_X,
                                                      indicator_features,
                                                      epoch,
                                                      start_e,
                                                      num_epochs,
                                                      False,
                                                      pert_ratio,
                                                      device)
                        batch_out_test, _ = model_c(embed_X, adj_ts_sliced,
                                                    after_nodes_list, idx_per_rel_list,
                                                    nonzero_rel_list, test_state, device)

                        batch_loss_test = criterion(batch_out_test, batch_y_test_s)  ###### TODO: l2 penalty #######
                        with torch.no_grad():
                            if multilabel:
                                y_pred = batch_out_test > 0
                                y_true = batch_y_test_s > 0.5

                                tp = int((y_true & y_pred).sum())
                                fp = int((~y_true & y_pred).sum())
                                fn = int((y_true & ~y_pred).sum())

                                try:
                                    precision = tp / (tp + fp)
                                    recall = tp / (tp + fn)
                                    batch_acc_test = 2 * (precision * recall) / (precision + recall) * 100
                                except ZeroDivisionError:
                                    batch_acc_test = 0.
                            else:
                                batch_acc_test = (batch_out_test.argmax(dim=1) == batch_y_test_s).sum().item() / len(
                                    batch_y_test_s) * 100
                        print("Repeat", i, ", Testing Epoch: ", epoch, " , batch number: ", batch_id, "/",
                              test_num_batches, "Accuracy:",
                              batch_acc_test)

                        loss += batch_loss_test
                        acc += batch_acc_test

                        if epoch == num_epochs - 1 and draw:
                            # plot_graph(batch_node_idx_s, data, after_nodes_list, batch_out_train, batch_y_train_s,
                            # y_train)
                            file_path = os.path.join(folder,
                                                     f'{data_name}_sampled_test_epoch{epoch}_batch{batch_id}.pkl')
                            sampled_dict = {'targets': batch_node_idx_s, 'after_nodes_list': after_nodes_list,
                                            'out': batch_out_train, 'batch_y': batch_y_train_s}
                            with open(file_path, 'wb') as f:
                                pkl.dump(sampled_dict, f)

                        layers_c = [model_c.batch_rgcn.comp1.to('cpu'), model_c.batch_rgcn.comp2.to('cpu')]
                        with open(f"test_{data_name}_comps.pkl", 'wb') as f:
                            pkl.dump(layers_c, f)

                    loss_test = loss / test_num_batches
                    acc_test = acc / test_num_batches
                elif test_state == 'full' and testing:
                    out, _ = model_c(embed_X, adj_ts.to(device), [], [], [], test_state, device=device)
                    out_test = out[test_idx, :]

                    loss_test = criterion(out_test, y_test)
                    with torch.no_grad():
                        acc_test = (out_test.argmax(dim=1) == y_test).sum().item() / test_idx.size(0) * 100

                epoch_ts_loss = loss_test.detach()
                epoch_ts_acc = acc_test

                print("Repeat", i, ", Epoch ", epoch, " final Test Accuracy: ", epoch_ts_acc,
                      "And Loss: ", epoch_ts_loss.item())
                if i == 0:
                    wandb.log({"test_epoch_loss": epoch_ts_loss, 'epoch': epoch})
                    wandb.log({"test_epoch_acc": epoch_ts_acc, 'epoch': epoch})

                results[i] = epoch_ts_acc
            # FULL BATCH #
            # Training
            else:
                start = time.time()
                if torch.cuda.is_available():
                    print('Using cuda.')
                    model_c.cuda()

                    train_idx = train_idx.cuda()
                    test_idx = test_idx.cuda()

                optimizer_c.zero_grad()
                out_train = model_c(embed_X=embed_X)[train_idx, :]
                # Get initial weigh norms for each layer
                if epoch == 0:
                    layer1_norm_e0 = torch.linalg.matrix_norm(
                        torch.einsum('rb, beh -> reh', model_c.rgcn.comps1, model_c.rgcn.bases1))
                    layer2_norm_e0 = torch.linalg.matrix_norm(
                        torch.einsum('rb, bho -> rho', model_c.rgcn.comps2, model_c.rgcn.bases2))

                loss_train = criterion(out_train, y_train)
                if l2 != 0.0 and modality == 'no':
                    loss_train = loss_train + l2 * embed_X.pow(2).sum()

                with torch.no_grad():
                    acc_train = (out_train.argmax(dim=1) == y_train).sum().item() / train_idx.size(0) * 100

                loss_train.backward()
                optimizer_c.step()

                print(f'train epoch time ({time.time() - start:.4}s).')
                print("Repeat", i, "Epoch ", epoch, " Train Accuracy: ", acc_train, "And Loss: ", loss_train.item())
                if i == 0:
                    wandb.log({"train_epoch_loss": loss_train.detach(), 'epoch': epoch})
                    wandb.log({"train_epoch_acc": acc_train, 'epoch': epoch})
                # Get layer norm in the end of training, subtract the initial norm and save
                if save_model_info and epoch == num_epochs - 1:
                    layer1_norm_end = torch.linalg.matrix_norm(
                        torch.einsum('rb, beh -> reh', model_c.rgcn.comps1, model_c.rgcn.bases1))
                    layer2_norm_end = torch.linalg.matrix_norm(
                        torch.einsum('rb, bho -> rho', model_c.rgcn.comps2, model_c.rgcn.bases2))
                    layer1_norm = layer1_norm_end - layer1_norm_e0
                    layer2_norm = layer2_norm_end - layer2_norm_e0
                    layer_norm = [layer1_norm, layer2_norm]
                    with open(f"{data_name}_weights.pkl", 'wb') as f:
                        pkl.dump(layer_norm, f)

                # Testing
                with torch.no_grad():
                    model_c.eval()

                out_test = model_c(embed_X=embed_X)[test_idx, :]

                loss_test = criterion(out_test, y_test)
                with torch.no_grad():
                    epoch_ts_acc = (out_test.argmax(dim=1) == y_test).sum().item() / test_idx.size(0) * 100

                print("Repeat", i, "Epoch ", epoch, " Test Accuracy: ", epoch_ts_acc, "And Loss: ", loss_test.item())
                if i == 0:
                    wandb.log({"test_epoch_loss": loss_test.detach(), 'epoch': epoch})
                    wandb.log({"test_epoch_acc": epoch_ts_acc, 'epoch': epoch})
                results[i] = epoch_ts_acc
    print(f'Acc: {results.mean():.2f} ± {results.std():.2f}')


if __name__ == '__main__':
    print('arguments ', ' '.join(sys.argv))
    fire.Fire(go)
