import torch
import numpy as np
import toml, fire, sys

from kgbench import load
import kgbench as kg
import random
from encoders import extract_embeddings
import torch, time
import torch.nn as nn
from mrgcn import MRGCN_Batch, MRGCN_Full
from utils import enrich, adj, sum_sparse, adj_with_neighbours

import pickle as pkl
import numpy as np
from sklearn.metrics import accuracy_score
import os
import time
import pdb

import wandb


def go(project="test", name='50-100', data_name='amplus', batch_size=128, feat_size=16, load_epoch=10, modality='all',
       l2=5e-4, lr=0.01, lr_d=0, prune=True, final=False, embed_size=16,
       bases=40, sampler='ladies', depth=2, samp0=50, samp1=100, self_loop_dropout=0.2, test_state='full', iteration=10):
    config = {
        "project": project,
        "name": name,
        "data_name": data_name,
        "batch_size": batch_size, "feat_size": feat_size,
        "load_epoch": load_epoch,
        "modality": modality,
        "l2": l2, "lr": lr, "lr_d": lr_d, "prune": prune, "final": final, "embed_size": embed_size,
        "bases": bases, "sampler": sampler, "depth": depth, "samp0": samp0, "samp1": samp1,
        "self_loop_dropout": self_loop_dropout, "test_state": test_state, "iteration": iteration
    }

    wandb.init(project=project, name=name, config=config)
    train_accs = []
    test_accs = []
    # pdb.set_trace()
    samp_num_list = [samp0, samp1]

    if os.path.isfile(f"data_{data_name}.pkl"):
        print(f'Using cached data {data_name}.')
        with open(f"data_{data_name}.pkl", 'rb') as file:
            data = pkl.load(file)
    else:
        print(f'No cache found (or relation limit is set). Loading data {data_name} from scratch.')
        data = load(data_name, torch=True, prune_dist=2 if prune else None, final=final)
        with open(f'data_{data_name}.pkl', 'wb') as f:
            pkl.dump(data, f)
    data = kg.group(data)
    print("batch size:", batch_size)
    train_idx = data.training[:, 0]
    y_train = data.training[:, 1]
    test_idx = data.withheld[:, 0]
    y_test = data.withheld[:, 1]
    num_classes = data.num_classes
    num_nodes = data.num_entities
    num_rels = data.num_relations

    # pdb.set_trace()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # pdb.set_trace()
    # train_idx = train_idx.to(device)
    y_train = y_train.to(device)
    # test_idx = test_idx.to(device)
    y_test = y_test.to(device)

    load_filename = f'embeddings_bsize%s_tr-ladies%s-%s-drop%s_%s.pkl' % (batch_size, samp0, lr,
                                                                       self_loop_dropout, load_epoch)
    load_path = os.path.join('./savedModels', load_filename)
    with open(load_path, 'rb') as file:
        embed_X = pkl.load(file)

    if batch_size > 0:
        model = MRGCN_Batch(n=num_nodes, edges=data.triples, feat_size=feat_size, embed_size=embed_size,
                            modality=modality,
                            num_classes=num_classes, num_rels=2 * num_rels + 1, num_bases=bases, sampler=sampler,
                            depth=depth,
                            samp_num_list=samp_num_list, self_loop_dropout=self_loop_dropout)

    else:
        model = MRGCN_Full(n=num_nodes, edges=data.triples, feat_size=feat_size, embed_size=embed_size,
                           modality=modality,
                           num_classes=num_classes, num_rels=2 * num_rels + 1, num_bases=bases,
                           self_loop_dropout=self_loop_dropout)

    # model.to(device)

    criterion = nn.CrossEntropyLoss()

    ###### training
    print('start training!')

    num_train = len(y_train)
    train_num_batches = num_train // batch_size
    num_test = len(y_test)
    train_num_batches = max(num_train // batch_size + 1, 1)
    train_batch_size = min(batch_size, num_train)
    test_num_batches = max(num_test // batch_size + 1 , 1)
    test_batch_size = min(batch_size, num_test)
    print("device: ", device)

    load_filename = f'model_bsize%s_tr-ladies%s-%s-drop%s_%s.pth' % (batch_size, samp0, lr,
                                                                       self_loop_dropout, load_epoch)
    load_path = os.path.join('./savedModels', load_filename)
    model.load_state_dict(torch.load(load_path))
    with torch.no_grad():
        if batch_size > 0:
            # testing
            total_it_loss = list()
            total_it_acc = list()
            for i in range (iteration):
                with torch.no_grad():
                    model.eval()
                    loss_list = list()
                    acc_list = list()
                if test_state == 'ladies' or test_state == 'full-mini' or sampler == 'full-mini-batch':
                    for batch_id in range(0, test_num_batches):
                        if batch_id == test_num_batches - 1:
                            batch_node_idx = test_idx[batch_id * test_batch_size:]
                            batch_y_test = y_test[batch_id * test_batch_size:]
                        else:
                            batch_node_idx = test_idx[batch_id * test_batch_size:(batch_id + 1) * test_batch_size]
                            batch_y_test = y_test[batch_id * test_batch_size:(batch_id + 1) * test_batch_size]

                        batch_node_idx_s, id_sorted = batch_node_idx.sort()
                        batch_y_test_s = batch_y_test[id_sorted]

                        # assert sum(batch_y_test_s == y_test[batch_id:batch_id + test_batch_size]) == batch_size
                        # optimizer.zero_grad()
                        batch_out_test = model(data=data, embed_X=embed_X, batch_id=batch_node_idx_s, test_state=test_state
                                               , device=device)  # [batch_node_idx, :]

                        batch_loss_test = criterion(batch_out_test, batch_y_test_s)  ###### TODO: l2 penalty #######
                        with torch.no_grad():
                            batch_acc_test = (batch_out_test.argmax(
                                dim=1) == batch_y_test_s).sum().item() / test_batch_size * 100
                        # batch_acc_test = accuracy_score(batch_out_test.argmax(dim=-1).detach().cpu(),
                        #                                 batch_y_test.detach().cpu()) * 100
                        print("Testing Epoch: ", load_epoch, " , batch number: ", batch_id, "/", test_num_batches, "Accuracy:",
                              batch_acc_test)

                        batch_loss_test = float(batch_loss_test)
                        batch_acc_test = float(batch_acc_test)

                        loss_list.append(batch_loss_test)
                        acc_list.append(batch_acc_test)

                elif test_state == 'full':
                    out_test = model(data=data, embed_X=embed_X, batch_id=0, test_state=test_state
                                     , device=device)[test_idx, :]

                    loss_test = criterion(out_test, y_test)  ###### TODO: l2 penalty #######
                    # acc_test = accuracy_score(out_test.argmax(dim=-1).detach().cpu(),
                    #                                 y_test.detach().cpu()) * 100
                    with torch.no_grad():
                        acc_test = (out_test.argmax(dim=1) == y_test).sum().item() / test_idx.size(0) * 100
                    # pdb.set_trace()
                    loss_test = float(loss_test)
                    acc_test = float(acc_test)

                    loss_list.append(loss_test)
                    acc_list.append(acc_test)

                test_total_loss = np.mean(loss_list)
                test_total_acc = np.mean(acc_list)

                print("Epoch ", load_epoch, " final Test Accuracy: ", test_total_acc, "And Loss: ", test_total_loss)
                wandb.log({"test_epoch_loss": test_total_loss})
                wandb.log({"test_epoch_acc": test_total_acc})

                total_it_loss.append(test_total_loss)
                total_it_acc.append(test_total_acc)
                i = i + 1
            test_it_average_loss = np.mean(total_it_loss)
            test_it_average_acc = np.mean(total_it_acc)
            print("Average iterations Test Accuracy: ", test_it_average_acc, "And Loss: ", test_it_average_loss)
            wandb.log({"test_average_iteration_loss": test_it_average_loss})
            wandb.log({"test_average_iteration_acc": test_it_average_acc})

if __name__ == '__main__':
    print('arguments ', ' '.join(sys.argv))
    fire.Fire(go)




