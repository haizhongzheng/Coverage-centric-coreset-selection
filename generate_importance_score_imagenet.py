import torch
import numpy as np
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import os, sys
import argparse
import pickle
from core.data import CoresetSelection, IndexDataset, CIFARDataset, ImageNetDataset

parser = argparse.ArgumentParser()

######################### Data Setting #########################
parser.add_argument('--data-dir', type=str, default='../data/',
                    help='The dir path of the data.')
parser.add_argument('--base-dir', type=str)
parser.add_argument('--task-name', type=str)
parser.add_argument('--data-score-path', type=str)

args = parser.parse_args()

def EL2N(td_log, data_importance, max_epoch):
    l2_loss = torch.nn.MSELoss(reduction='none')

    def record_training_dynamics(td_log):
        output = torch.exp(td_log['output'].type(torch.float))
        index = td_log['idx'].type(torch.long)

        label = targets[index]
        label_onehot = torch.nn.functional.one_hot(label, num_classes=num_classes)
        el2n_score = torch.sqrt(l2_loss(label_onehot, output).sum(dim=1))

        data_importance['el2n'][index] += el2n_score

    for i, item in enumerate(td_log):
        if item['epoch'] == max_epoch:
            return
        record_training_dynamics(item)

def training_dynamics_metrics(td_log, data_importance):
    def record_training_dynamics(td_log):
        output = torch.exp(td_log['output'].type(torch.float))
        predicted = output.argmax(dim=1)
        index = td_log['idx'].type(torch.long)

        label = targets[index]

        correctness = (predicted == label).type(torch.int)
        data_importance['forgetting'][index] += torch.logical_and(data_importance['last_correctness'][index] == 1, correctness == 0)
        data_importance['last_correctness'][index] = correctness
        data_importance['correctness'][index] += data_importance['last_correctness'][index]

        batch_idx = range(output.shape[0])
        target_prob = output[batch_idx, label]
        output[batch_idx, label] = 0
        other_highest_prob = torch.max(output, dim=1)[0]
        margin = target_prob - other_highest_prob
        data_importance['accumulated_margin'][index] += margin

    for i, item in enumerate(td_log):
        record_training_dynamics(item)

#Load all data
data_dir = args.data_dir
trainset = ImageNetDataset.get_ImageNet_train(os.path.join(data_dir, 'train'))
trainset = IndexDataset(trainset)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=800, shuffle=False, pin_memory=True, num_workers=16)

# Load all targets into array
targets = []
print(f'Load label info from datasets...')
print(f'Total batch: {len(trainloader)}')
for batch_idx, (idx, (_, y)) in enumerate(trainloader):
    targets += list(y.numpy())
    if batch_idx % 50 == 0:
        print(batch_idx)

print(len(targets))
data_importance = {}
targets = torch.tensor(targets)
data_size = targets.shape[0]
num_classes = 1000

data_importance['targets'] = targets.type(torch.int32)
data_importance['el2n'] = torch.zeros(data_size).type(torch.float32)
data_importance['correctness'] = torch.zeros(data_size).type(torch.int32)
data_importance['forgetting'] = torch.zeros(data_size).type(torch.int32)
data_importance['last_correctness'] = torch.zeros(data_size).type(torch.int32)
data_importance['accumulated_margin'] = torch.zeros(data_size).type(torch.float32)

for i in range(1,11):
    td_path = f"{args.base_dir}/{args.task_name}/training-dynamics/td-{args.task_name}-epoch-{i}.pickle"
    print(td_path)
    with open(td_path, 'rb') as f:
         td_data = pickle.load(f)
    EL2N(td_data['training_dynamics'], data_importance, max_epoch=11)

for i in range(1,61):
    td_path = f"{args.base_dir}/{args.task_name}/training-dynamics/td-{args.task_name}-epoch-{i}.pickle"
    print(td_path)
    with open(td_path, 'rb') as f:
         td_data = pickle.load(f)
    training_dynamics_metrics(td_data['training_dynamics'], data_importance)

data_score_path = args.data_score_path
print(f'Saving data score at {data_score_path}')
with open(data_score_path, 'wb') as handle:
    pickle.dump(data_importance, handle)