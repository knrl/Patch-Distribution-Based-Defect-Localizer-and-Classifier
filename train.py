#
#   @author: Mehmet Kaan Erol
#
#   For the PaDiM implementation used here: github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master/blob/main/main.py
import os
import random
import pickle
import numpy as np
from tqdm import tqdm
from random import sample

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2, resnet18
from torchvision.models import wide_resnet101_2

# Data-Manager
import dataset_manager.textile_dataset as TextileDataset

# configuration dictionary
from config import config_dict

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

def run(train_dataset_path, class_name, train_batch_size, train_feature_filepath, size):
    # load model
    model = wide_resnet50_2(pretrained=True, progress=True)
    t_d, d = 1792, 550

    # build model
    model.to(device)
    model.eval()
    random.seed(1024)
    torch.manual_seed(1024)
    if use_cuda:
        torch.cuda.manual_seed_all(1024)
    idx = torch.tensor(sample(range(0, t_d), d))

    outputs = []
    def hook(module, input, output):
        outputs.append(output)
    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)

    # Train
    train(
            model, 
            train_dataset_path, 
            outputs, 
            idx, 
            save=True, 
            train_batch_size=train_batch_size, 
            train_feature_filepath=train_feature_filepath, 
            size=size
        )

def feature_extractor(model, dataloader, t_outputs, outputs, idx):
    for (x,_) in tqdm(dataloader, '| feature extraction |'):
        with torch.no_grad():
            _ = model(x.to(device))
        for k, v in zip(t_outputs.keys(), outputs):
            t_outputs[k].append(v.cpu().detach())
        outputs = []
    for k, v in t_outputs.items():
        t_outputs[k] = torch.cat(v, 0)

    embedding_vectors = t_outputs['layer1']
    for layer_name in ['layer2', 'layer3']:
        embedding_vectors = embedding_concat(embedding_vectors, t_outputs[layer_name])

    embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
    return outputs, embedding_vectors

def train(model, dataset_path, outputs, idx, save, train_batch_size, train_feature_filepath, size):
    train_dataset = TextileDataset.TextileDataset(dataset_path=dataset_path,is_single=False,size=size)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, pin_memory=True)
    train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

    outputs, embedding_vectors = feature_extractor(model, train_dataloader, train_outputs, outputs, idx)
    B, C, H, W = embedding_vectors.size()
    embedding_vectors = embedding_vectors.view(B, C, H * W)
    mean = torch.mean(embedding_vectors, dim=0).numpy()
    cov = torch.zeros(C, C, H * W).numpy()
    I = np.identity(C)
    for i in range(H * W):
        cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I

    mean_i, conv_inv = [], []
    for i in range((size/4) * (size/4)):
        mean_i.append(mean[:, i])
        conv_inv.append(np.linalg.inv(cov[:, :, i]))

    train_outputs = [mean_i, conv_inv]
    if (save):
        with open(train_feature_filepath, 'wb') as f:
            pickle.dump(train_outputs, f, protocol=4)

def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)
    return z

if (__name__ == '__main__'):
    run(
            train_dataset_path = config_dict['train_dataset_path'],
            class_name = config_dict['class_name'],
            train_batch_size = config_dict['train_batch_size'],
            train_feature_filepath = config_dict['train_feature_filepath'],
            size = config_dict['size'],
        )
