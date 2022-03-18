#
#   @author: Mehmet Kaan Erol
#
import os
import random
import pickle
import numpy as np
from tqdm import tqdm
from random import sample
from skimage import morphology

from collections import OrderedDict
from sklearn.covariance import LedoitWolf
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import mahalanobis, euclidean
from skimage.segmentation import mark_boundaries

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2

# Data-Manager
import dataset_manager.textile_dataset as TextileDataset

# configuration dictionary
from config import config_dict

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

def run(test_dataset_path, class_name, test_batch_size, distance_metric, train_feature_filepath, is_single, size):
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

    # Dataset and Dataloader
    test_dataset = TextileDataset.TextileDataset(dataset_path=test_dataset_path,is_single=is_single,size=size,train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, pin_memory=True)
    test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

    # read trained model from train_feature_filepath
    with open(train_feature_filepath, 'rb') as f:
        train_outputs = pickle.load(f)

    # predict the test images
    outputs, test_imgs, y_list = [], [], []
    for (x,y) in test_dataloader:
        y_list.append(y)
        test_imgs.extend(x.cpu().detach().numpy())
        with torch.no_grad():
            _ = model(x.to(device))
        for k, v in zip(test_outputs.keys(), outputs):
            test_outputs[k].append(v.cpu().detach())
        outputs = []

    for k, v in test_outputs.items():
        test_outputs[k] = torch.cat(v, 0)

    embedding_vectors = embedding_concat(embedding_concat(test_outputs['layer1'], test_outputs['layer2']), test_outputs['layer3'])
    embedding_vectors = torch.index_select(embedding_vectors, 1, idx)

    B, C, H, W = embedding_vectors.size()
    embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()

    dist_list = []
    if (distance_metric == 'mahalonobis_no_sqrt'):
    for i in range(H * W):
            dist_list.append([mahal(sample[:, i], train_outputs[0][i], train_outputs[1][i], True) for sample in embedding_vectors])
    elif (distance_metric == 'euclidean'):
        for i in range(H * W):
            dist_list.append([euclidean(sample[:, i], train_outputs[0][i]) for sample in embedding_vectors])
    else:
        for i in range(H * W):
            dist_list.append([mahal(sample[:, i], train_outputs[0][i], train_outputs[1][i], False) for sample in embedding_vectors])

    dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)
    dist_list = torch.tensor(dist_list)
    score_map = F.interpolate(dist_list.unsqueeze(1), size=x.size(2), mode='bilinear',align_corners=False).squeeze().numpy()

    for i in range(score_map.shape[0]):
        score_map[i] = gaussian_filter(score_map[i], sigma=4)

    if (is_single):
        score_map = score_map.reshape(1, size, size)

    max_score = score_map.max()
    min_score = score_map.min()
    scores = (score_map - min_score) / (max_score - min_score)
    print('scores.shape ', scores.shape)
    return test_imgs, scores, y_list

def mahal(x=None, mn=None, inv_covmat=None, no_sqrt=True):
    x_mu = x - mn
    left = np.dot(x_mu, inv_covmat)
    mahal = np.dot(left, x_mu.T)
    if (no_sqrt):
        return mahal
    else:
        return mahal

def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = H1 // H2
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
            test_dataset_path = config_dict['test_dataset_path'], 
            class_name = config_dict['class_name'], 
            test_batch_size = config_dict['test_batch_size'],
            distance_metric=config_dict['distance_metric'],
            train_feature_filepath = config_dict['train_feature_filepath'],
            is_single = config_dict['is_single'],
            size = config_dict['size']
        )
