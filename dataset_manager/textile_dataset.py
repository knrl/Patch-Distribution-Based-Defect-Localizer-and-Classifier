#
#   @author: Mehmet Kaan Erol
#
import os
import cv2
import numpy as np
from PIL import Image
from skimage import filters
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

class TextileDataset(Dataset):
	def __init__(self, dataset_path='/home/ubuntu/dataset/', is_single=False, size=256, train=True):
		self.dataset_path = dataset_path
		self.is_single = is_single
		self.train = train

		# load dataset
		self.x = self.read_data()
		self.y_good, self.y_defect = [], []

		self.transform_x = T.Compose([T.Resize(size, Image.ANTIALIAS),
									T.CenterCrop(size),
									T.ToTensor(),
									T.Normalize(mean=[0.485, 0.456, 0.406],
									std=[0.229, 0.224, 0.225])])
		self.transform_mask = T.Compose([T.Resize(size, Image.NEAREST),
									T.CenterCrop(size),
									T.ToTensor()])

	def __getitem__(self, idx):
		x = self.x[idx]
		y = 1
		if (x in self.y_good):
			y = 0
		x = Image.open(x).convert('RGB')
		x = self.transform_x(x)	
		return x, y

	def __len__(self):
		return len(self.x)

	def read_data(self):
		self.images_path = self.dataset_path
		x = []

		if (self.is_single):
			x.append(self.images_path)
		else:
			if (self.train):
				img_dir = self.images_path
				img_fpath_list = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')]
				x.extend(img_fpath_list)
				img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
				x = list(sorted(x))
			else:
				x1, x2 = [], []
				img_dir = os.path.join(self.images_path,'good')
				img_fpath_list = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')]
				x1.extend(img_fpath_list)
				img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
				self.y_good = img_fname_list

				img_dir = os.path.join(self.images_path,'defect')
				img_fpath_list = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')]
				x2.extend(img_fpath_list)
				img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
				self.y_defect = img_fname_list
				x = x1 + x2
				x = list(sorted(x))
		return x
