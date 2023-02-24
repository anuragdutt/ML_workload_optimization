import torch
import torchvision
import time
import sys
from os import listdir
from os.path import isfile, join
import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
# import torch.nn as nn
# from torch.nn import Module
# from torch.nn import Conv2d
# from torch.nn import Linear
# from torch.nn import MaxPool2d
# from torch.nn import ReLU
# from torch.nn import LogSoftmax
# from torch import flatten
# import torch.optim as optim

class LeNet_5(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2) # by padding=2, makes size of mnist data 28x28 to 32x32.
		self.pool1 = nn.AvgPool2d(kernel_size=2)
		self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
		self.pool2 = nn.AvgPool2d(kernel_size=2)
		self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
		self.fc2 = nn.Linear(in_features=120, out_features=84)
		self.fc3 = nn.Linear(in_features=84, out_features=10)

	def forward(self, input):
		x = self.conv1(input)
		x = F.relu(x)
		x = self.pool1(x)
		x = self.conv2(x)
		x = F.relu(x)
		x = self.pool2(x)
		x = x.view(-1,16*5*5)
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)
		output = self.fc3(x)
		return output

# class LeNet5(nn.Module):
#     def __init__(self, num_classes):
#         super(ConvNeuralNet, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
#             nn.BatchNorm2d(6),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size = 2, stride = 2))
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size = 2, stride = 2))
#         self.fc = nn.Linear(400, 120)
#         self.relu = nn.ReLU()
#         self.fc1 = nn.Linear(120, 84)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Linear(84, num_classes)
		
#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = out.reshape(out.size(0), -1)
#         out = self.fc(out)
#         out = self.relu(out)
#         out = self.fc1(out)
#         out = self.relu1(out)
#         out = self.fc2(out)
#         return out


	# def __init__(self, num_class=10, num_channel=1, include_top=True):
	# 	super(LeNet5, self).__init__()
	# 	self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
	# 	self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
	# 	self.relu = nn.ReLU()
	# 	self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
	# 	self.include_top = include_top
	# 	if self.include_top:
	# 		self.flatten = nn.Flatten()
	# 		self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
	# 		self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
	# 		self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))


	# def construct(self, x):
	# 	x = self.conv1(x)
	# 	x = self.relu(x)
	# 	x = self.max_pool2d(x)
	# 	x = self.conv2(x)
	# 	x = self.relu(x)
	# 	x = self.max_pool2d(x)
	# 	if not self.include_top:
	# 		return x
	# 	x = self.flatten(x)
	# 	x = self.relu(self.fc1(x))
	# 	x = self.relu(self.fc2(x))
	# 	x = self.fc3(x)
	# 	return x

def preprocess_image(channels=1):
	image = np.random.rand(28,28,1)
	# image = image.resize((width, height), Image.ANTIALIAS)
	image_data = np.asarray(image).astype(np.float32)
	#print(image_data.shape)
	image_data = image_data.transpose([2, 0, 1]) # transpose to CHW
	#print(image_data.shape)
	mean = np.array([0.079, 0.05, 0]) + 0.406
	std = np.array([0.005, 0, 0.001]) + 0.224
	for channel in range(image_data.shape[0]):
		#print(channel)
		image_data[channel, :, :] = (image_data[channel, :, :] / 255 - mean[channel]) / std[channel]
	# image_data = np.expand_dims(image_data, 0)
	return image_data


if __name__ == "__main__":
	batch_size = sys.argv[1]
	batch_size = int(batch_size)
	model = LeNet_5()
	model = torch.load("../data/models/checkpoint_20epoch.pth")
	model.cuda()
	model.eval()
	print(model)

	np.random.seed(12345)
	img_const = preprocess_image(channels = 1)
	print(np.shape(img_const))
	img = torch.from_numpy(img_const).cuda()
	model(img)
	# img_const = preprocess_image(channels = 1)
	# model(img_const)
	# count = 64


	# print("Processed Images")

	# inputs = []
	# for i in range(0,int(64/batch_size)):
	# 	input = img_const
	# 	for batch in range(1,batch_size):
	# 		input = np.vstack((input, img_const))
	# 	inputs.append(torch.from_numpy(input).cuda())
	

	# print("TimePreModel --"+ str(time.time()))
	# for i in range(0,2):
	# 	for input in inputs:
	# 		print("*****************")
	# 		model(input)
