import torch
import torchvision
import time
import sys
from os import listdir
from os.path import isfile, join
import numpy as np

from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten


class LeNet(Module):
	def __init__(self, numChannels, classes):
		# call the parent constructor
		super(LeNet, self).__init__()
		# initialize first set of CONV => RELU => POOL layers
		self.conv1 = Conv2d(in_channels=numChannels, out_channels=20,
			kernel_size=(5, 5))
		self.relu1 = ReLU()
		self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		# initialize second set of CONV => RELU => POOL layers
		self.conv2 = Conv2d(in_channels=20, out_channels=50,
			kernel_size=(5, 5))
		self.relu2 = ReLU()
		self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		# initialize first (and only) set of FC => RELU layers
		self.fc1 = Linear(in_features=800, out_features=500)
		self.relu3 = ReLU()
		# initialize our softmax classifier
		self.fc2 = Linear(in_features=500, out_features=classes)
		self.logSoftmax = LogSoftmax(dim=1)

	def forward(self, x):
		# pass the input through our first set of CONV => RELU =>
		# POOL layers
		x = self.conv1(x)
		x = self.relu1(x)
		x = self.maxpool1(x)
		# pass the output from the previous layer through the second
		# set of CONV => RELU => POOL layers
		x = self.conv2(x)
		x = self.relu2(x)
		x = self.maxpool2(x)
		# flatten the output from the previous layer and pass it
		# through our only set of FC => RELU layers
		x = flatten(x, 1)
		x = self.fc1(x)
		x = self.relu3(x)
		# pass the output to our softmax classifier to get our output
		# predictions
		x = self.fc2(x)
		output = self.logSoftmax(x)
		# return the output predictions
		return output    

def preprocess_image(channels=3):
    image = np.random.rand(224,224,3)
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
    image_data = np.expand_dims(image_data, 0)
    return image_data


if __name__ == "__main__":
    batch_size = sys.argv[1]
    batch_size = int(batch_size)

    # model = torchvision.models.alexnet()
    model = LeNet(numChannels = 3, classes = 2)
    model.cuda()
    model.train()


#     print("count")

    imgs = []

    np.random.seed(12345)
    img_const = preprocess_image(channels = 3)
    count = 64


    print("Processed Images")

    inputs = []
    for i in range(0,int(64/batch_size)):
        input = img_const
        for batch in range(1,batch_size):
            input = np.vstack((input, img_const))
        inputs.append(torch.from_numpy(input).cuda())
    

    print("TimePreModel --"+ str(time.time()))
    for i in range(0,25):
        for input in inputs:
            print("*****************")
            model(input)
