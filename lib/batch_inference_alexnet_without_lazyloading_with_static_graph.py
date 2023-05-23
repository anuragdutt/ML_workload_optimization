import torch
import torchvision
import time
import sys
from os import listdir
from os.path import isfile, join
import numpy as np


# mypath = '../data/images'
# files = [f for f in listdir(mypath) if isfile(join(mypath, f))]


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
	# print(batch_size)

	np.random.seed(123456789)
	img_const = preprocess_image(channels = 3)
	count = 64


	print("Processed Images")

	imgs = []
	inputs = []
	for i in range(0,int(64/batch_size)):
		input = img_const
		for batch in range(1,batch_size):
			input = np.vstack((input, img_const))
		inputs.append(torch.from_numpy(input).cuda())
	

	print("TimePreModelLoading --"+ str(time.time()))	
	model = torchvision.models.alexnet()
	model.cuda()
	
	model.eval()
	
	traced_model = torch.jit.trace(model, inputs[0])
	cuda_model = traced_model.cuda()
	print("TimePostModelLoading --"+ str(time.time()))
#     print("count")



	print("Lazy loading run --"+ str(time.time()))
	tmp = cuda_model(inputs[0])
	print("TimePreModel --"+ str(time.time()))
	for i in range(0,500):
		for input in inputs:
			tmp = cuda_model(input)
	
	print("TimePostModel --"+ str(time.time()))


