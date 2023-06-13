import cv2
import numpy as np
import sys
import time
import torch

batch_size = sys.argv[1]  # Specify the desired batch size
batch_size = int(batch_size)

# Generate random images
def preprocess_image(channels=3):
    image = np.random.rand(224, 224, channels)
    image_data = np.asarray(image).astype(np.float32)
    image_data = image_data.transpose([2, 0, 1])  # Transpose to CHW
    mean = np.array([0.079, 0.05, 0]) + 0.406
    std = np.array([0.005, 0, 0.001]) + 0.224
    for channel in range(image_data.shape[0]):
        image_data[channel, :, :] = (image_data[channel, :, :] / 255 - mean[channel]) / std[channel]
    image_data = np.expand_dims(image_data, 0)
    return image_data

inputs = []
for i in range(0, int(64 / batch_size)):
    input_batch = preprocess_image(channels=3)
    for batch in range(1, batch_size):
        input_batch = np.vstack((input_batch, preprocess_image(channels=3)))
    inputs.append(input_batch)

print("Processed Images")
# Load YOLO
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")

# Enable CUDA for faster inference
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Set CUDA device (optional)
cv2.cuda.setDevice(0)  # Replace 0 with the index of the desired GPU device
print("TimePostModelLoading --"+ str(time.time()))


print("TimePreModel --"+ str(time.time()))

# Perform batch inference
for i in range(0,50):

    for input_batch in inputs:
        # Set the blob as input to the network
        net.setInput(torch.from_numpy(input_batch).cuda())

        # Perform forward pass
        layer_outputs = net.forward(output_layers)
print("TimePostModel --"+ str(time.time()))

