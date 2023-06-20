import cv2
import numpy as np
import sys
import time
import torch

batch_size = sys.argv[1]  # Specify the desired batch size
batch_size = int(batch_size)

np.random.seed(123456789)


# Generate random images
def preprocess_image(channels=3):
    image = np.random.rand(224, 224, channels)
    image_data = np.asarray(image).astype(np.float32)
#    image_data = image_data.transpose([2, 0, 1])  # Transpose to CHW
    mean = np.array([0.079, 0.05, 0]) + 0.406
    std = np.array([0.005, 0, 0.001]) + 0.224
    for channel in range(image_data.shape[2]):
        image_data[:, :, channel] = (image_data[:, :, channel] / 255 - mean[channel]) / std[channel]
#    image_data = np.expand_dims(image_data, 0)
    return image_data


image = preprocess_image(channels=3)
print(image.shape)
height, width, channels = image.shape
resize_ratio = 416 / max(width, height)
new_width = int(width * resize_ratio)
new_height = int(height * resize_ratio)
resized_image = cv2.resize(image, (new_width, new_height))

inputs = []
for i in range(0, int(64 / batch_size)):
    blobs = []
    for batch in range(1, batch_size):
        blob = cv2.dnn.blobFromImage(resized_image, 1/255.0, (416, 416), swapRB=True, crop=False)
        blobs.append(blob)
    input_batch = np.concatenate(blobs, axis=0)
    inputs.append(input_batch)
    
print("Processed Images")

print("TimePreModelLoading --"+ str(time.time()))

# Load YOLO
net = cv2.dnn.readNet("/home/pace/execution/ML_workload_optimization/models/yolov4-tiny.weights", "/home/pace/execution/ML_workload_optimization/models/yolov4-tiny.cfg")

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

    for input in inputs:
        # Set the blob as input to the network
        net.setInput(input)
        # Perform forward pass
        layer_outputs = net.forward(output_layers)

print("TimePostModel --"+ str(time.time()))

