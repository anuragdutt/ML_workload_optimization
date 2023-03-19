from torch2trt_dynamic import torch2trt_dynamic
import torch
from torch import nn
from torchvision.models.alexnet import alexnet
import time

# create some regular pytorch model...
model = alexnet(pretrained=True).eval().cuda()

# create example data
x = torch.ones((1, 3, 512, 512)).cuda()

# convert to TensorRT feeding sample data as input
opt_shape_param = [
    [
        [1, 3, 512, 512],   # min
        [1, 3, 512, 512],   # opt
        [1, 3, 512, 512]    # max
    ]
]
model_trt = torch2trt_dynamic(model, [x], fp16_mode=False, opt_shape_param=opt_shape_param)



print("TimePreModel --"+ str(time.time()))
time.sleep(1)
print("TimePreModelSleep --"+ str(time.time()))

model_trt(x)
print("TimeLazyLoad --"+ str(time.time()))
time.sleep(1)
print("TimeLazyLoadSleep --"+ str(time.time()))

for i in range(0, 2000):
    model_trt(x)


