import sys
import torch
from torch2trt import torch2trt
from torchvision.models.alexnet import alexnet
import time

class MatrixMultiplier(torch.nn.Module):
    def __init__(self):
        super(MatrixMultiplier, self).__init__()
   
    def forward(self, input1, input2):
        return torch.matmul(input1, input2)


multiplier = MatrixMultiplier()
#output = multiplier(torch.randn(3, 4), torch.randn(4, 5))

# create some regular pytorch model...
#model = torch.matmul
# create example data

m1 = (int(sys.argv[1]), int(sys.argv[1]))
m2 = (int(sys.argv[1]), int(sys.argv[1]))

x = torch.randn(m1).cuda()
y = torch.randn(m2).cuda()

# convert to TensorRT feeding sample data as input
model_trt = torch2trt(multiplier, [x, y])

model_trt(x, y)

torch.save(model_trt.state_dict(), 'matmul_trt.pth')


'''
print("TimePreModel --"+ str(time.time()))
time.sleep(1)
print("TimePreModelSleep --"+ str(time.time()))

model_trt(x, y)
print("TimeLazyLoad --"+ str(time.time()))
time.sleep(1)
print("TimeLazyLoadSleep --"+ str(time.time()))

for i in range(0, 24000):
    model_trt(x, y)
'''
