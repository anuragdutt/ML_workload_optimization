from torch2trt import TRTModule
import torch
import sys
import time

model_trt = TRTModule()

model_trt.load_state_dict(torch.load('matmul_trt.pth'))

m1 = (int(sys.argv[1]), int(sys.argv[1]))
m2 = (int(sys.argv[1]), int(sys.argv[1]))

x = torch.randn(m1).cuda()
y = torch.randn(m2).cuda()


print("TimePreModel --"+ str(time.time()))
time.sleep(1)
print("TimePreModelSleep --"+ str(time.time()))

model_trt(x, y)
print("TimeLazyLoad --"+ str(time.time()))
time.sleep(1)
print("TimeLazyLoadSleep --"+ str(time.time()))

for i in range(0, 24000):
    model_trt(x, y)

