import json
import time
import sys
import subprocess
import threading
import os
from os import walk
# from tegrastats import Tegrastats
# from tegra_parse import VALS, MTS, RAM, SWAP, IRAM, CPUS, TEMPS, WATTS
file_name = ""

cpu_scaling_available_frequencies = [102000,204000,307200,403200,518400,614400,
										710400,825600,921600,1036800,1132800,
										1224000,1326000,1428000,1479000]

gpu_available_frequencies = [76800000,153600000,230400000,307200000,384000000,
										460800000,537600000,614400000,691200000,
										768000000,844800000,921600000]

# PATH_TEGRASTATS = ['/usr/bin/tegrastats', '/home/nvidia/tegrastats']
# path_tegrastats=PATH_TEGRASTATS
# i2c_folder = "/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/"


stop_logging = [False]

files = []
# csv = []
# csv_dict = {}
for (dirpath, dirnames, filenames) in walk("../archive/analysis"):
    files.extend(filenames)
    break

# for (dirpath, dirnames, filenames) in walk("./logs/csv"):
#     csv.extend(filenames)
#     break

# print(files)
logs = []

for file_name in files:
    # toggle_break
    #Get the threshold from the command line dumps
    threshold = 0
    if( not len(file_name.split(".")) < 3):
        continue
    f = open("../archive/analysis/"+file_name,"r")
    for idx,line in enumerate(f.readlines()):
        if len(line.split("--")) == 2 and line.split("--")[0] == "TimeLazyLoadSleep ":
            threshold = float(line.split("--")[1])
    f.close()
    #print("Threshold is " + str(threshold))
    file_name = "power_stats_" + file_name.split(".")[0]+".performance.log"
    print(file_name)
    f = open("../archive/analysis/"+file_name,"r")
    model_name = 0
    batch = 2

    f_model_stats = open("../archive/parsed_csv/"+ os.path.basename(f.name)+".model.csv","w")
    f_stats = open("../archive/parsed_csv/"+ os.path.basename(f.name)+".csv","w")
    print("Timestamp, Complete_Board Power, GPU Power, CPU Power, %GPU, %CPU, Model, Batch, Test Image Name, Iteration, CPU, Cpu_freq, Gpu_freq",file=f_stats)
    print("Timestamp, Complete_Board Power, GPU Power, CPU Power, %GPU, %CPU, Model, Batch, Test Image Name, Iteration, CPU, Cpu_freq, Gpu_freq",file=f_model_stats)
    prev_tpower_time = 0
    prev_power_time = 0
    cnt = 0


    for idx,line in enumerate(f.readlines()):
        cnt += 1
        vals = line.split(',')
        # print(vals)
        if cnt == 1:
            print(vals)

        # sys.exit(0)
        # The first line contains the modelName, batch, test_img_name, iter_no, cpu, cpu_freq, gpu_freq

        if idx == 0:
            # print(vals)
            model_name = vals[0]
            batch = vals[1]
            img_file = vals[2]
            iter_no = vals[3]
            cpu = vals[4]
            cpu_freq = int(vals[5])
            gpu_freq = int(vals[6])

            # if int(vals[6]) == 0:
            #     toggle_break == 1:
            #     break

            # if int(vals[6]) == 12:
            #     gpu_freq = gpu_available_frequencies[11]
            # else:    
            #     gpu_freq = gpu_available_frequencies[int(vals[6])]
        else:
            #Here we store the first total_power and idle power to calculate the  delta
            if(idx == 1):
                prev_tpower_time = int(vals[5])
                prev_power_time = int(vals[4])
                continue
            stats  = ""
            #Timestamp, Complete_Board Power(milliwatts), GPU Power(milliwatts), CPU Power(milliwatts), %GPU, %CPU, Model, Batch Size, Batch
            print("{}, {}, {}, {}, {}, {}, {} , {}, {}, {}, {}, {}, {}".format(
                float(vals[0]), 
                int(vals[1]), 
                int(vals[2]), 
                int(vals[3]),
                #https://forums.developer.nvidia.com/t/how-the-gpu-utilization-of-jetson-tx2-is-measured/78220
                float(int(vals[6])/10.0),
                # This to represent the time spent not idling
                100-(int(vals[4])-prev_power_time)/(int(vals[5]) - prev_tpower_time)*100,
                model_name,
                batch,
                img_file,
                iter_no,
                cpu,
                cpu_freq,
                gpu_freq),file= f_stats if float(vals[0]) > threshold else f_model_stats)
            prev_power_time = int(vals[4])
            prev_tpower_time = int(vals[5])
    f.close()
    f_stats.close()
    f_model_stats.close()