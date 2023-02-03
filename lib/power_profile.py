from concurrent.futures import thread
import json
import time
import sys
import subprocess
import threading
import os
sys.path.append("../")
from tegrastats import Tegrastats
from tegra_parse import VALS, MTS, RAM, SWAP, IRAM, CPUS, TEMPS, WATTS


def tegra_stats(tegrastats):
	# Make configuration dict
	# logger.debug("tegrastats read")
	data = {}
	print("stats")
	f = open("demofile2.txt", "a")
	f.append(tegrastats)
	f.append("\n")
	f.close()

#argv 2 3 5 6 7 8 represent a unique experiment
def _decode(text):
	# Find and parse all single values
	stats = VALS(text)
	# Parse if exist MTS
	mts = MTS(text)
	if mts:
		stats['MTS'] = mts
	# Parse RAM
	stats['RAM'] = RAM(text)
	# If exists parse SWAP
	swap = SWAP(text)
	if swap:
		stats['SWAP'] = swap
	# If exists parse IRAM
	iram = IRAM(text)
	if iram:
		stats['IRAM'] = iram
	# Parse CPU status
	stats['CPU'] = CPUS(text)
	# Parse temperatures
	stats['TEMP'] = TEMPS(text)
	# Parse Watts
	stats['WATT'] = WATTS(text)
	return stats

def logging(i2c_folder, cpu_usage_file, gpu_usage_file, f_stats, stop_logging):
	print("Starting logs")
	try:
		# print("Timestamp, Complete_Board Power, GPU Power, CPU Power, %GPU, %CPU1, %CPU2, %CPU3, %CPU4, Model, Batch Size, NVP Model",file=f_stats)
		print("{},{},{},{},{},{},{}".format(
			sys.argv[2],
			sys.argv[3],
			"all.jpg",
			sys.argv[5],
			sys.argv[6],
			sys.argv[7],
			sys.argv[8],),file=f_stats)
		while True:
			f_INP_power = open(i2c_folder+"in_power0_input","r")
			f_GPU_power = open(i2c_folder+"in_power1_input","r")
			f_CPU_power = open(i2c_folder+"in_power2_input","r")

			# Ref: https://supportcenter.checkpoint.com/supportcenter/portal?eventSubmit_doGoviewsolutiondetails=&solutionid=sk65143
			# cpu 79242 0 74306 842486413 756859 6140 67701 0
			# The /proc/stat is filled with data in the above said format
			#  The very first line "cpu" aggregates the numbers in all of the other "cpuN" lines.
			# These numbers identify the amount of time the CPU has spent performing different kinds of work.
			# Time units are in USER_HZ or Jiffies (typically hundredths of a second). 
			# 1st column : user = normal processes executing in user mode
			# 2nd column : nice = niced processes executing in user mode
			# 3rd column : system = processes executing in kernel mode
			# 4th column : idle = twiddling thumbs
			# 5th column : iowait = waiting for I/O to complete
			# 6th column : irq = servicing interrupts
			# 7th column : softirq = servicing softirqs
			cpu_usage = open(cpu_usage_file,"r")
			gpu_usage = open(gpu_usage_file,"r")
			stats = {}
			cpu_data = cpu_usage.readline().split()
			gpu_data = gpu_usage.readline()
			
			#Here we calculate total cpu time across different columns
			total_cpu = 0
			for i in range(1,len(cpu_data)):
				total_cpu = total_cpu + int(cpu_data[i])

			print("%s, %d, %d, %d, %d, %d, %d, %s" %(
				time.time(), 
				int(f_INP_power.read()), 
				int(f_GPU_power.read()), 
				int(f_CPU_power.read()),
				#This represents the time spent idling
				int(cpu_data[4]),
				int(total_cpu),
				int(gpu_data),
				json.dumps(tegra_val)),file=f_stats)
				
			time.sleep(0.100)
			if stop_logging[0]:
				f_INP_power.close()
				f_GPU_power.close()
				f_CPU_power.close() 
				cpu_usage.close()
				gpu_usage.close()
				break
	except KeyboardInterrupt:
		pass

	print("Stopping logs")


def tegra_logging(tegra_val):
	print("Starting saving tegrastats")
	try:
		while True:
			pts = subprocess.Popen(['/usr/bin/tegrastats'], stdout=subprocess.PIPE)

			out = pts.stdout
			if out is not None:
				# Read line process output
				line = out.readline()
				# Decode line in UTF-8
				tegrastats_data = line.decode("utf-8")
				# Decode and store
				stats = _decode(tegrastats_data)
				#print(stats)
				# global tegra_val 
				tegra_val = stats		
			pts.terminate()
			if stop_logging[0]:
				break
		#print("\n",file=f_stats)
	except KeyboardInterrupt:
		pass

	print("Stopping logs")


if __name__ == "__main__":


	stats_nick = sys.argv[1]
	path_tegrastats = ['/usr/bin/tegrastats', '/home/nvidia/tegrastats']
	i2c_folder = "/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/"
	f_stats = open("logs/power_stats_"+stats_nick+".performance.log","w")
	cpu_usage_file = "/proc/stat"
	gpu_usage_file = "/sys/devices/gpu.0/load"
	stop_logging = [False]
	tegra_val = None 

	#sys.argv[1] represents name of the log file
	#sys.argv[2] represents model name
	#sys.argv[3] represents batch
	#sys.argv[4] represents file name
	#sys.argv[5] represents iterration number
	#sys.argv[6] represents cpu
	#sys.argv[7] represents cpu_freq
	#sys.argv[8] represents gpu_freq


# subprocess.Popen("sudo /usr/sbin/nvpmodel -m {}".format(sys.argv[3]),stdin=subprocess.PIPE,shell=True)

	subprocess.Popen("rm -rf /tmp/train_logs/",stdin=subprocess.PIPE,shell=True)

	# if "matmul" not in sys.argv[4]:
	command = """
	python3 lib/{} {}
	""".format(sys.argv[4],sys.argv[3])
	# else:
	# 	command = """ {}""".format(sys.argv[3])


	print(command)
# time.sleep(1)
# sys.exit(0)



	tegra_thread = threading.Thread(target=tegra_logging, args = [tegra_val]) #polls at 1 sec
	t = threading.Thread(target=logging, args=[i2c_folder, cpu_usage_file, gpu_usage_file, f_stats, stop_logging]) #polls at 1/10 sec
	tegra_thread.start()
	t.start()
	process = subprocess.Popen(command,stdin=subprocess.PIPE,shell=True)

	process.communicate()
	stop_logging[0] = True
	t.join()
	tegra_thread.join()
	# tegra.close()
	f_stats.close()


