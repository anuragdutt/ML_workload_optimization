import argparse
import os
import subprocess



subprocess.Popen("nvpmodel -m 0",stdin=subprocess.PIPE,shell=True)

def set_state(cpus,cpu_max_fq,gpu_max_fq):
	if os.getuid() != 0:
		raise Exception("This program is not run as sudo or elevated this it will not work")
		
	for i in range(4):
		filename = "/sys/devices/system/cpu/cpu{}/online".format(i)
		state = int(cpus>i)
		print(filename,state)
		with open(filename,"w") as f:
			f.write(str(state))

	filename = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq".format(i)
	state = cpu_max_fq
	print(filename,state)
	with open(filename,"w") as f:
		f.write(str(state))

	filename = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq".format(i)
	state = cpu_max_fq
	print(filename,state)
	with open(filename,"w") as f:
		f.write(str(state))

	filename = "/sys/devices/gpu.0/devfreq/57000000.gpu/min_freq".format(i)
	state = gpu_max_fq
	print(filename,state)
	with open(filename,"w") as f:
		f.write(str(state))

	filename = "/sys/devices/gpu.0/devfreq/57000000.gpu/max_freq".format(i)
	state = gpu_max_fq
	print(filename,state)
	with open(filename,"w") as f:
		f.write(str(state))


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='nvpmodel plus')
	parser.add_argument('--cpus', 		action='store', type=int, default= 4, help="input can be 1 to 4" 	)
	parser.add_argument('--cpu_max_fq',	action='store', type=int, default=-1, help="input can be 0 to 14"	)
	parser.add_argument('--gpu_max_fq',	action='store', type=int, default=-1, help="input can be 0 to 11"	)

	args = parser.parse_args()
	print(args)
	set_state(args.cpus,args.cpu_max_fq,args.gpu_max_fq)
