import subprocess
import time
from os import walk
import sys
import argparse
sys.path.append("../lib")


# cpu_scaling_available_frequencies = [102000,204000,307200,403200,518400,614400,
# 										710400,825600,921600,1036800,1132800,
# 										1224000,1326000,1428000,1479000]

# gpu_available_frequencies = [76800000,153600000,230400000,307200000,384000000,
# 										460800000,537600000,614400000,691200000,
# 										768000000,844800000,921600000]


def str_split(st, delim = ','):
	return st.split(delim)


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--inference", help="inference file (matmul, custom model, pretrained_model)",
						action="store_true", default = "batch_inference.py")
	parser.add_argument("-c", "--cpu_freq", help="cpu frequencies (102000,204000,307200,403200,518400,614400,710400,825600,921600,1036800,1132800,1224000,1326000,1428000,1479000)",
						action="store_true", default = "307200,710400,1132800,1479000")
	parser.add_argument("-g", "--gpu_freq", help="gpu frequencies (matmul, custom model, pretrained_model)",
						action="store_true", default = "230400000,460800000,691200000,921600000")
	parser.add_argument("-co", "--core", help="number of cores (1,2,3,4)", 
						action="store_true", default = 4)
	parser.add_argument("-e", "--experiment", help="number of experiments per configuration", \
						action="store_true", default = 5)
	parser.add_argument("-m", "--model_name", help="name of the model to be printed", \
						action="store_true", default = "alexnet")
	parser.add_argument("-b", "--batch_size", help="batch_size/matrix size in case of matmul", \
						action="store_true", default = "8,16,32,64")
	


	args = parser.parse_args()
	execution_file = args.inference
	cpu_freqs = str_split(args.cpu_freq)
	gpu_freqs = str_split(args.gpu_freq)
	cores = args.core
	experiments = args.experiment
	models = str_split(args.model_name)

	if models == "matmul":
		bat = 1
	else:
		batl = str_split(args.batch_size)
		bat = [int(b) for b in batl]



	# subprocess.Popen("echo -500 > /proc/`cat /var/run/sshd.pid`/oom_score_adj",stdin=subprocess.PIPE,shell=True)
# models_imagenet = [  "alexnet"]
# #models_imagenet = [ "alexnet"]
# models_detectnet = [ "ssd-mobilenet-v1", "ssd-mobilenet-v2"]

# imgs = next(walk("./images"), (None, None, []))[2]  # [] if no file

	command_logstart  = "script logs/{} -c \"sudo python3 lib/power_profile.py {} {} {} {} {} {} {} {} \""
	command_readplogs = "sudo python3 lib/nano_nvpmplus.py --cpus {} --cpu_max_fq {} --gpu_max_fq {}"



	for fl in [execution_file]:   # "batch_inference.py" "onnx_inference.py"
		for model in models:
			for iter in range(0, experiments):
				for cpu in [cores]:
					for cpu_max_fq in cpu_freqs:
						for gpu_max_fq in gpu_freqs:
							for batch in bat: #[8,16,32,64]
								# process3 = subprocess.Popen(command_readplogs.format(cpu, cpu_max_fq, gpu_max_fq),stdin=subprocess.PIPE,shell=True)
								low_pow_file = model + "_" + fl.split('.')[0]+"_"+str(iter)+"_"+time.strftime("%Y_%m_%d__%H_%M_%S")
								process1 = subprocess.Popen(command_logstart.format(low_pow_file + ".txt",low_pow_file, model, batch, fl, iter, cpu, cpu_max_fq, gpu_max_fq),stdin=subprocess.PIPE,shell=True)
								sys.exit(0)
								# PRAMODH: the following line is not working!
								# subprocess.Popen("echo 600 > /proc/`pidof script`/oom_score_adj",stdin=subprocess.PIPE,shell=True)
								# print("Waiting for process 1 ")
								process1.communicate()
								print("Process 1 done ")
								process3.terminate()


	subprocess.Popen("echo 600 > /proc/`pidof script`/oom_score_adj",stdin=subprocess.PIPE,shell=True)
	process1.communicate()
