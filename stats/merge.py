import pandas as pd
import os
import matplotlib.pyplot as plt
import time
import sys
pd.set_option('display.float_format', lambda x: '%.5f' % x)

files = os.listdir("/home/gary/data/columbia/research/ML_workload_optimization/archive/parsed_csv")
task = "mobilenet"
run_parameters = "base"
run_date = "2023_05_04"
run_device = "dle1"

new_rows = []
new_rows_model = []

directory_path = "/home/gary/data/columbia/research/ML_workload_optimization/archive/parsed_csv"
timestamps = set()
for filename in os.listdir(directory_path):
	if filename.endswith(".stats.csv"):
		timestamp = filename.split(".")[0]
		timestamps.add(timestamp)

# print(timestamps)
# sys.exit(0)
for timestamp in timestamps:
	
	loading_file = os.path.join(directory_path, f"{timestamp}.performance.log.loading.csv")
	model_file = os.path.join(directory_path, f"{timestamp}.performance.log.model.csv")
	stats_file = os.path.join(directory_path, f"{timestamp}.performance.log.stats.csv")

	model_data = pd.read_csv(model_file)
	loading_data = pd.read_csv(loading_file)
	stats_data = pd.read_csv(stats_file)

	# print(model_data.shape)
	# print(loading_data.shape)
	# print(stats_data.shape)

	model_data.columns = [x.strip() for x in model_data.columns]
	loading_data.columns = [x.strip() for x in loading_data.columns]
	stats_data.columns = [x.strip() for x in stats_data.columns]

	# if loading_data.shape[0]==0:
	# 	row = {}
	# 	# new_rows_dist.append(row)
	# 	continue
	if model_data.shape[0]==0:
		row = {}
		# new_rows_dist.append(row)
		continue
	elif stats_data.shape[0]==0:
		row = {}
		# new_rows_dist.append(row)
		continue
	else: 
		row = {
					# "Model"				:model_data.Model[0].strip(),
					# "Iteration"			:model_data.Iteration[0],
					# "Num_CPU"			:model_data.CPU[0],
					# "CPU_Freq"			:model_data.Cpu_freq[0],
					# "GPU_Freq"			:model_data.Gpu_freq[0],
					# "loading_Time"		:(loading_data.Timestamp.max() - loading_data.Timestamp.min()),
					# "loading_Energy"		:(loading_data.Timestamp.diff()*loading_data["Complete_Board Power"]).sum(),
					# "loading_CPU_Energy"		:(loading_data.Timestamp.diff()*loading_data["CPU Power"]).sum(),
					# "loading_GPU_Energy"		:(loading_data.Timestamp.diff()*loading_data["GPU Power"]).sum(),
					# "loading_Total_Avg_Power"		:(loading_data["Complete_Board Power"]).median(),
					# "loading_CPU_Avg_Power"		:(loading_data["CPU Power"]).median(),
					# "loading_GPU_Avg_Power"		:(loading_data["GPU Power"]).median(),
					# "loading_CPU_percent__%"		:(loading_data["%CPU"].quantile([0.25,0.5,0.75,1]).round(1).to_list()),
					# "loading_GPU_percent__%"		:(loading_data["%GPU"].quantile([0.25,0.5,0.75,1]).round(1).to_list()),
					"inference_Time"		:(stats_data.Timestamp.max() - stats_data.Timestamp.min()),
					"stats_Energy"		:(stats_data.Timestamp.diff()*stats_data["Complete_Board Power"]).sum(),
					"stats_CPU_Energy"		:(stats_data.Timestamp.diff()*stats_data["CPU Power"]).sum(),
					"stats_GPU_Energy"		:(stats_data.Timestamp.diff()*stats_data["GPU Power"]).sum(),
					"stats_Total_Avg_Power"		:(stats_data["Complete_Board Power"]).median(),
					"stats_CPU_Avg_Power"		:(stats_data["CPU Power"]).median(),
					"stats_GPU_Avg_Power"		:(stats_data["GPU Power"]).median(),
					"stats_CPU_percent__%"		:(stats_data["%CPU"].quantile([0.25,0.5,0.75,1]).round(1).to_list()),
					"stats_GPU_percent__%"		:(stats_data["%GPU"].quantile([0.25,0.5,0.75,1]).round(1).to_list()),

		}
		new_rows.append(row)
		# print(row)

rows_df = pd.DataFrame(new_rows)
print(rows_df)
rows_df.to_csv("/home/gary/data/columbia/research/ML_workload_optimization/archive/merged_csv/"+ task +"_" + run_parameters + "_" + run_device + "_" + run_date +".csv", index = False)

# rows_df_model = pd.DataFrame(new_rows_model)
# print(rows_df_model)

# # out_df = rows_df.merge(rows_df, on=["Model","Batch_Size","Iteration","Num_CPU","Cpu_freq","Gpu_freq","Test_Image_Name"])

# # rows_df.to_csv("../archive/merged_csv/"+ task +"_" +time.strftime("%Y_%m_%d__%H_%M_%S")+".csv", index = False)

	#     # Read each file and do something with them here...
	#     # For example:
	#     with open(model_file) as f:
	#         model_data = f.read()
	#     with open(loading_file) as f:
	#         stats_data = f.read()
	#     with open(stats_file) as f:
	#         stats_data = f.read()
		

# for file in files:
# 	if "model" in file:
# 		prepend_text="model_"
# 		new_rows_dist = new_rows_model
# 	else:
# 		prepend_text=""
# 		new_rows_dist = new_rows

# 	df = pd.read_csv("/home/gary/data/columbia/research/compiler_optimization_on_edge/archive/parsed_csv/"+file)
# 	df.columns = [x.strip() for x in df.columns]
# 	if df.shape[0]==0:
# 		row = {}
# 		new_rows_dist.append(row)
# 		continue

# 	row = {
# 				"Model"				:df.Model[0].strip(),
# 				"Iteration"			:df.Iteration[0],
# 				"Num_CPU"			:df.CPU[0],
# 				"Cpu_freq"			:df.Cpu_freq[0],
# 				"Gpu_freq"			:df.Gpu_freq[0],
# 				prepend_text+"Time_Taken__Sec"		:(df.Timestamp.max()-df.Timestamp.min()	),
# 				prepend_text+"Total_Energy__mWattSec"		:(df.Timestamp.diff()*df["Complete_Board Power"]).sum(),
# 				prepend_text+"CPU_Energy__mWattSec"		:(df.Timestamp.diff()*df["CPU Power"]).sum(),
# 				prepend_text+"GPU_Energy__mWattSec"		:(df.Timestamp.diff()*df["GPU Power"]).sum(),
# 				prepend_text+"CPU_percent__%"		:(df["%CPU"].quantile([0.25,0.5,0.75,1]).round(1).to_list()),
# 				prepend_text+"GPU_percent__%"		:(df["%GPU"].quantile([0.25,0.5,0.75,1]).round(1).to_list()),
# 				prepend_text+"Total_Avg_Power__mWatt"		:(df["Complete_Board Power"]).median(),
# 				prepend_text+"CPU_Avg_Power__mWatt"		:(df["CPU Power"]).median(),
# 				prepend_text+"GPU_Avg_Power__mWatt"		:(df["GPU Power"]).median(),
# 		}
# 	new_rows_dist.append(row)
# 	# print(row)
