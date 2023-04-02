import pandas as pd
import os
import matplotlib.pyplot as plt
import time
pd.set_option('display.float_format', lambda x: '%.5f' % x)

files = os.listdir("../archive/parsed_csv")
task = "matmul_tensorrt"

new_rows = []
new_rows_model = []

for file in files:
	if "model" in file:
		prepend_text="model_"
		new_rows_dist = new_rows_model
	else:
		prepend_text=""
		new_rows_dist = new_rows

	df = pd.read_csv("../archive/parsed_csv/"+file)
	df.columns = [x.strip() for x in df.columns]
	if df.shape[0]==0:
		row = {}
		new_rows_dist.append(row)
		continue

	row = {
				"Model"				:df.Model[0].strip(),
				"Batch_Size"		:df.Batch[0],
				"Iteration"			:df.Iteration[0],
				"Num_CPU"			:df.CPU[0],
				"Cpu_freq"			:df.Cpu_freq[0],
				"Gpu_freq"			:df.Gpu_freq[0],
				"Test_Image_Name"	:df['Test Image Name'][0],

				prepend_text+"Time_Taken__Sec"		:(df.Timestamp.max()-df.Timestamp.min()	),
				prepend_text+"Total_Energy__mWattSec"		:(df.Timestamp.diff()*df["Complete_Board Power"]).sum(),
				prepend_text+"CPU_Energy__mWattSec"		:(df.Timestamp.diff()*df["CPU Power"]).sum(),
				prepend_text+"GPU_Energy__mWattSec"		:(df.Timestamp.diff()*df["GPU Power"]).sum(),
				prepend_text+"CPU_percent__%"		:(df["%CPU"].quantile([0.25,0.5,0.75,1]).round(1).to_list()),
				prepend_text+"GPU_percent__%"		:(df["%GPU"].quantile([0.25,0.5,0.75,1]).round(1).to_list()),
				prepend_text+"Total_Avg_Power__mWatt"		:(df["Complete_Board Power"]).median(),
				prepend_text+"CPU_Avg_Power__mWatt"		:(df["CPU Power"]).median(),
				prepend_text+"GPU_Avg_Power__mWatt"		:(df["GPU Power"]).median(),
		}
	new_rows_dist.append(row)
	# print(row)

rows_df = pd.DataFrame(new_rows)
print(rows_df)

rows_df_model = pd.DataFrame(new_rows_model)
print(rows_df_model)

# out_df = rows_df.merge(rows_df, on=["Model","Batch_Size","Iteration","Num_CPU","Cpu_freq","Gpu_freq","Test_Image_Name"])

rows_df.to_csv("../archive/merged_csv/"+ task +"_" +time.strftime("%Y_%m_%d__%H_%M_%S")+".csv", index = False)