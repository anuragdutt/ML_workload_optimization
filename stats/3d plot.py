import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

dat = pd.read_csv("../archive/merged_csv/resnet_base_dle3_2023_06_16.csv")
dat["Total_Power"] = dat["stats_Total_Avg_Power"]
dat_d = pd.read_csv("../archive/merged_csv/resnet_dvfs_dle2_2023_06_16.csv")

b = 16
tmp = dat[dat.Batch_Size == b]
cpuf = dat.Cpu_Freq.unique()
cpuf = cpuf[~np.isnan(cpuf)]
gpuf = dat.Gpu_Freq.unique()
gpuf = gpuf[~np.isnan(gpuf)]

tmppl = []
for c in cpuf:
    for g in gpuf:
        t = tmp[(tmp.Cpu_Freq == c) & (tmp.Gpu_Freq == g)]
        ener = np.average(t.stats_Energy)
        tmppl.append([c, g, ener])

tmpp = pd.DataFrame(tmppl, columns = ["Cpu_Freq", "Gpu_Freq", "stats_Energy"])
#     print(tmpp)
X = tmpp.Gpu_Freq
Y = tmpp.Cpu_Freq
Z = tmpp.stats_Energy
# Create a 3D figure and axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Create the surface plot
surf = ax.plot_trisurf(X, Y, Z, cmap='viridis')

# Add color bar
fig.colorbar(surf)

# Set labels and title
ax.set_xlabel('Gpu Frequency')
ax.set_ylabel('Cpu Frequency')
ax.set_zlabel('Total Energy')
ax.set_title('3D Surface Plot')

# Show the plot
plt.show()