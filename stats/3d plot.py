import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import sys

fn = sys.argv[1]
dat = pd.read_csv("../archive/merged_csv/" + fn)
dat["Total_Power"] = dat["stats_Total_Avg_Power"]
# dat_d = pd.read_csv("../archive/merged_csv/resnet_dvfs_dle2_2023_06_16.csv")

b = 16
tmp = dat[dat.Batch_Size == b]
cpuf = dat.Cpu_Freq.unique()
cpuf = cpuf[~np.isnan(cpuf)]
gpuf = dat.Gpu_Freq.unique()
gpuf = gpuf[~np.isnan(gpuf)]

# cmap = 'Reds'
cmap = plt.cm.get_cmap('Reds')
cmap = cmap.reversed()

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
normalization_factor = 0.7  # Adjust this value to control the rate of normalization
Z_min = Z.min()
Z_max = Z.max()
Z_range = Z_max - Z_min

norm = plt.Normalize(Z_min, Z_min + Z_range * normalization_factor)


# Create a 3D figure and axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Create the surface plot
surf = ax.plot_trisurf(X, Y, Z, cmap = cmap, norm = norm)

# Add color bar
fig.colorbar(surf)

# Set labels and title
ax.set_xlabel('Gpu Frequency')
ax.set_ylabel('Cpu Frequency')
ax.set_zlabel('Total Energy')
ax.set_title('3D Surface Plot')

# Show the plot
plt.show()