"""
    This script plots the evaluation grid to determine the Williams expansion coefficients based on the FEA results.

    Needed:
        - Folder containing the nodemap files from the FEA

    Output:
        - folder 00_Plots containing the plots
"""


# Imports
import os
from crackpy.fracture_analysis.data_processing import InputData
from crackpy.structure_elements.data_files import Nodemap
from crackpy.structure_elements.material import Material
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Set matplotlib settings
plt.rcParams.update({
    "font.size": 25,
    "text.usetex": True,
    "font.family": "Computer Modern",
    "figure.figsize": [10, 10],
    "figure.dpi": 300
})

# Settings
NODEMAP_FILE = 'n10.00_20.00_0.00_nodemap.txt'
NODEMAP_PATH = '../01_Simulation_Output'

# Samples
SAMPLES_FILE = 'n10.00_20.00_0.00_nodemap_samples.csv'
SAMPLES_PATH = '../04_CrackPy_random_evaluation_pipeline/samples'
OUTPUT_PATH = '00_Plots_samples'

# check if output path exists
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# Get nodemap data
nodemap = Nodemap(name=NODEMAP_FILE, folder=NODEMAP_PATH)
material = Material(E=72000, nu_xy=0.33, sig_yield=350)
data = InputData(nodemap)
data.calc_stresses(material)
data.calc_eps_vm()

# Matplotlib plot
number_Colors = 120
number_labes = 5
legend_limit_max = 300
Legend_limit_min = 0.0
cm = 'coolwarm'

contour_vector = np.linspace(Legend_limit_min, legend_limit_max, number_Colors, endpoint=True)
label_vector = np.linspace(Legend_limit_min, legend_limit_max, number_labes, endpoint=True)
label_vector = np.round(label_vector, 2)

x_min = -3
x_max = 3
y_min = -3
y_max = 3

# Load points from samples file
samples = np.genfromtxt(os.path.join(SAMPLES_PATH, SAMPLES_FILE), delimiter=',', skip_header=1)
x_grid = samples[:, 0]
y_grid = samples[:, 1]

plt.clf()
fig = plt.figure(1)
ax = fig.add_subplot(111)

ax.plot([-3.5, 0], [0, 0], color='black', linewidth=3, linestyle='dashed')
plot = ax.tricontourf(data.coor_x, data.coor_y, data.sig_vm,
                      contour_vector,
                      extend='max', cmap=cm)
#ax.scatter(x_grid, y_grid, s=10, c='k')
ax.scatter(0, 0, s=500, c='black', marker='x', linewidths=5)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.2)
plt.colorbar(plot, ticks=label_vector,
             cax=cax,
             label='$\\sigma_{\\rm VM}$ [$MPa$]')
ax.set_xlabel('$x$ [$mm$]')
ax.set_ylabel('$y$ [$mm$]')
ax.axis('image')

ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-3.5, 3.5)
ax.set_xticks([-3, 0, 3])
ax.set_yticks([-3, 0, 3])

output_file = os.path.join(OUTPUT_PATH, f'03_{NODEMAP_FILE[:-4]}_FEA_random_samples.png')
plt.savefig(output_file, bbox_inches='tight', dpi=300)
plt.clf()
