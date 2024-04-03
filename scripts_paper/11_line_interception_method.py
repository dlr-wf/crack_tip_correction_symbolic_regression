"""

    Example script:
        Crack Detection (path, tip, angle) for a single nodemap
        using the line intercept method and different correction methods

    Needed:
        - Nodemap

    Output:
        - Crack tip position
        - Crack path
        - Crack angle
        - Plot of predictions

"""

# Imports
import os
from crackpy.crack_detection.line_intercept import CrackDetectionLineIntercept, plot_grid_errors
from crackpy.fracture_analysis.data_processing import InputData
from crackpy.structure_elements.data_files import Nodemap
from crackpy.structure_elements.material import Material
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

# Set matplotlib settings
plt.rcParams.update({
    "font.size": 25,
    "text.usetex": True,
    "font.family": "Computer Modern",
    "figure.figsize": [10, 10],
    "figure.dpi": 300
})
# Set colormap
plt.rcParams['image.cmap'] = 'coolwarm'

# Settings
NODEMAP_FILE = ''  # ADD FILE HERE!
DATA_PATH = ''  # ADD PATH HERE!

# Output Path
OUTPUT_PATH = '11_Plot_LIM'
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)


# Get nodemap data
nodemap = Nodemap(name=NODEMAP_FILE, folder=DATA_PATH)
data = InputData(nodemap)
data.read_header()
material = Material(E=72000, nu_xy=0.33, sig_yield=350)
data.calc_stresses(material)


######################################
# Crack detection with line intercept
######################################
cd = CrackDetectionLineIntercept(
    x_min=25.0,
    x_max=55.0,
    y_min=-15.0,
    y_max=15.0,
    data=data,
    tick_size_x=0.1,
    tick_size_y=0.1,
    grid_component='uy',
    eps_vm_threshold=0.5/100,
    window_size=3,
    angle_estimation_mm_radius=5.0
)
cd.run()


######################################
# DIC data and detected crack path
######################################

fmin = 0
fmax = 0.5
num_colors = 120
contour_vector = np.linspace(fmin, fmax, num_colors, endpoint=True)
label_vector = np.linspace(fmin, fmax, 6, endpoint=True)

plt.clf()
fig = plt.figure(1)
ax = fig.add_subplot(111)
plot = ax.tricontourf(data.coor_x, data.coor_y, data.eps_vm * 100.0, contour_vector, extend='max')

ax.plot(cd.crack_path[:, 0], cd.crack_path[:, 1], 'k--', linewidth=2, label='Crack path $B(x)$')
ax.plot(cd.x_coords[cd.tip_index:], cd.y_path[cd.tip_index:], 'k--', linewidth=2)

indexes = [100, 250]
# plot paths
ax.plot(cd.x_grid[:, indexes[0]], cd.y_grid[:, indexes[0]], linewidth=2, color='blue', alpha=0.5)
ax.plot(cd.x_grid[:, indexes[1]], cd.y_grid[:, indexes[1]], linewidth=2, color='red', alpha=0.5)

# plot crack tip
ax.scatter(cd.crack_tip[0], cd.crack_tip[1], marker='x', color='black', s=300, linewidths=4, label='Crack tip (LIM)')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.2)
fig.colorbar(plot, ticks=label_vector, cax=cax, label='Von Mises eqv. strain [\\%]')
ax.set_xlabel('$x$ [mm]')
ax.set_ylabel('$y$ [mm]')
ax.legend(loc='upper left')
ax.axis('image')
ax.set_xlim(25, 55)
ax.set_ylim(-15, 15)
ax.tick_params(axis='x', pad=15)
plt.savefig(os.path.join(OUTPUT_PATH, "line_intercept_example.png"), bbox_inches='tight')

######################################
# tanh fit
######################################

plt.clf()
fig = plt.figure(1)
ax = fig.add_subplot(111)

ax.plot(cd.y_grid[:, indexes[0]], cd.disp_y_grid[:, indexes[0]], linewidth=8, color='blue', alpha=0.5)
ax.plot(cd.y_grid[:, indexes[0]], cd._tanh_funct(cd.coefficients_fitted[:, indexes[0]], cd.y_grid[:, indexes[0]]), "--", color="black", linewidth=2)
ax.plot(cd.y_grid[:, indexes[1]], cd.disp_y_grid[:, indexes[1]], linewidth=8, color='red', alpha=0.5)
ax.plot(cd.y_grid[:, indexes[1]], cd._tanh_funct(cd.coefficients_fitted[:, indexes[1]], cd.y_grid[:, indexes[1]]), "--", color="black", linewidth=2)

ax.set_xlabel('$y$ [mm]')
ax.set_ylabel('$u_{y}$ [mm]')
ax.tick_params(axis='x', pad=15)
plt.savefig(os.path.join(OUTPUT_PATH, "line_intercept_example_tanh.png"), bbox_inches='tight')
