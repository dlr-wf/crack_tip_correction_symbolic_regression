"""

    Example script:
        Crack Detection path and tip with the line interception method

    Needed:
        - Nodemap

    Output:
        - Crack tip position
        - Crack path
        - Plot of prediction

"""
# Imports
import os

import matplotlib.pyplot as plt
import numpy as np

from crackpy.crack_detection.line_intercept import CrackDetectionLineIntercept
from crackpy.crack_detection.correction import CrackTipCorrection
from crackpy.fracture_analysis.data_processing import InputData
from crackpy.fracture_analysis.optimization import OptimizationProperties
from crackpy.structure_elements.data_files import Nodemap
from crackpy.structure_elements.material import Material
from mpl_toolkits.axes_grid1 import make_axes_locatable


# utility functions
def get_mid_nodemap(nm_path):
    nodemaps = os.listdir(nm_path)
    for file in nodemaps:
        if file.split("_")[-1] == "connections.txt":
            nodemaps.remove(file)
    x_indxs = []
    y_indxs = []
    for file in nodemaps:
        x_indxs.append(float(file.split("_")[7]))
        y_indxs.append(float(file.split("_")[8]))

    x_indx = str(int(np.asarray(x_indxs).mean()))
    y_indx = str(int(np.asarray(y_indxs).mean()))

    for file in nodemaps:
        if file.split("_")[7] == x_indx and file.split("_")[8] == y_indx:
            return file

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
DATA_PATH = ''  # ADD PATH HERE!
OUTPUT_PATH = '13_Application/biaxial/Mic_DIC/pipeline'
MAX_FORCE = 45.0
material = Material(E=72000, nu_xy=0.33, sig_yield=350)


# open empty file
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
with open(os.path.join(OUTPUT_PATH, "crack_correction_results.txt"), "w") as out_file:
    out_file.write(
        "####################################################################\n"
        "# Crack estimation with line intercept method (LIM)\n"
        "#\n"
        "# Symbolic regression correction method:\n"
        "# d_x = - a[-1] / a_[1] \n"
        "# d_y = - b[-1] / a[1]\n"
        "#\n"
        "####################################################################\n"
    )
    out_file.write(
        f"{'Filename':>60},"
        f"{'CT x [mm]':>15},{'ĆT y [mm]':>15},"
        f"{'Corr dx [mm]':>15},{'Corr dy [mm]':>15}"
        f"\n"
    )

    # get all discrete cycles where data is available
    cycles = 202149

    # get middle nodemap for each cycle at max load
    path_to_nodemaps = os.path.join(DATA_PATH, f"cycle_{cycles}", "max_load", "dic_results")

    meta_data_to_keywords = {
        'force': 'experimental_data_load_main_axis_fy',
        'experimental_data_load_main_axis_fy': 'experimental_data_load_main_axis_fy',
        'experimental_data_load_side_axis_fx': 'experimental_data_load_side_axis_fx',
        'cycles': 'experimental_data_cycles_catman',
        'experimental_data_cycles_catman': 'experimental_data_cycles_catman',
        'experimental_data_position_side_axis_dy': 'experimental_data_position_side_axis_dy',
        'TIMESTEP': 'TIMESTAMP',
        'alignment_translation_x': 'alignment_translation_x',
        'alignment_translation_y': 'alignment_translation_y',
        'alignment_translation_z': 'alignment_translation_z',
        'alignment_deviation': 'alignment_deviation',
        'alignment_rotation_x': 'alignment_rotation_x',
        'alignment_rotation_y': 'alignment_rotation_y',
        'alignment_rotation_z': 'alignment_rotation_z',
        'experimental_data_indx_x': 'experimental_data_indx_x',
        'experimental_data_indx_y': 'experimental_data_indx_y',
        'experimental_data_set_x': 'experimental_data_set_x',
        'experimental_data_set_y': 'experimental_data_set_y',
        'experimental_data_set_z': 'experimental_data_set_z',
        'experimental_data_x_real': 'experimental_data_x_real',
        'experimental_data_y_real': 'experimental_data_y_real',
        'experimental_data_z_real': 'experimental_data_z_real'
    }

    nodemap_file = get_mid_nodemap(path_to_nodemaps)
    # Get nodemap data
    nodemap = Nodemap(name=nodemap_file, folder=path_to_nodemaps)
    data = InputData(nodemap, meta_keywords=meta_data_to_keywords)
    data.read_header(meta_attributes_to_keywords=meta_data_to_keywords)
    data.calc_stresses(material)

    # Run crack detection
    print(f"Crack detection for {nodemap_file} ...")
    cd = CrackDetectionLineIntercept(
        x_min=data.coor_x.min() + 3.,
        x_max=data.coor_x.max() - 3.,
        y_min=data.coor_y.min() + 1.0,
        y_max=data.coor_y.max() - 1.0,
        data=data,
        tick_size_x=0.1,
        tick_size_y=0.1,
        grid_component='uy',
        eps_vm_threshold=0.015,
        window_size=3,
        angle_estimation_mm_radius=5.
    )
    cd.run()

    # Fine-tune crack tip position
    opt_props = OptimizationProperties(
        angle_gap=45,
        min_radius=2,
        max_radius=4,
        tick_size=0.05,
        terms=[-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
    )
    correction = CrackTipCorrection(data, cd.crack_tip, cd.crack_angle, material)
    crack_tip_corr_symreg = correction.correct_crack_tip(
        opt_props,
        max_iter=100,
        step_tol=1e-3,
        method='symbolic_regression',
        verbose=True
    )

    # Plot prediction
    # Set colormap
    plt.rcParams['image.cmap'] = 'coolwarm'
    plt.rcParams.update({'font.size': 25})
    plt.rcParams['figure.figsize'] = [10, 10]
    plt.rcParams['figure.dpi'] = 300

    results = {
        'Iterative Correction': crack_tip_corr_symreg
    }
    fname = f"{nodemap_file[:-4]}.png"
    folder = os.path.join(OUTPUT_PATH, 'plots')

    crack_tip_results = results

    if not os.path.exists(folder):
        os.makedirs(folder)

    # Matplotlib plot
    number_Colors = 120
    number_labes = 5
    legend_limit_max = 0.68
    Legend_limit_min = 0.0

    contour_vector = np.linspace(Legend_limit_min, legend_limit_max, number_Colors, endpoint=True)
    label_vector = np.linspace(Legend_limit_min, legend_limit_max, number_labes, endpoint=True)
    label_vector = np.round(label_vector, 2)

    plt.clf()
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    plot = ax.tricontourf(cd.data.coor_x, cd.data.coor_y, cd.data.eps_vm * 100.0, contour_vector,
                          extend='max')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=1)
    plt.colorbar(plot, ticks=label_vector, cax=cax, orientation="horizontal",
                 label='Von Mises eqv. strain [\\%]')
    ax.plot(cd.crack_path[:, 0], cd.crack_path[:, 1], 'k--', linewidth=2, label='Crack path $B(x)$')
    ax.scatter(cd.crack_tip[0], cd.crack_tip[1], s=300, color='black', marker='x',
               linewidth=3, label='Crack tip (LIM)')

    for method, ct_corr in crack_tip_results.items():
        ax.scatter(cd.crack_tip[0] + ct_corr[0], cd.crack_tip[1] + ct_corr[1], s=300, marker='x',
                   linewidth=3, color='green', label=method)

    ax.legend(loc='upper right')
    ax.set_xlabel('$x$ [mm]')
    ax.set_ylabel('$y$ [mm]')
    ax.axis('image')
    ax.tick_params(axis='x', pad=15)
    ax.set_xlim([data.coor_x.min(), data.coor_x.min() + 25.7])
    ax.set_ylim([data.coor_y.min(), data.coor_y.min() + 16.0])

    plt.savefig(os.path.join(folder, fname), bbox_inches='tight')

    # Write results to file
    out_file.write(
        f'{nodemap_file:>60},'
        f'{cd.crack_tip[0]:>15.5f},{cd.crack_tip[1]:>15.5f},'
        f'{crack_tip_corr_symreg[0]:>15.5f},{crack_tip_corr_symreg[1]:>15.5f}'
        f'\n')


