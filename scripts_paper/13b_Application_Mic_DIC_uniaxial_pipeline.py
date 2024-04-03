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
from crackpy.crack_detection.utils.utilityfunctions import get_nodemaps_and_stage_nums
from crackpy.fracture_analysis.data_processing import InputData
from crackpy.fracture_analysis.optimization import OptimizationProperties
from crackpy.structure_elements.data_files import Nodemap
from crackpy.structure_elements.material import Material
from mpl_toolkits.axes_grid1 import make_axes_locatable
from natsort import natsorted

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
OUTPUT_PATH = '13_Application/uniaxial/Mic_DIC/pipeline'
MAX_FORCE = 15000
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
        f"{'CT x [mm]':>12},{'ĆT y [mm]':>12},"
        f"{'Corr dx [mm]':>15},{'Corr dy [mm]':>15}"
        f"\n"
    )

    # iterate over all nodemaps in folder in a sorted manner
    stages_to_filenames, _ = get_nodemaps_and_stage_nums(DATA_PATH)
    for stage in natsorted(list(stages_to_filenames)):
        if stage < 48 or stage > 163:
            continue
        file = stages_to_filenames[stage]
        if file.endswith(".txt"):
            # Get nodemap data
            nodemap = Nodemap(name=file, folder=DATA_PATH)
            data = InputData(nodemap)
            data.calc_stresses(material)
            if data.force is not None and data.force > MAX_FORCE - 50:

                # Run crack detection
                print(f"Crack detection for {file} ...")
                cd = CrackDetectionLineIntercept(
                    x_min=34.0,
                    x_max=40.0,
                    y_min=-2.0,
                    y_max=2.0,
                    data=data,
                    tick_size_x=0.05,
                    tick_size_y=0.08,
                    grid_component='uy',
                    eps_vm_threshold=0.015,
                    window_size=3,
                    angle_estimation_mm_radius=1.0
                )
                cd.run()
                print("Crack angle: ", cd.crack_angle)

                # Fine-tune crack tip position
                opt_props = OptimizationProperties(
                    angle_gap=45,
                    min_radius=1.0,
                    max_radius=2.0,
                    tick_size=0.02,
                    terms=[-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
                )
                correction = CrackTipCorrection(data, cd.crack_tip, cd.crack_angle, material)
                crack_tip_corr_symreg = correction.correct_crack_tip(
                    opt_props,
                    max_iter=50,
                    step_tol=1e-3,
                    method='symbolic_regression',
                    verbose=True
                )

                # Plot prediction
                results = {
                    'Iterative correction': crack_tip_corr_symreg
                }
                fname = file[:-4] + '.png'
                folder = os.path.join(OUTPUT_PATH, 'plots')

                crack_tip_results=results

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

                ax.legend(loc='lower left')
                ax.set_xlabel('$x$ [mm]')
                ax.set_ylabel('$y$ [mm]')
                ax.axis('image')
                ax.tick_params(axis='x', pad=15)
                ax.set_xlim([33.8, 41.8])
                ax.set_ylim([-2.7, 2.8])

                plt.savefig(os.path.join(folder, fname), bbox_inches='tight')
                plt.close()

                # Write results to file
                out_file.write(
                    f'{file:>60},'
                    f'{cd.crack_tip[0]:>12.5f},{cd.crack_tip[1]:>12.5f},'
                    f'{crack_tip_corr_symreg[0]:>15.5f},{crack_tip_corr_symreg[1]:>15.5f}'
                    f'\n'
                )
