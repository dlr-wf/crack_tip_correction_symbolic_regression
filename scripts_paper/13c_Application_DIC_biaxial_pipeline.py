"""

    Example script:
        Crack Detection path and tip for biax data with the line interception method

    Needed:
        - Nodemap

    Output:
        - Crack tip position
        - Crack path
        - Plot of prediction

"""
# Imports
import os

import cv2
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
OUTPUT_PATH = '13_Application/biaxial/DIC'
MAX_FORCE = 45.000  # kN
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
        f"{'Filename':>80},"
        f"{'CT x [mm]':>12},{'CT y [mm]':>12},{'Angle [°]':>12},"
        f"{'Corr dx [mm]':>15},{'Corr dy [mm]':>15},"
        f"{'Cycle [1]':>15}"
        f"\n"
    )

    # iterate over all nodemaps in folder in a sorted manner
    stages_to_filenames, _ = get_nodemaps_and_stage_nums(DATA_PATH)
    for stage in sorted(list(stages_to_filenames)):
        if stage < 200:
            continue
        file = stages_to_filenames[stage]
        if file.endswith(".txt"):
            # Get nodemap data
            nodemap = Nodemap(name=file, folder=DATA_PATH)

            meta_attributes_to_keywords = {
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
                'alignment_rotation_z': 'alignment_rotation_z'
            }

            data = InputData(nodemap, meta_keywords=meta_attributes_to_keywords)
            data.read_header(meta_attributes_to_keywords=meta_attributes_to_keywords)
            data.calc_stresses(material)
            if data.force is not None and data.force > MAX_FORCE - .5:

                # Run crack detection
                print(f"Crack detection for {file} ...")
                cd = CrackDetectionLineIntercept(
                    x_min=0.,
                    x_max=data.coor_x.max() - data.coor_x.max()*0.1,
                    y_min=-50.0,
                    y_max=50.0,
                    data=data,
                    tick_size_x=1.,
                    tick_size_y=1.,
                    grid_component='uy',
                    eps_vm_threshold=0.005,
                    window_size=10,
                    angle_estimation_mm_radius=30
                )
                cd.run()
                print("Crack angle: ", cd.crack_angle)

                # Fine-tune crack tip position
                opt_props = OptimizationProperties(
                    angle_gap=45,
                    min_radius=10,
                    max_radius=40,
                    tick_size=0.25,
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
                fmin= 0.0
                fmax = 0.68
                crack_tip_results = results

                if not os.path.exists(folder):
                    os.makedirs(folder)

                num_colors = 120
                contour_vector = np.linspace(fmin, fmax, num_colors, endpoint=True)
                label_vector = np.linspace(fmin, fmax, 6, endpoint=True)

                plt.clf()
                fig = plt.figure(1)
                ax = fig.add_subplot(111)
                plot = ax.tricontourf(cd.data.coor_x, cd.data.coor_y, cd.data.eps_vm*100.0, contour_vector,
                                      extend='max')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.2)
                plt.colorbar(plot, ticks=label_vector, cax=cax, label='Von Mises eqv. strain [\\%]')
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
                ax.set_xlim([0, 160.0])
                ax.set_ylim([-80, 80])

                plt.savefig(os.path.join(folder, fname), bbox_inches='tight')
                plt.close()

                # Write results to file
                out_file.write(
                    f'{file:>80},'
                    f'{cd.crack_tip[0]:>12.5f},{cd.crack_tip[1]:>12.5f},{cd.crack_angle:>12.5f},'
                    f'{crack_tip_corr_symreg[0]:>15.5f},{crack_tip_corr_symreg[1]:>15.5f},'
                    f'{data.cycles:>15}'
                    f'\n'
                )

# Create video
##############################################################################################
VIDEO_PATH = OUTPUT_PATH
PREFIX = ''
vid_name = '1223TSt1_DIC_biaxial_video'
folder = os.path.join(OUTPUT_PATH, 'plots')
for subdir, dirs, files in os.walk(folder):
    if len(files) != 0:
        print(f"Found {len(files)} files in {subdir}!")
        rel_path = os.path.relpath(subdir, folder)
        # take subdirectory as video name

        # get image's size
        img = cv2.imread(os.path.join(subdir, files[0]))
        height, width, layers = img.shape
        size = (width, height)

        # fetch images
        img_array = []
        for filename in natsorted(files):
            img = cv2.imread(os.path.join(subdir, filename))

            height, width, layers = img.shape
            current_size = (width, height)
            if current_size != size:
                raise ValueError(f"The images for the video {vid_name} have different sizes!")

            img_array.append(img)

        # make video
        # fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # for Windows (notebook + Windows workstations)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # for Linux (RedHat workstations)

        video = cv2.VideoWriter(os.path.join(VIDEO_PATH, PREFIX + vid_name + '.avi'), fourcc, fps=4,
                                frameSize=size)

        for img in img_array:
            video.write(img)
        video.release()
        print(f"Finished video {vid_name}")
