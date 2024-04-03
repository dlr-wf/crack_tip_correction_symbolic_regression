import itertools
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from natsort import natsorted

from crackpy.fracture_analysis.data_processing import InputData, apply_mask
from crackpy.fracture_analysis.optimization import OptimizationProperties
from crackpy.structure_elements.data_files import Nodemap
from crackpy.structure_elements.material import Material
from crackpy.crack_detection.correction import CustomCorrection

from cracktipcorr.equations import latex_to_sympy

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
DATA_PATH = "01_Simulation_Output"
NODEMAP_FILES = {
    "mode_I": [
        '0.00_10.00_0.00_nodemap.txt',
        '10.00_10.00_0.00_nodemap.txt'
    ],
    "mode_II": [
        '0.00_0.00_10.00_nodemap.txt',
        '10.00_0.00_10.00_nodemap.txt'
    ],
    "mixed_mode": [
        '0.00_10.00_10.00_nodemap.txt',
        '10.00_10.00_10.00_nodemap.txt'
    ]
}
FORMULAS = {
    "mode_I": {"dx": [0, 1, 12], "dy": [0, 1, 6]},
    "mode_II": {"dx": [0, 4, 10], "dy": [0, 1, 2, 9]},
    "mixed_mode": {"dx": [2, 4, 11], "dy": [2, 4, 8]}
}
# Correction formulas from Symbolic Regression
JSON_FILE = 'pareto_front.json'
JSON_DIR = '06_Pareto_Plots'

OUTPUT_PATH = '08_Convergence_study_FEA'

# Initialize crack randomly
np.random.seed(42)
crack_tip = np.random.uniform(low=[-2, -2], high=[2, 2])
crack_angle = 0
print(f"Initial crack tip position: {crack_tip}")

# Material properties
material = Material(E=72000, nu_xy=0.33, sig_yield=350)

# read json file
with open(os.path.join(JSON_DIR, JSON_FILE)) as json_file:
    symreg_data = json.load(json_file)


for mode_f, formulas in FORMULAS.items():
    for dx_num, dy_num in itertools.product(formulas["dx"], formulas["dy"]):
        dx_latex = symreg_data[f"log_{mode_f}_dx"][str(dx_num)]['latex']
        dy_latex = symreg_data[f"log_{mode_f}_dy"][str(dy_num)]['latex']
        print(f'Formulas ({mode_f}, {dx_num}, {dy_num}): dx = {dx_latex}, dy = {dy_latex}')

        # make path for formulas
        formula_dir = os.path.join(OUTPUT_PATH, mode_f, f'dx_{dx_num}_dy_{dy_num}')
        if not os.path.exists(formula_dir):
            os.makedirs(formula_dir)

        # Convert formulas to sympy functions
        _, dx_lambdified = latex_to_sympy(dx_latex)
        _, dy_lambdified = latex_to_sympy(dy_latex)

        for mode_nodemap, nodemap_files in NODEMAP_FILES.items():
            for file in nodemap_files:
                FILE_OUTPUT_PATH = os.path.join(formula_dir, mode_nodemap + "_" + file[:-4])
                if not os.path.exists(FILE_OUTPUT_PATH):
                    os.makedirs(FILE_OUTPUT_PATH)
                print(f"Processing {file}...")
                with open(os.path.join(FILE_OUTPUT_PATH, "crack_tip_correction.txt"), "w") as out_file:
                    out_file.write(
                        f"{'Filename':>60},"
                        f"{'CT x [mm]':>15},{'ĆT y [mm]':>15},"
                        f"{'Corr x [mm]':>15},{'Corr y [mm]':>15},"
                        f"\n"
                    )

                    # Get nodemap data
                    nodemap = Nodemap(name=file, folder=DATA_PATH)
                    data = InputData(nodemap)
                    data.calc_stresses(material)
                    # Remove unnecessary data to speed up evaluation
                    # The mask must be at least as large as the evaluation area
                    mask = np.all([data.coor_x < 15,
                                   data.coor_x > -15,
                                   data.coor_y < 15,
                                   data.coor_y > -15], axis=0)
                    data = apply_mask(data, mask)

                    # Fine-tune crack tip position
                    opt_props = OptimizationProperties(
                        # --> see Rethore paper for details
                        angle_gap=45,
                        min_radius=5,
                        max_radius=10,
                        tick_size=0.25,
                        terms=[-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
                    )
                    correction = CustomCorrection(data, crack_tip, crack_angle, material)
                    crack_tip_corr = correction.custom_correct_crack_tip(
                        opt_props,
                        dx_lambdified=dx_lambdified,
                        dy_lambdified=dy_lambdified,
                        max_iter=50,
                        step_tol=5e-3,
                        verbose=True,
                        damper=1
                    )
                    correction.iteration_log.to_csv(os.path.join(FILE_OUTPUT_PATH, f"{file[:-4]}_iteration_log.csv"),
                                                    index=False)

                    ##############################################################################################
                    # Line plot
                    ##############################################################################################

                    plt.clf()
                    fig = plt.figure(1)
                    ax = fig.add_subplot(111)
                    ax.plot(correction.iteration_log['Iteration'],
                            correction.iteration_log['dx'], linewidth=3, color='red', label='$dx$')
                    ax.plot(correction.iteration_log['Iteration'],
                            correction.iteration_log['dy'], linewidth=3, color='blue', label='$dy$')

                    # second y axis
                    ax2 = ax.twinx()
                    ax2.plot(correction.iteration_log['Iteration'], correction.iteration_log['a_-1'],
                             linewidth=3, color='salmon', linestyle='dashed', label='$A_{\mathrm{-1}}$')
                    ax2.plot(correction.iteration_log['Iteration'], correction.iteration_log['b_-1'],
                             linewidth=3, color='royalblue', linestyle='dashed', label='$B_{\mathrm{-1}}$')

                    # Set ticks on x-axis to integers
                    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

                    ax.set_xlabel('Iteration [1]')
                    ax.set_ylabel('$dx$, $dy$ [mm]')
                    ax2.set_ylabel('$A_{\mathrm{-1}}$, $B_{\mathrm{-1}}$ [$\mathrm{MPa \cdot mm^{3/2}}$]')

                    ax.legend(loc='upper center')
                    ax2.legend(loc='upper right')

                    plt.savefig(os.path.join(FILE_OUTPUT_PATH, file[:-4] + '_convergence.png'), bbox_inches='tight')

                    ##############################################################################################
                    # Plot crack detection results
                    ##############################################################################################
                    plt.rcParams['image.cmap'] = 'coolwarm'
                    plt.rcParams.update({'font.size': 25})
                    plt.rcParams['figure.figsize'] = [10, 10]
                    plt.rcParams['figure.dpi'] = 300

                    crack_tip_results = {
                        'Final': crack_tip_corr
                    }

                    for i in range(len(correction.iteration_log['Iteration']) + 1):
                        fmin = 0.0
                        fmax = 0.3

                        num_colors = 120
                        contour_vector = np.linspace(fmin, fmax, num_colors, endpoint=True)
                        label_vector = np.linspace(fmin, fmax, 6, endpoint=True)

                        plt.clf()
                        fig = plt.figure(1)
                        ax = fig.add_subplot(111)
                        plot = ax.tricontourf(data.coor_x, data.coor_y, data.eps_vm * 100, contour_vector, extend='max')
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("bottom", size="5%", pad=1)
                        plt.colorbar(plot, ticks=label_vector, cax=cax, orientation="horizontal",
                                     label='Von Mises eqv. strain [\\%]')
                        ax.scatter(crack_tip[0], crack_tip[1], s=300, color='black', marker='x',
                                   linewidth=5, label='Initial')

                        # plot line from x=-3 to x=0 at y=0
                        ax.plot([-3, 0], [0, 0], color='black', linewidth=2, linestyle='dashed')

                        for method, ct_corr in crack_tip_results.items():
                            ax.scatter(crack_tip[0] + ct_corr[0], crack_tip[1] + ct_corr[1], s=300, marker='x',
                                       linewidth=5, color='green', zorder=99, label=method)
                        ax.scatter(correction.iteration_log['crack_tip_x'][0:int(i)],
                                   correction.iteration_log['crack_tip_y'][0:int(i)], s=200, zorder=98, marker='x',
                                   color='tan', linewidth=3, label='Iterations')

                        ax.legend(loc='lower left')
                        ax.set_xlabel('$x$ [mm]')
                        ax.set_ylabel('$y$ [mm]')
                        ax.axis('image')
                        ax.tick_params(axis='x', pad=15)
                        #ax.set_xticks([38.0, 38.5, 39.0, 39.5])

                        ax.set_xlim([-3, 3])
                        ax.set_ylim([-3, 3])

                        folder = os.path.join(FILE_OUTPUT_PATH, 'plots')
                        if not os.path.exists(folder):
                            os.makedirs(folder)
                        plt.savefig(os.path.join(folder, file[:-4] + f'_iteration_{int(i)}.png'),
                                    bbox_inches='tight')

                    # Write results to file
                    out_file.write(
                        f'{file:>60},'
                        f'{crack_tip[0]:>15.5f},{crack_tip[1]:>15.5f},'
                        f'{crack_tip_corr[0]:>15.5f},{crack_tip_corr[1]:>15.5f}'
                        f'\n'
                    )

                ##############################################################################################
                VIDEO_PATH = FILE_OUTPUT_PATH
                PREFIX = ''
                vid_name = file + "_convergence_study"
                folder = os.path.join(FILE_OUTPUT_PATH, 'plots')
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

                        video = cv2.VideoWriter(os.path.join(VIDEO_PATH, PREFIX + vid_name + '.avi'), fourcc, fps=1,
                                                frameSize=size)

                        for img in img_array:
                            video.write(img)
                        video.release()
                        print(f"Finished video {vid_name}")