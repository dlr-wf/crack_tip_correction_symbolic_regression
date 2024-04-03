"""
    This script plots the evaluation grid to determine the Williams expansion coefficients based on the FEA results.

    Needed:
        - Folder containing the nodemap files from the FEA

    Output:
        - folder 00_Plots containing the plots
"""
import itertools
import os
import matplotlib.pyplot as plt
from crackpy.fracture_analysis.data_processing import InputData, apply_mask
from crackpy.structure_elements.data_files import Nodemap
from crackpy.structure_elements.material import Material
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import numpy as np
import json
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
DATA_PATH = '01_Simulation_Output'
SAMPLES_DIR = '04_CrackPy_random_evaluation_pipeline/samples'
SAMPLE_FILES = {
    "mode_I": [
        '0.00_10.00_0.00_nodemap_samples.csv',
        '10.00_10.00_0.00_nodemap_samples.csv'
    ],
    "mode_II": [
        '0.00_0.00_10.00_nodemap_samples.csv',
        '10.00_0.00_10.00_nodemap_samples.csv'
    ],
    "mixed_mode": [
        '0.00_10.00_10.00_nodemap_samples.csv',
        '10.00_10.00_10.00_nodemap_samples.csv'
    ]
}
PLOT_DIR = '07_Plots_vector_fields'

# Correction formulas from Symbolic Regression
JSON_FILE = 'pareto_front.json'
JSON_DIR = '06_Pareto_Plots'

PLOT_BACKGROUND = True
N_SAMPLES = 100
RANDOM_STATE = 42

if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

# Material properties
material = Material(E=72000, nu_xy=0.33, sig_yield=350)

# read json file
with open(os.path.join(JSON_DIR, JSON_FILE)) as json_file:
    symreg_data = json.load(json_file)

FORMULAS = {
    "mode_I": {"dx": [0, 1, 12], "dy": [0, 1, 6]},
    "mode_II": {"dx": [0, 4, 10], "dy": [0, 1, 2, 9]},
    "mixed_mode": {"dx": [2, 4, 11], "dy": [2, 4, 8]}
}

for mode_f, formulas in FORMULAS.items():
    for dx_num, dy_num in itertools.product(formulas["dx"], formulas["dy"]):
        dx_latex = symreg_data[f"log_{mode_f}_dx"][str(dx_num)]['latex']
        dy_latex = symreg_data[f"log_{mode_f}_dy"][str(dy_num)]['latex']
        print(f'Formulas ({mode_f}, {dx_num}, {dy_num}): dx = {dx_latex}, dy = {dy_latex}')

        # make path for formulas
        formula_dir = os.path.join(PLOT_DIR, mode_f, f'dx_{dx_num}_dy_{dy_num}')
        if not os.path.exists(formula_dir):
            os.makedirs(formula_dir)

        dx_sympy_formula, dx_lambdify_function = latex_to_sympy(dx_latex)
        dy_sympy_formula, dy_lambdify_function = latex_to_sympy(dy_latex)
        print(f"dx_sympy_formula: {dx_sympy_formula} || {dx_latex}")
        print(f"dy_sympy_formula: {dy_sympy_formula} || {dy_latex}")

        # iterate over sample files
        for mode, files_list in SAMPLE_FILES.items():
            for file in files_list:
                print(f'Plotting {mode, file}')
                try:
                    df = pd.read_csv(os.path.join(SAMPLES_DIR, file))

                    # randomly choose lines of df to not overload the plot
                    df = df.sample(n=N_SAMPLES, random_state=RANDOM_STATE)
                    df = df.reset_index(drop=True)

                    # Get the stresses from the filename
                    stresses = file.replace('n', '-').split('_')
                    sigma_xx = float(stresses[0])
                    sigma_yy = float(stresses[1])
                    sigma_xy = float(stresses[2])

                    dx_corr = dx_lambdify_function(df['a_-3'], df['a_-2'], df['a_-1'], df['a_0'], df['a_1'], df['a_2'],
                                                   df['a_3'], df['a_4'], df['a_5'], df['a_6'], df['a_7'],
                                                   df['b_-3'], df['b_-2'], df['b_-1'], df['b_0'], df['b_1'], df['b_2'],
                                                   df['b_3'], df['b_4'], df['b_5'], df['b_6'], df['b_7'])
                    dy_corr = dy_lambdify_function(df['a_-3'], df['a_-2'], df['a_-1'], df['a_0'], df['a_1'], df['a_2'],
                                                   df['a_3'], df['a_4'], df['a_5'], df['a_6'], df['a_7'],
                                                   df['b_-3'], df['b_-2'], df['b_-1'], df['b_0'], df['b_1'], df['b_2'],
                                                   df['b_3'], df['b_4'], df['b_5'], df['b_6'], df['b_7'])

                    # Plot Vector Field
                    plt.clf()
                    fig = plt.figure(1)
                    ax = fig.add_subplot(111)
                    background = df['error']

                    if PLOT_BACKGROUND:
                        # Get nodemap data
                        nodemap = Nodemap(name=file[:-12] + '.txt', folder=DATA_PATH)
                        data = InputData(nodemap)
                        data.calc_stresses(material)
                        # Remove unnecessary data to speed up evaluation
                        # The mask must be at least as large as the evaluation area
                        mask = np.all([data.coor_x < 15,
                                       data.coor_x > -15,
                                       data.coor_y < 15,
                                       data.coor_y > -15], axis=0)
                        data = apply_mask(data, mask)

                        fmin = 0.0
                        fmax = 0.3
                        num_colors = 120
                        contour_vector = np.linspace(fmin, fmax, num_colors, endpoint=True)
                        label_vector = np.linspace(fmin, fmax, 6, endpoint=True)

                        plot = ax.tricontourf(data.coor_x, data.coor_y, data.eps_vm * 100, contour_vector, extend='max')
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="5%", pad=0.2)
                        fig.colorbar(plot, ticks=label_vector, cax=cax, label='Von Mises eqv. strain [\\%]')

                    ax.scatter(0, 0, s=500, c='black', marker='x', linewidths=5)
                    ax.plot([-4, 0], [0, 0], color='black', linewidth=3, linestyle='dashed')
                    for i in range(len(df)):
                        ax.arrow(df['dx'][i], df['dy'][i], dx_corr[i], dy_corr[i], head_width=0.1, head_length=0.1,
                                 fc='k', ec='k', linewidth=0.5, length_includes_head=True)

                    ax.scatter(df['dx'], df['dy'], color='black', marker='o', s=10)
                    ax.set_xlabel('$x$ [mm]')
                    ax.set_ylabel('$y$ [mm]')
                    ax.axis('image')
                except:
                    continue

                # set axis limits
                ax.set_xlim([-4, 4])
                ax.set_ylim([-4, 4])

                plot_name = f'{mode}_{file[:-4]}_Vector_Field.png'
                plt.savefig(os.path.join(formula_dir, plot_name), bbox_inches='tight')
