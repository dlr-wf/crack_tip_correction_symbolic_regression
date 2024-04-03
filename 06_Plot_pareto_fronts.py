"""
    This script plots the Pareto front of the symbolic regression results and saves the formulas in a latex table.
    Needed:
        - Folder containing the symbolic regression results

    Output:
        - folder 06_Pareto_Plots containing the plots and latex table
"""

import matplotlib.pyplot as plt
import pandas as pd
import os
from cracktipcorr.equations import sympy_converter
from sympy.physics.units import millimeter, newton
import json

# General settings
OUTPUT_PATH = os.path.join('06_Pareto_Plots')
inline_annotation = True

# Set plot settings
plt.rcParams.update({'font.size': 20})
plt.rcParams['figure.figsize'] = [10, 8]
plt.rcParams['figure.dpi'] = 300

# check if output path exists
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

#################################################################################################
RESULTs = {}

# Mode I
RESULTs['log_mode_I_dx'] = {
    'path': '05_1_PhySO_log_mode_I',
    'direction': 'dx',
    'title': 'Pareto front - Mode I correction - $\Delta x$',
    'label': 'eqs_pareto_I_dx',
    'caption': r"Pareto front - Mode I correction - $\Delta x$"
}
RESULTs['log_mode_I_dy'] = {
    'path': '05_1_PhySO_log_mode_I',
    'direction': 'dy',
    'title': 'Pareto front - Mode I correction - $\Delta y$',
    'label': 'eqs_pareto_I_dy',
    'caption': r"Pareto front - Mode I correction - $\Delta y$"
}

# Mode II
RESULTs['log_mode_II_dx'] = {
    'path': '05_2_PhySO_log_mode_II',
    'direction': 'dx',
    'title': 'Pareto front - Mode II correction - $\Delta x$',
    'label': 'eqs_pareto_II_dx',
    'caption': r"Pareto front - Mode II correction - $\Delta x$"
}
RESULTs['log_mode_II_dy'] = {
    'path': '05_2_PhySO_log_mode_II',
    'direction': 'dy',
    'title': 'Pareto front - Mode II correction - $\Delta y$',
    'label': 'eqs_pareto_II_dy',
    'caption': r"Pareto front - Mode II correction - $\Delta y$"
}

# Mixed mode
RESULTs['log_mixed_mode_dx'] = {
    'path': '05_3_PhySO_log_mixed_mode',
    'direction': 'dx',
    'title': 'Pareto front - Mode I/II correction - $\Delta x$',
    'label': 'eqs_pareto_II_dx',
    'caption': r"Pareto front - Mode I/II correction - $\Delta x$"
}
RESULTs['log_mixed_mode_dy'] = {
    'path': '05_3_PhySO_log_mixed_mode',
    'direction': 'dy',
    'title': 'Pareto front - Mode I/II correction - $\Delta y$',
    'label': 'eqs_pareto_II_dy',
    'caption': r"Pareto front - Mode I/II correction - $\Delta y$"
}


with open(os.path.join(OUTPUT_PATH, 'latex_table.txt'), 'w') as txt_file:

    json_export = {}

    for RESULT in RESULTs:
        json_export[RESULT] = {}
        print(f'Plotting {RESULT}')
        FOLDER = RESULTs[RESULT]['path']
        direction = RESULTs[RESULT]['direction']
        title = RESULTs[RESULT]['title']
        caption = RESULTs[RESULT]['caption']
        label = RESULTs[RESULT]['label']

        INP_FILE = 'curves_pareto.csv'
        print(f'Plotting {FOLDER}')
        INP_PATH = os.path.join(FOLDER, direction, INP_FILE)

        OUT_FILE = f'pareto_front_{FOLDER}_{direction}.png'
        OUT_FILE_TEX = f'pareto_front_{FOLDER}_{direction}.txt'

        txt_file.write(r"\begin{table}[htbp!]" + "\n")
        txt_file.write(r"\centering" + "\n")
        txt_file.write(r" \begin{tabular}{|c c c c c|} " + "\n")
        txt_file.write(r"\hline" + "\n")
        txt_file.write(r"\multicolumn{5}{|c|}{" + f"{caption}" + r"} \\" + "\n")
        txt_file.write(r"\hline" + "\n")
        txt_file.write(r"\# & Complexity & Reward & RMSE & Equation \\ [0.5ex]" + "\n")
        txt_file.write(r"\hline" + "\n")
        OUT_PATH = os.path.join(OUTPUT_PATH, OUT_FILE)

        # Read data
        try:
            df = pd.read_csv(INP_PATH)

            # Plot
            fig, ax = plt.subplots()
            ax.plot(df['complexity'], df['rmse'], '-o', color='blue', markersize=10)
            ax.set_xlabel('Complexity [1]')
            ax.set_ylabel('Root Mean Squared Error [1]')
            ax.set_title(f'{title}')

            for index, row in df.iterrows():
                # convert formula to latex using sympy
                str_formula = row['expression']
                complexity = row['complexity']
                reward = row['reward']
                rmse = row['rmse']
                susb_dict = {}
                if 'k' in df.columns:
                    susb_dict['k'] = row['k']
                if 'm' in df.columns:
                    susb_dict['m'] = row['m'] * millimeter
                if 'n' in df.columns:
                    susb_dict['n'] = row['n'] * newton

                sympy_latex, sympy_formula = sympy_converter(str_formula,
                                                             susb_dict=susb_dict,
                                                             use_units=False,
                                                             evalf=True,
                                                             simplify=True,
                                                             tolerance=0.05,
                                                             round_digits=5,
                                                             latex_output=True)

                json_export[RESULT][index] = {'complexity': complexity,
                                              'reward': reward,
                                              'rmse': rmse,
                                              'latex': sympy_latex,
                                              'formula': sympy_formula}
                if inline_annotation:
                    ax.annotate(f"       ${sympy_latex}$", (row['complexity'], row['rmse']), rotation=45, fontsize=10)

                ax.annotate(f"{index}", (row['complexity'], row['rmse']), rotation=45)
                txt_file.write(f"{index} & {complexity} & {reward:8.5f} & {rmse:8.5f} & ${sympy_latex}$" + r"\\" + "\n")

            plt.savefig(OUT_PATH, bbox_inches='tight', dpi=300)
            plt.close()
            txt_file.write(r"\hline" + "\n")
            txt_file.write(r"\end{tabular}" + "\n")
            txt_file.write(r"\caption{" + caption + "}\n")
            txt_file.write(r"\label{table:" + f"{label}" + "}\n")
            txt_file.write(r"\end{table}" + "\n\n")

        except:
            print(f'File {INP_PATH} not found')


with open(os.path.join(OUTPUT_PATH, 'pareto_front.json'), 'w') as outfile:
    json.dump(json_export, outfile, indent=4, default=str)
