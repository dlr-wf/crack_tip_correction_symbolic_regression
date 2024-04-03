import matplotlib.pyplot as plt
import pandas as pd
import os
from cracktipcorr.equations import sympy_converter
from sympy.physics.units import millimeter, newton

OUTPUT_PATH = os.path.join('06_Plots_Pareto')

# Set matplotlib settings
plt.rcParams.update({
    "font.size": 25,
    "text.usetex": True,
    "font.family": "Computer Modern",
    "figure.dpi": 300,
    "figure.figsize": [10, 8]
})

# check if output path exists
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

#################################################################################################

RESULTs = {}
# Mode I
RESULTs['log_mode_I_dx'] = {
    'path': '05_1_PhySO_log_mode_I',
    'direction': 'dx',
    'title': 'Pareto front - Mode I correction - $d_x$',
    'label': 'appendix_eqs_pareto_I_dx',
    'caption': r"Pareto formulas - Mode I correction - $d_x$",
    'indices_to_plot': [0, 1, 12],
    'mode': 'mode I'
}
RESULTs['log_mode_I_dy'] = {
    'path': '05_1_PhySO_log_mode_I',
    'direction': 'dy',
    'title': 'Pareto front - Mode I correction - $d_y$',
    'label': 'appendix_eqs_pareto_I_dy',
    'caption': r"Pareto formulas - Mode I correction - $d_y$",
    'indices_to_plot': [0, 1, 6],
    'mode': 'mode I'
}

# Mode II
RESULTs['log_mode_II_dx'] = {
    'path': '05_2_PhySO_log_mode_II',
    'direction': 'dx',
    'title': 'Pareto front - Mode II correction - $d_x$',
    'label': 'appendix_eqs_pareto_II_dx',
    'caption': r"Pareto formulas - Mode II correction - $d_x$",
    'indices_to_plot': [0, 4, 10],
    'mode': 'mode II'
}
RESULTs['log_mode_II_dy'] = {
    'path': '05_2_PhySO_log_mode_II',
    'direction': 'dy',
    'title': 'Pareto front - Mode II correction - $d_y$',
    'label': 'appendix_eqs_pareto_II_dy',
    'caption': r"Pareto formulas - Mode II correction - $d_y$",
    'indices_to_plot': [0, 1, 2, 9],
    'mode': 'mode II'
}

# Mixed mode
RESULTs['log_mixed_mode_dx'] = {
    'path': '05_3_PhySO_log_mixed_mode',
    'direction': 'dx',
    'title': 'Pareto front - Mixed mode correction - $d_x$',
    'label': 'appendix_eqs_pareto_mixed_dx',
    'caption': r"Pareto formulas - Mixed mode correction - $d_x$",
    'indices_to_plot': [2, 4, 11],
    'mode': 'mixed mode'
}
RESULTs['log_mixed_mode_dy'] = {
    'path': '05_3_PhySO_log_mixed_mode',
    'direction': 'dy',
    'title': 'Pareto front - Mixed mode correction - $d_y$',
    'label': 'appendix_eqs_pareto_mixed_dy',
    'caption': r"Pareto formulas - Mixed mode correction - $d_y$",
    'indices_to_plot': [2, 4, 8],
    'mode': 'mixed mode'
}

with open(os.path.join(OUTPUT_PATH, 'latex_table.txt'), 'w') as txt_file:

    for RESULT in RESULTs:
        print(f'Plotting {RESULT}')
        FOLDER = RESULTs[RESULT]['path']
        direction = RESULTs[RESULT]['direction']
        title = RESULTs[RESULT]['title']
        caption = RESULTs[RESULT]['caption']
        label = RESULTs[RESULT]['label']
        indices_to_plot = RESULTs[RESULT]['indices_to_plot']

        INP_FILE = 'curves_pareto.csv'
        print(f'Plotting {FOLDER}')
        INP_PATH = os.path.join('..', FOLDER, direction, INP_FILE)

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
        df = pd.read_csv(INP_PATH)

        # Plot
        fig, ax = plt.subplots()
        ax.plot(df['complexity'], df['rmse'], color='blue')
        ax.set_xlabel('Complexity [1]')
        ax.set_ylabel('RMSE [mm]')
        ax.set_title(f'{title}')

        for index, row in df.iterrows():
            """if index not in indices_to_plot:
                continue"""
            # convert formula to latex using sympy
            str_formula = row['expression']
            complexity = row['complexity']
            reward = row['reward']
            rmse = row['rmse']
            if 'k' in df.columns:
                susb_dict = {'k': row['k'],
                             'm': row['m'] * millimeter,
                             'n': row['n'] * newton,
                             }
            else:
                susb_dict = {}
            sp_formula, _ = sympy_converter(str_formula,
                                            susb_dict=susb_dict,
                                            use_units=False,
                                            evalf=True,
                                            simplify=True,
                                            tolerance=0.05,
                                            round_digits=5,
                                            latex_output=True)
            print(f'{index}: {str_formula}, {row["k"]}, {row["m"]}, {row["n"]}, {sp_formula}')
            ax.scatter(row['complexity'], row['rmse'], color='blue', marker='o', s=100)
            ax.annotate(f"{index}", (row['complexity']+0.01, row['rmse']+0.01), rotation=45)
            txt_file.write(f"{index} & {complexity} & {reward:8.5f} & {rmse:8.5f} & ${sp_formula}$" + r"\\" + "\n")
        plt.savefig(OUT_PATH, bbox_inches='tight', dpi=300)
        plt.close()
        txt_file.write(r"\hline" + "\n")
        txt_file.write(r"\end{tabular}" + "\n")
        txt_file.write(r"\caption{" + caption + "}\n")
        txt_file.write(r"\label{table:" + f"{label}" + "}\n")
        txt_file.write(r"\end{table}" + "\n\n")

# One plot per dx and dy
# use colorblind friendly colors
COLORS = {
    'mode I': "tab:blue",
    'mode II': "tab:orange",
    'mixed mode': "tab:green"
}

for coord in ["dx", "dy"]:
    print(f'Plotting {coord}')
    OUT_FILE = f'pareto_fronts_{coord}.png'
    OUT_PATH = os.path.join(OUTPUT_PATH, OUT_FILE)

    # initiate plot
    plt.clf()
    fig, ax = plt.subplots()

    for RESULT in RESULTs:
        direction = RESULTs[RESULT]['direction']
        if direction != coord:
            continue
        print(f'Plotting {RESULT}')
        FOLDER = RESULTs[RESULT]['path']
        title = RESULTs[RESULT]['title']
        caption = RESULTs[RESULT]['caption']
        label = RESULTs[RESULT]['label']
        indices_to_plot = RESULTs[RESULT]['indices_to_plot']
        mode = RESULTs[RESULT]['mode']

        INP_FILE = 'curves_pareto.csv'
        INP_PATH = os.path.join('..', FOLDER, direction, INP_FILE)

        # Read data
        df = pd.read_csv(INP_PATH)

        # Plot
        ax.plot(df['complexity'], df['rmse'], color=COLORS[mode], label=f'{mode}')
        #ax.legend()
        ax.set_xlabel('Complexity [1]')
        ax.set_ylabel('RMSE [mm]')
        ax.set_xticks(range(4, 22, 2))

        for index, row in df.iterrows():
            if index not in indices_to_plot:
                continue
            ax.scatter(row['complexity'], row['rmse'], color=COLORS[mode], marker='o', s=100)
            ax.annotate(f"{index}", (row['complexity']+0.01, row['rmse']+0.01), rotation=45)
    plt.legend()
    plt.savefig(OUT_PATH, bbox_inches='tight', dpi=300)
    plt.close(fig)
