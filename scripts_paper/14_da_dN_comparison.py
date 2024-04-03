import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def is_stringfloat(element: str) -> bool:
    """Checks if element can be converted to a float"""
    try:
        float(element)
        return True
    except ValueError:
        return False


# Set matplotlib settings
plt.rcParams.update({
    "font.size": 20,
    "text.usetex": True,
    "font.family": "Computer Modern",
    "figure.figsize": [8, 6],
    "figure.dpi": 300
})
CRACK_DETECTION_PATHS = [
    os.path.join("13_Application", "uniaxial", "DIC"),
    os.path.join("13_Application", "biaxial", "DIC")
]

for PATH in CRACK_DETECTION_PATHS:
    # open crack correction results file and read the content
    with open(os.path.join(PATH, "crack_correction_results.txt"), "r") as in_file:
        still_need_to_read_header = True
        for line in in_file:
            if line.startswith("#"):
                continue
            elif still_need_to_read_header:
                columns = line.strip('\n').strip(' ').split(',')
                columns = [element.strip(' ') for element in columns]
                df = pd.DataFrame(columns=columns)
                still_need_to_read_header = False
                continue
            else:
                values = []
                for val in line.strip('\n').split(','):
                    val = val.strip(' ')
                    if is_stringfloat(val):
                        val = float(val)
                    values.append([val])

                # read values of tagged content
                columns_to_values = pd.DataFrame.from_dict(dict(zip(columns, values)))
                df = pd.concat([df, columns_to_values], ignore_index=True)

    print(df.columns)

    # Calculate crack lengths
    df['CT corr x [mm]'] = df['CT x [mm]'] + df['Corr dx [mm]']
    df['CT corr y [mm]'] = df['CT y [mm]'] + df['Corr dy [mm]']
    df['N'] = df['Cycle [1]']

    # differences
    df['CT dx [mm]'] = df['CT x [mm]'].diff()
    df['CT dy [mm]'] = df['CT y [mm]'].diff()
    df['CT corr dx [mm]'] = df['CT corr x [mm]'].diff()
    df['CT corr dy [mm]'] = df['CT corr y [mm]'].diff()
    df['dN'] = df['Cycle [1]'].diff()

    # delete all rows where dN is 0
    df = df[df['dN'] != 0]

    # means
    df['CT dx [mm] mean'] = df['CT dx [mm]'].mean()
    df['CT dy [mm] mean'] = df['CT dy [mm]'].mean()
    df['CT corr dx [mm] mean'] = df['CT corr dx [mm]'].mean()
    df['CT corr dy [mm] mean'] = df['CT corr dy [mm]'].mean()

    # Filter for cycles
    if PATH == os.path.join("13_Application", "uniaxial", "1221EBr0001_DIC"):
        df = df[df['N'] > 450000]
    if PATH == os.path.join("13_Application", "biaxial", "1223TSt1_DIC"):
        df = df[df['N'] > 350000]

    # Plot delta of crack length da_x over crack length a_x
    plt.plot(df['N'], df['CT dx [mm]'] / df['dN'], label='LIM', color="gray", linestyle='--')
    plt.plot(df['N'], df['CT corr dx [mm]'] / df['dN'], label='Correction', color="green")
    plt.xlabel('Load cycle $N$')
    plt.ylabel('$\Delta a / \Delta N$ [mm/cycle]')
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    plt.legend()
    plt.savefig(os.path.join(PATH, "lineplot_da_x_dN_over_N.png"))
    plt.close()
