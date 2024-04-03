# contains some helper functions

import os
import pandas as pd
import numpy as np

def clear_ansys_directory(ansys_folder: str):
    """Clears all solver output files from the ansys_folder.
    This is useful for cleaning up the ansys_folder before a new run.

    :param ansys_folder: Path to the folder containing the solver output files.
    """

    files = os.listdir(ansys_folder)
    delete_endings = ('.out', '_.inp', '.db', '.DSP', '.esav', '.mntr', '.iges', '.page', '.sh',
                      '.lock', '.rst', '.err', '.esav', '.full', '.stat', '.log', 'anstmp')
    for ansys_file in files:
        if ansys_file.endswith(delete_endings):
            os.remove(os.path.join(ansys_folder, ansys_file))


def read_grid_data(GRID_DATA_PATH,
                   only_mode: str = None,
                   williams_terms_a=['a_-1', 'a_0', 'a_1', 'a_2', 'a_3', 'a_4', 'a_5'],
                   williams_terms_b=['a_-1', 'a_0', 'a_1', 'a_2', 'a_3', 'a_4', 'a_5'],
                   endswith="grid.csv"):
    """
    Read the grid data from the GRID_DATA_PATH folder

    Parameters
    ----------
    GRID_DATA_PATH : str
        Path to the folder containing the grid data files
    only_mode : None, 'I' or 'II', optional
    williams_terms_a : list, optional
        List of Williams terms for the a coefficients, by default ['a_-1', 'a_0', 'a_1', 'a_2', 'a_3', 'a_4', 'a_5']
    williams_terms_b : list, optional
        List of Williams terms for the b coefficients, by default ['a_-1', 'a_0', 'a_1', 'a_2', 'a_3', 'a_4', 'a_5']
    endswith : str, optional
        File ending of the grid data files, by default "grid.csv"

    Returns
    -------
    a_data : list
        List of a coefficients for each grid data file
    b_data : list
        List of b coefficients for each grid data file
    delta_x : list
        List of delta x values for each grid data file
    delta_y : list
        List of delta y values for each grid data file
    training_data_files : list
        List of file names of the grid data files

    """
    assert only_mode in [None, 'I', 'II'], 'only_mode must be None, "I" or "II"'
    # Read data from each result file
    a_data = []
    b_data = []
    delta_x = []
    delta_y = []
    training_data_files = []
    for file in os.listdir(GRID_DATA_PATH):
        if file.endswith(endswith):
            stresses = file.replace('n', '-').split('_')
            sigma_xx = float(stresses[0])
            sigma_yy = float(stresses[1])
            sigma_xy = float(stresses[2])

            read_file = True
            if sigma_yy == 0 and sigma_xy == 0.0:  # always skip pure T-stress case
                read_file = False

            if only_mode == 'I':
                if sigma_xy != 0.0 or sigma_yy == 0.0:
                    read_file = False
            elif only_mode == 'II':
                if sigma_yy != 0.0 or sigma_xy == 0.0:
                    read_file = False

            if read_file:
                print(f'reading file: {file}')
                training_data_files.append(file)
                path = os.path.join(GRID_DATA_PATH, file)
                df2 = pd.read_csv(path)
                a_data.append(df2[williams_terms_a].values)
                b_data.append(df2[williams_terms_b].values)

                delta_x.append(- df2['dx'])
                delta_y.append(- df2['dy'])

    # Convert lists to arrays
    a_data = np.concatenate(a_data)
    b_data = np.concatenate(b_data)
    delta_x = np.concatenate(delta_x)
    delta_y = np.concatenate(delta_y)

    return a_data, b_data, delta_x, delta_y, training_data_files
