import os
from tqdm import tqdm

import matplotlib.pyplot as plt

from crackpy.crack_detection.line_intercept import plot_grid_errors as plot_errors
from cracktipcorr.samples import CrackTipCorrectionRandomSamples
from crackpy.fracture_analysis.data_processing import InputData
from crackpy.fracture_analysis.optimization import OptimizationProperties
from crackpy.structure_elements.data_files import Nodemap
from crackpy.structure_elements.material import Material


# Set colormap
plt.rcParams['image.cmap'] = 'coolwarm'
plt.rcParams['figure.dpi'] = 300

# Settings
DATA_PATH = os.path.join('01_Simulation_Output')
OUTPUT_PATH = '04_CrackPy_random_evaluation_pipeline'

if not os.path.exists(os.path.join(OUTPUT_PATH, 'samples')):
    os.makedirs(os.path.join(OUTPUT_PATH, 'samples'))
if not os.path.exists(os.path.join(OUTPUT_PATH, 'plots')):
    os.makedirs(os.path.join(OUTPUT_PATH, 'plots'))

crack_tip = [0, 0]
crack_angle = 0.0
material = Material(E=72000, nu_xy=0.33, sig_yield=350)

opt_props = OptimizationProperties(
    # --> see Rethore paper for details
    angle_gap=45,
    min_radius=5,
    max_radius=10,
    tick_size=0.25,
    terms=[-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
)

# iterate over all files with "nodemap.txt" extension in Data folder
files = os.listdir(DATA_PATH)
files = [file for file in files if file.endswith("nodemap.txt")]
for file in tqdm(files):
    nodemap = Nodemap(name=file, folder=DATA_PATH)
    data = InputData(nodemap)
    data.read_header()
    data.calc_stresses(material)

    correction_samples = CrackTipCorrectionRandomSamples(data, crack_tip, crack_angle, material)
    _, df = correction_samples.correct_crack_tip_random_samples(
        opt_props,
        x_min=-3,
        x_max=3,
        y_min=-3,
        y_max=3,
        n_samples=1000,
        workers=20,
        verbose=False,
        seed=42
    )
    plot_errors(df, fname=file[:-4] + '_samples.png',
                folder=os.path.join(OUTPUT_PATH, 'plots'))
    df.to_csv(os.path.join(OUTPUT_PATH, 'samples', file[:-4] + '_samples.csv'), index=False)
