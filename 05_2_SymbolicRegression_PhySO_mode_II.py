"""
    This script is used to perform symbolic regression on crackpy grid data using the PhySO package.
    The results are used in our work.
    This configuration determines delta_x and delta_y for a pure mode I loading.
    The code was adapted using the examples from:
    https://github.com/WassimTenachi/PhySO/blob/main/demos/demos_sr/demo_mechanical_energy/demo_mechanical_energy.py

    Needed:
        - Folder with grid data files

    Output:
        - symbolic_regression folder containing the symbolic regression results
"""

# External packages
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Internal code import
from cracktipcorr.utils import read_grid_data
import physo
from physo.learn import monitoring
from physo.task import benchmark

# Device
DEVICE = 'cpu'
# if torch.cuda.is_available():
#    DEVICE = 'cuda'
print(f'Using device: {DEVICE}')
torch.cuda.is_available()

# Fix seed for reproducibility to 42
torch.manual_seed(42)
np.random.seed(42)

# #### Data points

# Define the result files to read
DATA_PATH = os.path.join('04_CrackPy_random_evaluation_pipeline', 'samples')

# Chose if only mode I should be considered
mode = 'II'
williams_terms_a = ['a_-3', 'a_-2', 'a_-1', 'a_0', 'a_1', 'a_2', 'a_3', 'a_4', 'a_5', 'a_6', 'a_7']
williams_terms_b = ['b_-3', 'b_-2', 'b_-1', 'b_0', 'b_1', 'b_2', 'b_3', 'b_4', 'b_5', 'b_6', 'b_7']

# Create output directory if not existing
LOG_DIR = '05_2_PhySO_log_mode_II'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Define dictionary for json output
dump_dict = {}
dump_dict['williams_terms_a'] = williams_terms_a
dump_dict['williams_terms_b'] = williams_terms_b
dump_dict['mode'] = mode
dump_dict['start_time'] = datetime.now()

# Read data from each result file
a_data, b_data, delta_x, delta_y, training_data_files = read_grid_data(DATA_PATH,
                                                                       only_mode=mode,
                                                                       williams_terms_a=williams_terms_a,
                                                                       williams_terms_b=williams_terms_b,
                                                                       endswith="samples.csv")

# Dataset
print(f'The dataset contains {len(a_data)} samples from {len(training_data_files)} files.')
dump_dict['training_data_files'] = training_data_files

# Create the input and output data for regression
X_array = np.concatenate((a_data, b_data), axis=1).T

trainings = [[delta_x, 'dx'],
             [delta_y, 'dy']]

for y_array, name in trainings:
    LOG_PATH = os.path.join(LOG_DIR, name)

    # Create output directory if not existing
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)

    # Save the data
    n_dim = X_array.shape[0]
    fig, ax = plt.subplots(n_dim, 1, figsize=(10, 5))
    for i in range(n_dim):
        curr_ax = ax if n_dim == 1 else ax[i]
        curr_ax.plot(X_array[i], y_array, 'k.', )
        curr_ax.set_xlabel("X[%i]" % (i))
        curr_ax.set_ylabel("y")
    plt.savefig(os.path.join(LOG_PATH, 'inp_data.png'))

    # #### Run config
    # Stack of all input variables
    X = torch.tensor(X_array).to(DEVICE)
    # Output of symbolic function to guess
    y = torch.tensor(y_array).to(DEVICE)

    # ------ Constants ------
    const1 = torch.tensor(np.array(1.)).to(DEVICE)
    const2 = torch.tensor(np.array(2.)).to(DEVICE)
    const3 = torch.tensor(np.array(4.)).to(DEVICE)

    # ### Library config
    args_make_tokens = {
        # operations
        "op_names": ["mul", "add", "sub", "div", "abs", "inv", "n2", "neg", "exp", "log"],
        "use_protected_ops": True,
        # input variables
        "input_var_ids": {"a_(-3)": 0,
                          "a_(-2)": 1,
                          "a_(-1)": 2,
                          "a_(0)": 3,
                          "a_(1)": 4,
                          "a_(2)": 5,
                          "a_(3)": 6,
                          "a_(4)": 7,
                          "a_(5)": 8,
                          "a_(6)": 9,
                          "a_(7)": 10,
                          "b_(-3)": 11,
                          "b_(-2)": 12,
                          "b_(-1)": 13,
                          "b_(0)": 14,
                          "b_(1)": 15,
                          "b_(2)": 16,
                          "b_(3)": 17,
                          "b_(4)": 18,
                          "b_(5)": 19,
                          "b_(6)": 20,
                          "b_(7)": 21, },
        # Units: $\text{N} \cdot \text{mm}^{-1-n/2}$
        "input_var_units": {"a_(-3)": [1, 1 / 2],
                            "a_(-2)": [1, 0],
                            "a_(-1)": [1, -1 / 2],
                            "a_(0)": [1, -1],
                            "a_(1)": [1, -3 / 2],
                            "a_(2)": [1, -2],
                            "a_(3)": [1, -5 / 2],
                            "a_(4)": [1, -3],
                            "a_(5)": [1, -7 / 2],
                            "a_(6)": [1, -4],
                            "a_(7)": [1, -9 / 2],
                            "b_(-3)": [1, 1 / 2],
                            "b_(-2)": [1, 0],
                            "b_(-1)": [1, -1 / 2],
                            "b_(0)": [1, -1],
                            "b_(1)": [1, -3 / 2],
                            "b_(2)": [1, -2],
                            "b_(3)": [1, -5 / 2],
                            "b_(4)": [1, -3],
                            "b_(5)": [1, -7 / 2],
                            "b_(6)": [1, -4],
                            "b_(7)": [1, -9 / 2], },
        "input_var_complexity": {"a_(-3)": 1.,
                                 "a_(-2)": 1.,
                                 "a_(-1)": 1.,
                                 "a_(0)": 1.,
                                 "a_(1)": 1.,
                                 "a_(2)": 1.,
                                 "a_(3)": 1.,
                                 "a_(4)": 1.,
                                 "a_(5)": 1.,
                                 "a_(6)": 1.,
                                 "a_(7)": 1.,
                                 "b_(-3)": 1.,
                                 "b_(-2)": 1.,
                                 "b_(-1)": 1.,
                                 "b_(0)": 1.,
                                 "b_(1)": 1.,
                                 "b_(2)": 1.,
                                 "b_(3)": 1.,
                                 "b_(4)": 1.,
                                 "b_(5)": 1.,
                                 "b_(6)": 1.,
                                 "b_(7)": 1., },
        # constants
        "constants": {"1": const1,
                      "2": const2,
                      "4": const3},
        "constants_units": {"1": [0, 0],
                            "2": [0, 0],
                            "4": [0, 0]},
        "constants_complexity": {"1": 1.,
                                 "2": 1.,
                                 "4": 1., },
        # free constants (comment this out if you don't want free constants)
        "free_constants": {"k", 'm', 'n'},
        "free_constants_init_val": {"k": 1., "m": 1., "n": 1.},
        "free_constants_units": {"k": [0, 0], "m": [0, 1], "n": [1, 0]},
        "free_constants_complexity": {"k": 1., "m": 1., "n": 1.},
    }

    library_config = {"args_make_tokens": args_make_tokens,
                      "superparent_units": [0, 1],
                      "superparent_name": "delta",
                      }

    # ### Learning config
    reward_config = {
        "reward_function": physo.physym.reward.SquashedNRMSE,  # PHYSICALITY
        "zero_out_unphysical": True,
        "zero_out_duplicates": False,
        "keep_lowest_complexity_duplicate": False,
        "parallel_mode": False,
        "n_cpus": 60,
    }

    BATCH_SIZE = int(1e3)
    MAX_LENGTH = 35
    GET_OPTIMIZER = lambda model: torch.optim.Adam(
        model.parameters(),
        lr=0.0025,  # 0.001, #0.0050, #0.0005, #1,  #lr=0.0025
    )

    learning_config = {
        # Batch related
        'batch_size': BATCH_SIZE,
        'max_time_step': MAX_LENGTH,
        'n_epochs': 1000,
        # Loss related
        'gamma_decay': 0.7,
        'entropy_weight': 0.005,
        # Reward related
        'risk_factor': 0.05,
        'rewards_computer': physo.physym.reward.make_RewardsComputer(**reward_config),
        # Optimizer
        'get_optimizer': GET_OPTIMIZER,
        'observe_units': True,
    }

    # ### Free constant optimizer config
    free_const_opti_args = {
        'loss': "MSE",
        'method': 'LBFGS',
        'method_args': {
            'n_steps': 15,
            'tol': 1e-8,
            'lbfgs_func_args': {
                'max_iter': 4,
                'line_search_fn': "strong_wolfe",
            },
        },
    }

    # ### Priors config
    priors_config = [
        # ("UniformArityPrior", None),
        # LENGTH RELATED
        ("HardLengthPrior", {"min_length": 4, "max_length": MAX_LENGTH, }),
        ("SoftLengthPrior", {"length_loc": 6, "scale": 5, }),
        # RELATIONSHIPS RELATED
        ("NoUselessInversePrior", None),
        ("PhysicalUnitsPrior", {"prob_eps": np.finfo(np.float32).eps}),  # PHYSICALITY
        ("NestedFunctions", {"functions": ["exp", ], "max_nesting": 1}),
        ("NestedFunctions", {"functions": ["log", ], "max_nesting": 1}),
        # ("NestedTrigonometryPrior", {"max_nesting": 1}),
        # ("OccurrencesPrior", {"targets": ["1", ], "max": [3, ]}),
    ]

    # ### Cell config
    cell_config = {
        "hidden_size": 128,
        "n_layers": 1,
        "is_lobotomized": False,
    }

    # ### Logger
    save_path_training_curves = os.path.join(LOG_PATH, 'curves.png')
    save_path_log = os.path.join(LOG_PATH, 'log.log')

    run_logger = monitoring.RunLogger(save_path=save_path_log,
                                      do_save=True)

    run_visualiser = monitoring.RunVisualiser(epoch_refresh_rate=5,
                                              save_path=save_path_training_curves,
                                              do_show=False,
                                              do_prints=True,
                                              do_save=True, )

    # ### Run config
    run_config = {
        "learning_config": learning_config,
        "reward_config": reward_config,
        "free_const_opti_args": free_const_opti_args,
        "library_config": library_config,
        "priors_config": priors_config,
        "cell_config": cell_config,
        "run_logger": run_logger,
        "run_visualiser": run_visualiser,
    }

    dump_dict['run_config'] = run_config

    # ## Dummy epoch for prior tuning
    benchmark.dummy_epoch(X, y, run_config)

    # ## Run
    rewards, candidates = physo.fit(X, y, run_config,
                                    stop_reward=0.9999,
                                    stop_after_n_epochs=5)

    # ## Results
    run_visualiser.make_visualisation()
    run_visualiser.save_visualisation()
    run_visualiser.save_data()

    # ### Save config
    dump_dict['end_time'] = datetime.now()
    with open(os.path.join(f"{LOG_PATH}", "symbolic_regression.json"), 'w') as json_file:
        json.dump(dump_dict, json_file, indent=4, default=str)
