from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from crackpy.crack_detection.correction import CrackTipCorrectionGridSearch
from crackpy.fracture_analysis.optimization import OptimizationProperties


class CrackTipCorrectionRandomSamples(CrackTipCorrectionGridSearch):
    def __init__(self, data, crack_tip, crack_angle, material):
        super().__init__(data, crack_tip, crack_angle, material)

    def correct_crack_tip_random_samples(
            self,
            opt_props: OptimizationProperties,
            x_min: float,
            x_max: float,
            y_min: float,
            y_max: float,
            n_samples: int,
            workers: int = 1,
            verbose: bool = False,
            seed: int = 0
    ):
        """Correct crack tip position using random search of the smallest Williams fitting error.
        Warning! Bruteforce method with long runtime. Parallelized version.

        Args:
            opt_props: OptimizationProperties used for the Williams fitting
            x_min: minimum x coordinate for the grid search relative to the current crack tip position
            x_max: maximum x coordinate for the grid search relative to the current crack tip position
            y_min: minimum y coordinate for the grid search relative to the current crack tip position
            y_max: maximum y coordinate for the grid search relative to the current crack tip position
            n_samples: number of random samples
            workers: number of parallel jobs
            verbose: If True, print the current iteration
            seed: random seed

        Returns:
            crack tip correction as array of x and y coordinate, dataframe of error values for each grid point

        """
        # Choose random samples of pairs (delta_x, delta_y) from the Latin hypercube [x_min, x_max] x [y_min, y_max]
        np.random.seed(seed)
        delta_x = np.random.uniform(x_min, x_max, n_samples)
        delta_y = np.random.uniform(y_min, y_max, n_samples)
        shifts_x_y = np.concatenate((delta_x.reshape(-1, 1), delta_y.reshape(-1, 1)), axis=1)

        delta_phi = 0
        results = []

        with ProcessPoolExecutor(max_workers=workers) as executor:
            for shift_x_y in shifts_x_y:
                results.append(
                    executor.submit(self._parallel_grid_search, shift_x_y, delta_phi, opt_props, verbose))

        columns = ['dx', 'dy', 'dphi', 'error']
        for term in opt_props.terms:
            columns.append(f'a_{term}')
        for term in opt_props.terms:
            columns.append(f'b_{term}')

        df = pd.DataFrame(columns=columns)
        max_error = 1e10
        ct_corr = [0, 0]
        for i, result in enumerate(results):
            output = result.result()
            error = output[3]
            if error < max_error:
                max_error = error
                ct_corr = [output[0], output[1]]
            output[0] += self.crack_tip[0]
            output[1] += self.crack_tip[1]
            output[2] += self.crack_angle
            df.loc[i] = output

        return ct_corr, df