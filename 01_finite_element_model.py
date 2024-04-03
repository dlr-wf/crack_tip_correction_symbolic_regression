import os
import shutil
import time
import ansys
from itertools import product
from ansys.mapdl.core import launch_mapdl

from cracktipcorr.fe_model import AnsysSimulation, kill_ansys, delete_ansys_leftovers


# Change default Ansys path if necessary
# from ansys.tools.path import change_default_ansys_path
# change_default_ansys_path('/ansys_inc/v231/ansys/bin/ansys231')

# Set output path
OUTPUT_PATH = '01_Simulation_Output'
ANSYS_PATH = 'ansys'

# Set boundary conditions for the parameter space
sigma_xxs = [-10.0, 0.0, 10.0]
sigma_yys = [0.0, 10.0, 20.0]
sigma_xys = [-10.0, 0.0, 10.0]

# Create output directory if not existing
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

for sigma_xx, sigma_yy, sigma_xy in product(sigma_xxs, sigma_yys, sigma_xys):
    if not os.path.exists(ANSYS_PATH):
        os.makedirs(ANSYS_PATH)
    # skip if all zero
    if sigma_xx == sigma_yy == sigma_xy == 0.0:
        print("Skipping all zero case")
    else:
        f_name = f"{sigma_xx:.2f}_{sigma_yy:.2f}_{sigma_xy:.2f}".replace("-", "n")
        print(f_name)
        start_time = time.time()
        try:
            mapdl = launch_mapdl(
                nproc=4,
                run_location=ANSYS_PATH,
                override=True,
                print_com=True,
            )
            ansys_simulation = AnsysSimulation(mapdl, OUTPUT_PATH)
            ansys_simulation.set_material_elements(youngs_modulus=72000, nuxy=0.33)
            ansys_simulation.set_geometry(width=100, height=100, a_w_ratio=0.5)
            ansys_simulation.set_mesh(esize=0.2)
            ansys_simulation.set_boundary_conditions(sigma_xx=sigma_xx, sigma_yy=sigma_yy, sigma_xy=sigma_xy)
            ansys_simulation.solve()
            ansys_simulation.export_nodemap(filename=f_name)
            ansys_simulation.export_vtk(filename=f_name)
            ansys_simulation.postprocessing_cint()
            ansys_simulation.plot(filename=f_name, max_stress=100)
            ansys_simulation.export_results(filename=f_name)
            print(ansys_simulation.cint_post)
            # ansys_simulation.mapdl.open_gui()
            ansys_simulation.exit()
            end_time = time.time() - start_time
            print(f"Simulation: {f_name}, runtime: {end_time:.2f} s")

        # Catch exceptions
        except ansys.mapdl.core.errors.LockFileException:
            delete_ansys_leftovers(ansys_folder=ANSYS_PATH)

        except ansys.mapdl.core.errors.MapdlExitedError:
            print('Mapdl Session Terminated. Retrying...')
            delete_ansys_leftovers(ansys_folder=ANSYS_PATH)

        except OSError:
            print('OSError. Retrying...')

        finally:
            shutil.rmtree(ANSYS_PATH)
            time.sleep(1)
            kill_ansys()

print("Done!")
