#### Comprehensive exam computing artifact
#### Author: Brian Kyanjo
#### Supervisor: Prof. Donna Calhoun

There are three scripts: [test.py](./test.py), [approximate_solver.py](./approximate_solver.py), [exact_solver.py](./exact_solver.py), and a notebook: [main.ipynb](./main.ipynb) in this folder. 
* The two scripts [approximate_solver.py](./approximate_solver.py) and [exact_solver.py](./exact_solver.py) are called by the [test.py](./test.py) script which contains other functions like initial and boundary conditions, bathymetry, problem_test, and the Riemannsoln which is prompt in the note book to output samples of visualizations depending on the selected case.
* The notebook  [main.ipynb](./main.ipynb) imports the [test.py](./test.py) script to prompt the Riemannsoln function and other functions mentioned above to perform simulations. All the simulations are performed in this notebook, and it gives the user chance to cary out different test cases.

__Usage:__ To run this artifact, the user has to read the user manaul provided in the [main.ipynb](./main.ipynb) notebook at the begining of the page. All adjustments are well explained in this notebook for all variables,parameters and cases. Executing this notebook will prompt the Riemannsoln function imported from the [test.py](./test.py) script to output temporal evolution of either height, velocity or momentum field.

