# Bayesian Deep Learning: Comparing approximations of the posterior distribution over neural network weights

`SWAG.py`: Contains `SWAG` class: The method to approximate the SWAG posterior
`KFAC.py`: CContains `KFAC` class: The method to approximate the SWAG posterior
`BMA.py`: Contains two the function to approximate the Bayesian Model Average with Monte Carlo

`BO.py`: Bayesian Optimasation of `Deterministic_net` class with `GPyOp` methods
`Calibration.py`: Contains two functions. One for calibrating `KFAC`, `SWAG` and `Deterministic_net` with temperature scaling, the other optimises `KFAC`and `SWAG`
`run.py`: Contains functions which runs `Deterministic_net`, `KFAC` and `SWAG`

`data.py` Contains the nessesay classes to use `torch` dataloader and load `MNIST`, `EMNIST`, `Fashion_MNIST` from `torchvision.datasets` and `make_moons` from `sklearn.dataset`

`main_swag.py`, `main_kfac.py` and `main_deterministic.py`: Contains code that makes it possible to run the above with relevant settings. 

`utils.py` and `utils_2.py`: Contains various plot and nessesary read/write methods. 




