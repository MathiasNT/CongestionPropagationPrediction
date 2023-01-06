# Code for Congestion propagation prediction after accidents using incident reports

## Code for simulating traffic with incidents
The code for running simulations with incidents can be found in the util_folder/simulation_utils and be run using the libsumo_run_pool.py. Note that this runs multiple simulations in parallel.

## Code for inferring data from simulations
Using the notebook notebooks/util_notebooks/data_preprocess2.ipynb the resulting congestion propagation data can be inferred from the simulated experiments

## Code for inferring data from real world data
Simillarly congestion propagation data can be inferred from real world data using the notebook notebooks/util_notebooks/full_realworld_dataset_inf.ipynb

## Training models
Training models can be done using model_train.py using the config files in configs/

## Results notebooks
Notebooks with the results can be found in saved_models under the experiment names

### Simulations for experiment results
The raw simulation data can be provided upon request
