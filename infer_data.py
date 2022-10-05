# TODO This is unused as it was implemented in notebook instead. Might need to be coded up for cleanliness later.
from ml_utils.data_utils.preprocess_utils import infer_incident_data
import os

if __name__ == '__main__':
    scenario = 'motorway'
    experiment_name = 'incident3'
    path = f'{scenario}/Results/{experiment_name}/simulations'
    result_folders = os.listdir(path)
    simulation_folders = [folder for folder in result_folders if folder.startswith(experiment_name)]
    simulation_paths = [f'{path}/{simulation}' for simulation in simulation_folders[:10]]

    infer_incident_data(simulation_path=simulation_paths[0])