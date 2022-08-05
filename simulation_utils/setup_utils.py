import os
import xml.etree.ElementTree as ET
from sumolib import checkBinary
import glob

def setup_incident_sim(scenario_folder, simulation_name, run_num, begin, end, trip_info, verbose):
    # Create simulation folder and results filenames
    sim_num = 0
    simulation_folder = f'{scenario_folder}/Results/{simulation_name}_{sim_num}'
    while os.path.isdir(simulation_folder):
        sim_num+=1
        simulation_folder = f'{scenario_folder}/Results/{simulation_name}_{sim_num}'
    os.mkdir(simulation_folder)
    edge_file = f'{simulation_folder}/edgedata.xml'
    trips_file = f'{simulation_folder}/tripsdata.xml'    

    # Create temp add file
    add_path = f'{scenario_folder}/Simulations/Base/edgedata'
    add_base = f'{add_path}.add.xml'
    add_temp = f'{add_path}_temp{run_num}.add.xml'
    xml_tree = ET.parse(add_base)    
    xml_root = xml_tree.getroot()
    edge_data_settings = xml_root.find('edgeData')
    edge_data_settings.set('file', edge_file)
    xml_tree.write(add_temp)

    # Create temp config file
    config_path = f'{scenario_folder}/Simulations/Base/simulation'
    config_base = f'{config_path}.sumo.cfg'
    config_temp = f'{config_path}_temp{run_num}.sumo.cfg'
    xml_tree = ET.parse(config_base)
    xml_root = xml_tree.getroot()
    add_elem = xml_root.find('input').find('additional-files')
    old_add_files = add_elem.get('value')
    add_elem.set('value', f'{old_add_files},edgedata_temp{run_num}.add.xml')
    xml_tree.write(config_temp)

    sumoCmd = [checkBinary('sumo'), "-c", config_temp, "--begin", f"{begin}", "--end", f"{end}"] #TODO update this with begin and end from incident
    
    if trip_info:
        sumoCmd = sumoCmd + ['--tripinfo-output', f'{simulation_folder}/trips.xml']

    if verbose:
        sumoCmd = sumoCmd + ['--error-log', f'{simulation_folder}/error_log.xml']
        sumoCmd = sumoCmd + ['--message-log', f'{simulation_folder}/message_log.xml']

    return {'sumoCmd':sumoCmd, 'simulation_folder':simulation_folder}

def setup_counterfactual_sim(scenario_folder, simulation_folder, run_num, begin, end, trip_info, verbose):
    # Create simulation folder and results filenames
    edge_file = f'{simulation_folder}/edgedata_counterfactual.xml'

    # Create temp add file
    add_path = f'{scenario_folder}/Simulations/Base/edgedata'
    add_base = f'{add_path}.add.xml'
    add_temp = f'{add_path}_temp{run_num}_counterfactual.add.xml'
    xml_tree = ET.parse(add_base)    
    xml_root = xml_tree.getroot()
    edge_data_settings = xml_root.find('edgeData')
    edge_data_settings.set('file', edge_file)
    xml_tree.write(add_temp)

    # Create temp config file
    config_path = f'{scenario_folder}/Simulations/Base/simulation'
    config_base = f'{config_path}.sumo.cfg'
    config_temp = f'{config_path}_temp{run_num}_counterfactual.sumo.cfg'
    xml_tree = ET.parse(config_base)
    xml_root = xml_tree.getroot()
    add_elem = xml_root.find('input').find('additional-files')
    old_add_files = add_elem.get('value')
    add_elem.set('value', f'{old_add_files},edgedata_temp{run_num}_counterfactual.add.xml')
    xml_tree.write(config_temp)

    sumoCmd = [checkBinary('sumo'), "-c", config_temp, "--begin", f"{begin}", "--end", f"{end}"] #TODO update this with begin and end from incident
    
    if trip_info:
        sumoCmd = sumoCmd + ['--tripinfo-output', f'{simulation_folder}/trips_counterfactual.xml']

    if verbose:
        sumoCmd = sumoCmd + ['--error-log', f'{simulation_folder}/error_log_counterfactual.xml']
        sumoCmd = sumoCmd + ['--message-log', f'{simulation_folder}/message_log_counterfactual.xml']

    return {'sumoCmd':sumoCmd, 'simulation_folder':simulation_folder}

def setup_gui_sim(scenario_folder, begin, end):
    # Create temp config file
    config_path = f'{scenario_folder}/Simulations/Base/simulation.sumo.cfg'

    sumoCmd = [checkBinary('sumo-gui'), "-c", config_path, "--begin", f"{begin}", "--end", f"{end}"]

    return {'sumoCmd':sumoCmd, 'simulation_folder':'temp'}

def cleanup_temp_files(scenario_folder):
    sim_folder = f'{scenario_folder}/Simulations/Base'
    for file in glob.glob(f"{sim_folder}/*temp*"):
        os.remove(file)
