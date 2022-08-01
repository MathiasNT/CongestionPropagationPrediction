import os
import xml.etree.ElementTree as ET
from sumolib import checkBinary
import glob

def setup_run(scenario_folder, edge_filename, run_num, begin, end):
    # Create temp add file
    add_path = f'{scenario_folder}/Simulations/Base/edgedata'
    add_base = f'{add_path}.add.xml'
    add_temp = f'{add_path}_temp{run_num}.add.xml' 
    edge_file = f'{scenario_folder}/Results/{edge_filename}{run_num}.xml'
    
    xml_tree = ET.parse(add_base)    
    xml_root = xml_tree.getroot()
    edge_data_settings = xml_root[0]
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

    sumoCmd = [checkBinary('sumo'), "-c", config_temp, "--begin", f"{begin}", "--end", f"{end}", "--start", "1", "--quit-on-end", "1"]
    return sumoCmd

def cleanup_temp_files(scenario_folder):
    sim_folder = f'{scenario_folder}/Simulations/Base'
    for file in glob.glob(f"{sim_folder}/*temp*"):
        os.remove(file)
