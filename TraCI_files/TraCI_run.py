#TODO Implement parallel
#TODO test if config is only used on startup
#TODO Check if information comes from all threads
#TODO Implement naming for results for parallel runs
#TODO Check if runs are deterministic or if we can change it up

import os
import sys
import argparse
import numpy as np
from sumolib import checkBinary
#import traci
import libsumo as traci
import xml.etree.ElementTree as ET
from time import time
from multiprocessing import Process

from incident_utils import block_lanes

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare encironment varialbe 'SUMO_HOME'")


def get_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--gui", action="store_true",
                            default=False, help="run the gui version of sumo")
    arg_parser.add_argument("--scenario", choices=['motorway', 'national', 'urban', 'experiment'], help='Which scenatio to run.', required=True)
    arg_parser.add_argument("--begin", type=int, default=0, help="Start time of the simulator")
    arg_parser.add_argument("--end", type=int, default=86400, help="End time of the simulator")
    arg_parser.add_argument("--edge_filename", type=str, default="edgedata", help="The name of the edge data file without xml extension.")
    arg_parser.add_argument("--n_runs", type=int, default=1, help="The number of simulations to run in parallel")
    args = arg_parser.parse_args()
    return args

def setup_run(scenario_folder, edge_filename, run_num):
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

    return config_temp


# contrains Traci control loop
def run(sumoCmd, start_step, end_step):
    start_time = time()
    step = start_step
    incidents = []    
   
    # 'full block' example with gradual release. Not sure if the simulation looks real but I don't know how much better we can get. However this here is as good as I expect QTIP would have been if not better
    #incidents = ['48290550_0_300_1100_1200','48290550_1_300_1100_1200','48290550_2_300_1100_1200','48290550_3_300_1100_1600']   
    
    traci.start(sumoCmd)
    
    while traci.simulation.getMinExpectedNumber() > 0 and traci.simulation.getTime() <= end_step:
        incidents = block_lanes(incidents, step)

        traci.simulationStep()
        
        step+=1

    traci.close()
    sys.stdout.flush()
    end_time = time()
    print(f'finished in {end_time - start_time}')


# main entry point
if __name__ == "__main__":
    args = get_args()
    # check binary
    if args.gui:
        sumoBinary = checkBinary('sumo-gui')
    else:
        sumoBinary = checkBinary('sumo')

    if args.scenario=='motorway':
        print('Motorway scenario selected')
        #scenario_folder = 'C:/Users/mnity/Desktop/quick_adap_to_incidents/Motorway'
        scenario_folder = '/home/manity/Quick_adap/quick_adap_to_incidents/Motorway'
    elif args.scenario=='national':
        print('National scenario selected')
        #scenario_folder = 'C:/Users/mnity/Desktop/quick_adap_to_incidents/National'
        scenario_folder = '/home/manity/Quick_adap/quick_adap_to_incidents/National'
    elif args.scenario=='urban':
        print('Urban scenario selected')
        #scenario_folder = 'C:/Users/mnity/Desktop/quick_adap_to_incidents/Urban'
        scenario_folder = '/home/manity/Quick_adap/quick_adap_to_incidents/Urban'
    elif args.scenario=='experiment':
        print('Experiment scenario')
        #scenario_folder = 'C:/Users/mnity/Desktop/quick_adap_to_incidents/Experiment'
        scenario_folder = '/home/manity/Quick_adap/quick_adap_to_incidents/Experiment'
    else:
        assert 'Please select scenario with --scenario'

   
    print(f"Running {args.n_runs} simulation of {args.scenario} with start time {args.begin} and end time {args.end}")

    
    #scenario_path1 = setup_run(scenario_folder=scenario_folder, edge_filename=f'{args.edge_filename}', run_num=1)
    #sumoCmd1 = [sumoBinary, "-c", scenario_path1, "--begin", f"{args.begin}", "--end", f"{args.end}", "--start", "1", "--quit-on-end", "1"]
    #run(sumoCmd=sumoCmd1, start_step=args.begin, end_step=args.end)
    
    
    
    
    
    #Scenario 1
    scenario_path1 = setup_run(scenario_folder=scenario_folder, edge_filename=f'{args.edge_filename}', run_num=1)
    sumoCmd1 = [sumoBinary, "-c", scenario_path1, "--begin", f"{args.begin}", "--end", f"{args.end}", "--start", "1", "--quit-on-end", "1"]
    p1 = Process(target=run, args=(sumoCmd1, args.begin, args.end))
    
    #Scenario 2 
    scenario_path2 = setup_run(scenario_folder=scenario_folder, edge_filename=f'{args.edge_filename}', run_num=2)
    sumoCmd2 = [sumoBinary, "-c", scenario_path2, "--begin", f"{args.begin}", "--end", f"{args.end}", "--start", "1", "--quit-on-end", "1"]
    p2 = Process(target=run, args=(sumoCmd2, args.begin, args.end))

    p1.start()
    p2.start()
    p1.join()
    p2.join()