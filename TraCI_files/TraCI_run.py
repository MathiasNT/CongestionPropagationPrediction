import os
import sys
import argparse
import numpy as np
from sumolib import checkBinary
import traci
import xml.etree.ElementTree as ET

from incident_utils import block_lanes

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare encironment varialbe 'SUMO_HOME'")


def get_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--nogui", action="store_true",
                            default=False, help="run the commanline version of sumo")
    arg_parser.add_argument("--scenario", choices=['motorway', 'national', 'urban', 'experiment'], help='Which scenatio to run.', required=True)
    arg_parser.add_argument("--begin", type=int, default=0, help="Start time of the simulator")
    arg_parser.add_argument("--end", type=int, default=86400, help="End time of the simulator")
    arg_parser.add_argument("--edge_file", type=str, default="edgedata.xml", help="The name of the edge data file.")
    args = arg_parser.parse_args()
    return args

def setup_run(scenario_folder, edge_file):
    xml_file = f'{scenario_folder}/Simulations/Base/edgedata.add.xml'
    xml_tree = ET.parse(xml_file)    
    xml_root = xml_tree.getroot()
    edge_data_settings = xml_root[0]
    edge_data_settings.set('file', f'{scenario_folder}/Results/{edge_file}')
    xml_tree.write(xml_file)
    print(f'Now saving results at {edge_data_settings.get("file")}')





# contrains Traci control loop
def run(start_step, end_step):
    step = start_step
    print(f'step {step}') 

    incidents = []    
   
    # 'full block' example with gradual release. Not sure if the simulation looks real but I don't know how much better we can get. However this here is as good as I expect QTIP would have been if not better
    #incidents = ['48290550_0_300_1100_1200','48290550_1_300_1100_1200','48290550_2_300_1100_1200','48290550_3_300_1100_1600']   
   
    
    
    
    while traci.simulation.getMinExpectedNumber() > 0 and traci.simulation.getTime() <= end_step:
        traci.simulationStep()
        
        incidents = block_lanes(incidents, step)

        step+=1

        

    traci.close()
    sys.stdout.flush()




# main entry point
if __name__ == "__main__":
    args = get_args()
    # check binary
    if args.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
    
    if args.scenario=='motorway':
        print('Motorway scenario selected')
        scenario_folder = 'C:/Users/mnity/Desktop/quick_adap_to_incidents/Motorway'
        scenario_path = f'{scenario_folder}/Simulations/Base/M50_simulation.sumo.cfg'
    elif args.scenario=='national':
        print('National scenario selected')
        scenario_folder = 'C:/Users/mnity/Desktop/quick_adap_to_incidents/National'
        scenario_path = f'{scenario_folder}/Simulations/Base/N7_simulation.sumo.cfg'
    elif args.scenario=='urban':
        print('Urban scenario selected')
        scenario_folder = 'C:/Users/mnity/Desktop/quick_adap_to_incidents/Urban'
        scenario_path = f'{scenario_folder}/Simulations/Base/DCC_simulation.sumo.cfg'
    elif args.scenario=='experiment':
        print('Experiment scenario')
        scenario_folder = 'C:/Users/mnity/Desktop/quick_adap_to_incidents/Experiment'
        scenario_path = f'{scenario_folder}/Simulations/Base/traci_exp.sumo.cfg'
    else:
        assert 'Please select scenario with --scenario'

    setup_run(scenario_folder=scenario_folder, edge_file=args.edge_file)

    print(f"Running simulation {args.scenario} with start time {args.begin} and end time {args.end}")
    # traci starts sumo as a subprocess and then this script connects and runs
    traci.start([sumoBinary, "-c", scenario_path, "--tripinfo-output", 'tripinfo.xml',  "--begin", f"{args.begin}", "--end", f"{args.end}"]) # "--start", "1", "--quit-on-end", "1", 
    run(start_step=args.begin, end_step=args.end)