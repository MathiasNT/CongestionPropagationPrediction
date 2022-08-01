#TODO Implement naming for results for parallel runs

import os
import sys
import argparse
import glob
import numpy as np
from sumolib import checkBinary
import libsumo as traci
import xml.etree.ElementTree as ET
from time import time
from multiprocessing import Process

from incident_utils_libsumo import block_lanes
from setup_utils import setup_run, cleanup_temp_files

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

# contrains Traci control loop
def run(sumoCmd, start_step, end_step):
    start_time = time()
    step = start_step
    incidents = []    
   
    # 'full block' example with gradual release. Not sure if the simulation looks real but I don't know how much better we can get. However this here is as good as I expect QTIP would have been if not better
    #incidents = ['48290550_0_300_1100_1200','48290550_1_300_1100_1200','48290550_2_300_1100_1200','48290550_3_300_1100_1600']   
    
    #incidents = ['E3_0_10_500_1200']

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

   
    print(f"Running {args.n_runs} simulations of scenario '{args.scenario}' with start time {args.begin} and end time {args.end}")

    scenario_paths = []
    sumoCmds = []
    processes = []

    # Create and start all sims
    for run_num in range(0, args.n_runs):
        sumoCmds.append(setup_run(scenario_folder=scenario_folder, edge_filename=args.edge_filename, run_num=run_num, begin=args.begin, end=args.end))
        processes.append(Process(target=run, args=(sumoCmds[run_num], args.begin, args.end)))
        processes[run_num].start()

    # Wait for all sims to terminate
    for process in processes:
        process.join()

    cleanup_temp_files(scenario_folder=scenario_folder)