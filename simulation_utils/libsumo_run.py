#TODO Implement naming for results for parallel runs

import os
import sys
import argparse
from sumolib import checkBinary # For some reason this import fixes problems with importing libsumo.
import libsumo as traci
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
                            default=False, help="run the gui version of sumo.")
    arg_parser.add_argument("--scenario", choices=['motorway', 'national', 'urban', 'experiment'], help='Which scenatio to run.', required=True)
    arg_parser.add_argument("--begin", type=int, default=0, help="Start time of the simulator.")
    arg_parser.add_argument("--end", type=int, default=86400, help="End time of the simulator.")
    arg_parser.add_argument("--simulation_name", type=str, default="edgedata", help="The name of the simulation run. Will be name of numbere results folder.")
    arg_parser.add_argument("--n_runs", type=int, default=1, help="The number of simulations to run in parallel.")
    arg_parser.add_argument("--trip_info", action="store_true", default=False, help="Save information of all trips.")
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

    scenario_folder = f'/home/manity/Quick_adap/quick_adap_to_incidents/{args.scenario}'
    print(f'{args.scenario} selected')
    
   
    print(f"Running {args.n_runs} simulations of scenario '{args.scenario}' with start time {args.begin} and end time {args.end}")

    simulation_settings = []
    processes = []

    # Create and start all sims
    for run_num in range(0, args.n_runs):
        simulation_settings.append(
            setup_run(scenario_folder=scenario_folder,
                      simulation_name=args.simulation_name,
                      run_num=run_num,
                      begin=args.begin,
                      end=args.end,
                      trip_info=args.trip_info))
        processes.append(
            Process(target=run,
                    args=(simulation_settings[run_num]['sumoCmd'], args.begin, args.end)))
        processes[run_num].start()

    # Wait for all sims to terminate
    for process in processes:
        process.join()

    cleanup_temp_files(scenario_folder=scenario_folder)