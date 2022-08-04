#TODO Implement naming for results for parallel runs

import os
from shutil import ExecError
import sys
import argparse
from sumolib import checkBinary
from traitlets import default # For some reason this import fixes problems with importing libsumo.
import libsumo as traci
from time import time
from multiprocessing import Process

from incident_utils_libsumo import block_lanes, IncidentSettings
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
    arg_parser.add_argument("--n_random_incidents", type=int, default=0, help="The number of random incidents to simulate")
    arg_parser.add_argument("--n_non_incidents", type=int, default=0, help="The number of simulations without incident to run") #TODO implement
    arg_parser.add_argument("--incidents_settings_file", type=str, default=None, help="Path to the incident settings file") #TODO implement
    arg_parser.add_argument("--do_counterfactual", action='store_true', default=False, help="For any incident run the counterfactual of no incident")
    arg_parser.add_argument("--trip_info", action="store_true", default=False, help="Save information of all trips.")
    arg_parser.add_argument("--verbose", action="store_true", default=False, help="Save error and message log of SUMO warnings and errors")
    args = arg_parser.parse_args()

    if args.n_random_incidents == 0 and args.n_non_incidents == 0 and args.incidents_settings_file is None:
        raise Exception("Please set either number of random or non incidents or use a incidents settings file")

    if (args.n_random_incidents == 0) + (args.n_non_incidents == 0) + (args.incidents_settings_file is None) != 2:
        raise Exception("Please ONLY set either number of random or non incidents or use a incidents settings file")
    return args

# contrains Traci control loop
def run(simulation_settings, start_time, end_time, incident_settings):
    start_wtime = time()
    sim_time = start_time
    step = start_time * 2
   
    traci.start(simulation_settings['sumoCmd'])

    incident_settings.random()
    incident_settings.save_incident_information(simulation_settings['simulation_folder'])

    while traci.simulation.getMinExpectedNumber() > 0 and traci.simulation.getTime() <= end_time:
        block_lanes(incident_settings, step)

        #print(f'step {step}, my time {sim_time}, true sim time {traci.simulation.getTime()}')

        traci.simulationStep()
        
        step+=1
        sim_time+=0.5

    traci.close()
    sys.stdout.flush()
    end_wtime = time()
    print(f'finished in {end_wtime - start_wtime}')


# main entry point
if __name__ == "__main__":
    args = get_args()
    
    if args.n_random_incidents > 0:
        print(f"Running {args.n_random_incidents} simulations of scenario '{args.scenario}' with random incidents")
        incident_settings = []
        for i in range(args.n_random_incidents):
            incident_settings.append(IncidentSettings(run_num=i, is_random=True))
            # TODO has to be able to do no incident as well        
        n_runs = args.n_random_incidents
    if args.n_non_incidents > 0:
        print(f"Running {args.n_non_incidents} simulations of scenario '{args.scenario}' with no incidents")
    if args.incidents_settings_file is not None:
        print(f"Running simulations of scenario '{args.scenario}' using incidents in {args.incidents_settings_file}")

    scenario_folder = f'/home/manity/Quick_adap/quick_adap_to_incidents/{args.scenario}'


    simulation_settings = []
    processes = []

    #TODO setup random generation of incident times here as it is needed for selection of time, so I can make more simulations in the time
    #TODO setup the needed framework for different incidents in simulation here.
    #TODO       The idea is generate them above this and input a list of incidents (or just incident? yea to start) as input to run.
    # Create and start all sims
    for run_num in range(0, n_runs):
        simulation_settings.append(
            setup_run(scenario_folder=scenario_folder,
                      simulation_name=args.simulation_name,
                      run_num=run_num,
                      begin=args.begin,
                      end=args.end,
                      trip_info=args.trip_info,
                      verbose=args.verbose))
        
        processes.append(
            Process(target=run,
                    args=(simulation_settings[run_num], args.begin, args.end, incident_settings[run_num])))

        processes[run_num].start()

    # Wait for all sims to terminate
    for process in processes:
        process.join()

    cleanup_temp_files(scenario_folder=scenario_folder)