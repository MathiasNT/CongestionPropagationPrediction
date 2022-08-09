#TODO Implement naming for results for parallel runs

import os
from shutil import ExecError
import sys
import argparse
from sumolib import checkBinary # For some reason this import fixes problems with importing libsumo.
import libsumo as traci
from time import time
from multiprocessing import Process

from incident_utils_libsumo import IncidentSettings, SUMOIncident, create_counterfactual
from setup_utils import setup_counterfactual_sim, setup_incident_sim, cleanup_temp_files

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
    arg_parser.add_argument("--incident_only", action='store_true', default=False, help="If true only simulates the time around the congestion")
    arg_parser.add_argument("--simulation_name", type=str, default="edgedata", help="The name of the simulation run. Will be name of numbere results folder.")
    arg_parser.add_argument("--n_random_incidents", type=int, default=0, help="The number of random incidents to simulate")
    arg_parser.add_argument("--n_non_incidents", type=int, default=0, help="The number of simulations without incident to run") #TODO implement
    arg_parser.add_argument("--incidents_settings_file", type=str, default=None, help="Path to the incident settings file") #TODO implement
    arg_parser.add_argument("--do_counterfactuals", action='store_true', default=False, help="For any incident run the counterfactual of no incident")
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

    if incident_settings.is_random:
        incident_settings.random()

    if incident_settings.is_incident:
        sumo_incident = SUMOIncident(incident_settings=incident_settings)
        sumo_incident.traci_init(simulation_settings['scenario_folder'])

    incident_settings.save_incident_information(simulation_settings['simulation_folder'])

    while traci.simulation.getMinExpectedNumber() > 0 and traci.simulation.getTime() <= end_time:
        if incident_settings.is_incident:
           sumo_incident.sim_incident(step)

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

    # Hardcoded values TODO check if they need to be fixed
    simulation_warmup_time = 3600 # 1 hour
    simulation_congestion_time = 14400 # 4 hours

    args = get_args()
    
    incident_settings = []
    if args.n_random_incidents > 0:
        print(f"Running {args.n_random_incidents} simulations of scenario '{args.scenario}' with random incidents")
        for i in range(args.n_random_incidents):
            incident_settings.append(IncidentSettings(run_num=i, is_random=True))
        n_runs = args.n_random_incidents

        if args.do_counterfactuals:
            counterfactual_settings = []
            for i in range(args.n_random_incidents):
                counterfactual_settings.append(create_counterfactual(incident_settings[i]))                  

    elif args.n_non_incidents > 0:
        print(f"Running {args.n_non_incidents} simulations of scenario '{args.scenario}' with no incidents")
        for i in range(args.n_non_incidents):
            incident_settings.append(IncidentSettings(run_num=i))
        n_runs = args.n_non_incidents

        if args.do_counterfactuals:
            raise Exception('No reason to do counterfactuals without incidents')

    elif args.incidents_settings_file is not None:
        print(f"Running simulations of scenario '{args.scenario}' using incidents in {args.incidents_settings_file}")
        # TODO has to be able to do incident file as well        

    scenario_folder = f'/home/manity/Quick_adap/quick_adap_to_incidents/{args.scenario}'


    sim_settings = []
    processes = []
    counterfactual_sim_settings = []
    counterfactual_processes = []

    # Create and start all sims
    for run_num in range(0, n_runs):

        if args.incident_only:
            simulation_start_time = (incident_settings[run_num].start_time - simulation_warmup_time)
            simulation_end_time = (simulation_start_time + incident_settings[run_num].duration_time + simulation_congestion_time)
        else:
            simulation_start_time =  args.begin
            simulation_end_time = args.end

        sim_settings.append(
            setup_incident_sim(scenario_folder=scenario_folder,
                      simulation_name=args.simulation_name,
                      run_num=run_num,
                      begin=simulation_start_time,
                      end=simulation_end_time,
                      trip_info=args.trip_info,
                      verbose=args.verbose
            )
        )
        
        processes.append(
            Process(target=run,
                    args=(sim_settings[run_num], simulation_start_time, simulation_end_time, incident_settings[run_num])
            )
        )

        print(f'Starting run {run_num} at time {simulation_start_time}. Running until {simulation_end_time}')
        processes[run_num].start()

        if args.do_counterfactuals:
            counterfactual_sim_settings.append(
                setup_counterfactual_sim(
                    scenario_folder=scenario_folder,
                    simulation_folder=sim_settings[run_num]['simulation_folder'],
                    run_num=run_num,
                    begin=simulation_start_time,
                    end=simulation_end_time,
                    trip_info=args.trip_info,
                    verbose=args.verbose
                )
            )

            counterfactual_processes.append(
                Process(target=run,
                        args=(counterfactual_sim_settings[run_num], simulation_start_time, simulation_end_time, counterfactual_settings[run_num])
                )
            )

            print(f'Starting counterfactual {run_num} at time {simulation_start_time}. Running until {simulation_end_time}')
            counterfactual_processes[run_num].start()
            

    # Wait for all sims to terminate
    for process in processes:
        process.join()

    if args.do_counterfactuals:
        for process in counterfactual_processes:
            process.join()

    cleanup_temp_files(scenario_folder=scenario_folder)