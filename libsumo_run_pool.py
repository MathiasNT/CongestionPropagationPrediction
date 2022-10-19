import os
from shutil import ExecError
import sys
import argparse
from sumolib import checkBinary # For some reason this import fixes problems with importing libsumo.
import libsumo as traci
from time import time
from multiprocessing import Pool
import json
import numpy as np

from util_folder.simulation_utils.incident_utils import IncidentSettings, SUMOIncident, create_counterfactual
from util_folder.simulation_utils.setup_utils import setup_counterfactual_sim, setup_incident_sim, cleanup_temp_files
from util_folder.simulation_utils.file_utils import xml2csv_file
from util_folder.preprocess_utils import infer_incident_data

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare encironment varialbe 'SUMO_HOME'")


def get_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--scenario", choices=['motorway', 'national', 'urban', 'experiment'], help='Which scenatio to run.', required=True)

    arg_parser.add_argument("--begin", type=int, default=0, help="Start time of the simulator.")
    arg_parser.add_argument("--end", type=int, default=86400, help="End time of the simulator.")
    arg_parser.add_argument("--incident_only", action='store_true', default=False, help="If true only simulates the time around the congestion")

    arg_parser.add_argument("--simulation_name", type=str, default="edgedata", help="The name of the simulation run. Will be name of numbere results folder.")

    arg_parser.add_argument("--n_random_incidents", type=int, default=0, help="The number of random incidents to simulate")
    arg_parser.add_argument("--n_non_incidents", type=int, default=0, help="The number of simulations without incident to run")
    arg_parser.add_argument("--incidents_settings_file", type=str, default=None, help="Path to the incident settings file")

    arg_parser.add_argument("--do_counterfactuals", action='store_true', default=False, help="For any incident run the counterfactual of no incident")

    arg_parser.add_argument("--trip_info", action="store_true", default=False, help="Save information of all trips.")

    arg_parser.add_argument("--verbose", action="store_true", default=False, help="Save error and message log of SUMO warnings and errors")

    arg_parser.add_argument("--data_frequency", type=int, default=10, help="The time resolution of the output files")
    args = arg_parser.parse_args()

    if args.n_random_incidents == 0 and args.n_non_incidents == 0 and args.incidents_settings_file is None:
        raise Exception("Please set either number of random or non incidents or use a incidents settings file")

    if (args.n_random_incidents == 0) + (args.n_non_incidents == 0) + (args.incidents_settings_file is None) != 2:
        raise Exception("Please ONLY set either number of random or non incidents or use a incidents settings file")
    return args

def run(simulation_settings, start_time, end_time, incident_settings):
    #TODO this only redirect the print statements. SUMO warnings are a WIP, see https://github.com/eclipse/sumo/issues/10344
    old_stdout = sys.stdout
    if simulation_settings['counterfactual']:
        print(f"Starting counterfactual for {simulation_settings['simulation_folder'].split('/')[-1]}")
        sys.stdout = open(f"{simulation_settings['simulation_folder']}/log_counterfactual.out", 'w')
        sys.stderr = open(f"{simulation_settings['simulation_folder']}/log_counterfactual.err", 'w')
    else:
        print(f"Starting {simulation_settings['simulation_folder'].split('/')[-1]}")
        sys.stdout = open(f"{simulation_settings['simulation_folder']}/log.out", 'w')
        sys.stderr = open(f"{simulation_settings['simulation_folder']}/log.err", 'w')


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

        traci.simulationStep()
        
        step+=1
        sim_time+=0.5

    traci.close()
    sys.stdout.flush()
    end_wtime = time()

    sys.stdout = old_stdout
    if simulation_settings['counterfactual']:
        xml2csv_file(f'{simulation_settings["simulation_folder"]}/detectordata_counterfactual.xml')
        xml2csv_file(f'{simulation_settings["simulation_folder"]}/edgedata_counterfactual.xml')
        os.remove(f'{simulation_settings["simulation_folder"]}/detectordata_counterfactual.xml')
        os.remove(f'{simulation_settings["simulation_folder"]}/edgedata_counterfactual.xml')

        print(f"Finished counterfactual for {simulation_settings['simulation_folder'].split('/')[-1]} in {end_wtime - start_wtime}")
    else:
        xml2csv_file(f'{simulation_settings["simulation_folder"]}/detectordata.xml')
        xml2csv_file(f'{simulation_settings["simulation_folder"]}/edgedata.xml')
        os.remove(f'{simulation_settings["simulation_folder"]}/detectordata.xml')
        os.remove(f'{simulation_settings["simulation_folder"]}/edgedata.xml')

        print(f"Finished {simulation_settings['simulation_folder'].split('/')[-1]} in {end_wtime - start_wtime}")

    # TODO Implement the data preprocess step here to take advantage of the parallel here anyway
    # TODO Would need a larger refactor to make sure both simulations are finished before running this.
    #if simulation_settings['counterfactual']:
        #input_data, target_data, inci_data, counter_data, ind_to_edge = infer_incident_data(f'{incident_settings["simulation_folder"]}')
        #np.save(f'{simulation_settings["simulation_folder"]}/input_data.npy', input_data)
        #np.save(f'{simulation_settings["simulation_folder"]}/target_data.npy', target_data)
        #np.save(f'{simulation_settings["simulation_folder"]}/inci_data.npy', inci_data)
        #np.save(f'{simulation_settings["simulation_folder"]}/counter_data.npy', counter_data)
        #json.dump( ind_to_edge, open( "file_name.json", 'w' ) )
    #else:


# main entry point
if __name__ == "__main__":

    # Hardcoded values TODO check if they need to be fixed
    simulation_warmup_time = 1200 # 3600 1 hour
    simulation_run_time = 14400 # 14400 4 hours

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
        with open(args.incidents_settings_file, 'r') as f:
            incident_settings_dicts = json.load(f)
        print(f'Found settings{incident_settings_dicts}.')
        n_runs = len(incident_settings_dicts)
        for i in range(n_runs):
            incident_settings.append(IncidentSettings(run_num=i, is_random=False))
            incident_settings[i].load_incident_dict(incident_settings_dicts[i])

        if args.do_counterfactuals:
            counterfactual_settings = []
            for i in range(n_runs):
                counterfactual_settings.append(create_counterfactual(incident_settings[i]))                  

    scenario_folder = f'/home/manity/Quick_adap/quick_adap_to_incidents/Simulation_scenarios/{args.scenario}'

    sim_settings = []
    jobs = []
    counterfactual_sim_settings = []
    counterfactual_jobs = []

    # Create and start all sims
    for run_num in range(0, n_runs):

        if args.incident_only:
            simulation_start_time = (incident_settings[run_num].start_time - simulation_warmup_time)
            simulation_end_time = (simulation_start_time + simulation_run_time)
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
                      data_freq=args.data_frequency,
                      verbose=args.verbose
            )
        )
        
        jobs.append([sim_settings[run_num], simulation_start_time, simulation_end_time, incident_settings[run_num]])

        if args.do_counterfactuals:
            counterfactual_sim_settings.append(
                setup_counterfactual_sim(
                    scenario_folder=scenario_folder,
                    simulation_folder=sim_settings[run_num]['simulation_folder'],
                    run_num=run_num,
                    begin=simulation_start_time,
                    end=simulation_end_time,
                    trip_info=args.trip_info,
                    data_freq=args.data_frequency,
                    verbose=args.verbose
                )
            )

            jobs.append((counterfactual_sim_settings[run_num], simulation_start_time, simulation_end_time, counterfactual_settings[run_num]))
     
    # TODO figure out what can be done about how to avoid killing the master when a thread gets an error
    with Pool(os.cpu_count() - 4) as pool:
        print(f'Running {len(jobs)} simulations')
        if args.do_counterfactuals:
            print(f'with counterfactuals')
        pool.starmap(run, jobs)
    

    cleanup_temp_files(scenario_folder=scenario_folder)