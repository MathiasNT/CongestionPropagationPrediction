
import os
from shutil import ExecError
import sys
import argparse
from sumolib import checkBinary # For some reason this import fixes problems with importing libsumo.
import traci
from time import time
from multiprocessing import Process

from incident_utils import IncidentSettings, SUMOIncident 
from setup_utils import setup_gui_sim, cleanup_temp_files

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
    args = arg_parser.parse_args()

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


    while traci.simulation.getMinExpectedNumber() > 0 and traci.simulation.getTime() <= end_time:
        if incident_settings.is_incident:
           sumo_incident.sim_incident(step, reroute=True) 

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
    simulation_warmup_time = 100 # 3600 is 1 hour
    simulation_congestion_time = 14400 # 4 hours

    ############ CHANGE THIS BLOCK FOR DIFFERENT SCENARIOS ###########

    # 'full block' example with gradual release. Not sure if the simulation looks real but I don't know how much better we can get. However this here is as good as I expect QTIP would have been if not better
    #incidents = ['48290550_0_300_11000_1200','48290550_1_300_11000_1200','48290550_2_300_11000_1200','48290550_3_300_11000_1600']   
    
    #python .\traci_run.py --scenario experiment --gui
    #incidents = [':J0_0_10_500_1200']
    #incidents = ['E3_0_10_500_1200', 'E3_1_10_500_1200']
    #incidents = ['E1_2_10_500_1200', 'E1_1_10_500_1200'] # THis incident but sim didn't crash?????? at 80
    #incidents = ['E1_0_10_500_1200', 'E1_1_10_500_1200'] # THis incident but sim didn't crash?????? at 80
    #incidents = ['E1_2_10_500_1200', 'E1_1_10_500_1200', 'E1_0_10_500_1200'] # THis incident but sim didn't crash?????? at 80
    
    
    #python .\traci_run.py --scenario motorway --gui --begin 50000
    #incidents = ['48290550_0_300_50500_1200','48290550_1_300_50500_1200','48290550_2_300_50500_1200','4829550_3_300_50500_1600']   
    
    # python .\traci_run.py --scenario urban --gui --begin 50000
    #incidents = ['360313821_0_50_50500_1200','360313821_1_50_50500_1200','360313821_2_50_50500_1200'] # Shows the need for rerouting
    scenario = 'motorway'
    scenario_folder = f'C:/Users/mnity/Desktop/quick_adap_to_incidents/Simulation_scenarios/{scenario}'
    
    incident_settings = IncidentSettings(run_num=0)

    ## Experiment 
    #incident_settings.set_incident(
        #edge='E1',
        #lanes=[0,1],
        #pos=80,
        #start_time=100,
        #duration=1140,
        #is_incident=True,
    #)

    ## Motorway
    #incident_settings.set_incident(
        #edge='360361373',
        #lanes=[0,1,2],
        #pos=358.1722741760613,
        #start_time=67208,
        #duration=1140,
        #is_incident=True,
    #)
    
    incident_settings.set_incident(
        edge='360361373-AddedOnRampEdge',
        lanes=[0,1, 2, 3],
        pos=21.82098870155751, 
        start_time=86399,
        duration=1224,
        is_incident=True,
    )

    #incident_settings.set_incident(
        #edge="4414080#0.187",
        #lanes=[0,1],
        #pos=63.18027276725517,
        #start_time=64049,
        #duration=1132,
        #is_incident=True,
    #)
    
    ## National
    #incident_settings.set_incident(
        #edge='123961236#1',
        #lanes=[0,1,2],
        #pos=358.1722741760613,
        #start_time=67208,
        #duration=1140,
        #is_incident=True,
    #)


    ## Urban
    #incident_settings.set_incident(
        #edge='14327272#0',
        #lanes=[0,1],
        #pos=70,
        #start_time=67208,
        #duration=1140,
        #is_incident=True,
    #)

    ###########################

    #simulation_start_time = 0
    #simulation_end_time = 84000

    simulation_start_time = (incident_settings.start_time - simulation_warmup_time)
    simulation_end_time = (simulation_start_time + incident_settings.duration_time + simulation_congestion_time)

    sim_settings = setup_gui_sim(scenario_folder=scenario_folder,
                      begin=simulation_start_time,
                      end=simulation_end_time,
                    )

    run(simulation_settings=sim_settings, start_time=simulation_start_time, end_time=simulation_end_time, incident_settings=incident_settings)

    cleanup_temp_files(scenario_folder=scenario_folder)