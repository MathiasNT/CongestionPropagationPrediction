#import traci

import libsumo as traci
import numpy as np
import json


def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()

def create_counterfactual(incident_settings):
    counterfactual = IncidentSettings(run_num=incident_settings.run_num)
    counterfactual.set_incident(start_time=incident_settings.start_time, duration=incident_settings.duration_time)
    return counterfactual

#TODO Think if we want to be able to specify the distributions to get the incidents from?
class IncidentSettings():
    def __init__(self, run_num, is_random=False):
        self.run_num = run_num
        self.is_incident = False
        self.is_random = is_random
        self.random_seed = np.random.randint(2**32 - 1)
        
        self.edge = None
        self.lanes = None
        self.pos = None 
        self.start_time = None
        self.start_step = None
        self.duration_time = None
        self.duration_steps = None

        if self.is_random:
            self.random_time()
            self.random_duration()

    def set_incident(self, edge=None, lanes=None, pos=None, start_time=None, duration=None, is_incident=False):
        # Fills out an incident with predefined values. Note that not all needs to be filled out if you want some of it random.
        self.is_incident = is_incident
        self.edge = edge
        self.lanes = lanes
        self.pos = pos
        self.start_time = start_time
        self.start_step = start_time * 2
        self.duration_time = duration
        self.duration_steps = duration * 2

    def random(self):
        # Fills out incident variables that are not set randomly

        np.random.seed(self.random_seed)

        if self.edge is None:
            self.random_edge()
        if self.lanes is None:
            self.random_lanes()
        if self.pos is None:
            self.random_pos() 
        if self.start_time is None:
            self.random_time()
        if self.duration_time is None:
            self.random_duration()

        self.is_incident = True

    def random_edge(self):
        # TODO could use lane.getMaxSpeed to select only highway edges
        object_list = traci.edge.getIDList() 
        edge_list = [edge for edge in object_list if not edge.startswith(':')] # Remove junctions
        self.edge = np.random.choice(edge_list)
    
    def random_lanes(self):
        n_lanes = traci.edge.getLaneNumber(self.edge)
        lane_names = np.arange(0, n_lanes)
        n_blocked_lanes = np.random.randint(1, n_lanes + 1)
        blocked_lanes = np.random.choice(lane_names, n_blocked_lanes, replace=False)
        self.lanes = blocked_lanes.tolist()
    
    def random_pos(self):
        edge_length = traci.lane.getLength(f'{self.edge}_0')
        self.pos = np.random.uniform(10, edge_length - 10) # Leave room at the end of edges to avoid bugs
        return

    def random_time(self):
        self.start_time = np.rint(np.random.uniform(0, 84000)).astype(int)
        self.start_step = self.start_time * 2
        return

    def random_duration(self):
        self.duration_time = np.rint(np.random.normal(1200, 100)).astype(int) # This could be made more precise by rounding to to 0.5, if SUMO can take that time
        self.duration_steps = self.duration_time * 2
        return

    def save_incident_information(self, folder_path):
        if self.is_incident:
            json_str = json.dumps(self.__dict__, default=np_encoder)
            file = open(f'{folder_path}/incident_settings.json', 'w+') 
            file.write(json_str)
            #TODO Figure out if I want more information
        return


def block_lanes(incident_settings, step):
    """Blocks lanes at the specified position and timesteps

    Args:
        lane_pos_times (list of strings): A list of strings with format "{edge}_{lane}_{pos}_{time}_{duration}" that defines the blockages
        step (int): Simulation step for time tracking
        duration (int, optional): Duration of incidents. Defaults to 1200.

    Returns:
        list of strings: A list of strings for blockages that have not started nor finished yet. 
    """

    edge = incident_settings.edge
    pos = incident_settings.pos
    start_step = incident_settings.start_step
    duration = incident_settings.duration_time # TODO double check if i want to run on time
    run_num = incident_settings.run_num

    #TODO Update this for the new incident framework where a single incident have multiple blocked lanes
    if step==start_step: # Create block if time, check for collisions first.
        for lane in incident_settings.lanes:
            incident_veh_id = f'incident_veh_{edge}_{lane}_{pos}'
            incident_route_id = f"incident_route_{edge}_{lane}"
            #Check for vehicles on the incident lane and remove them if necessary
            on_edge = traci.lane.getLastStepVehicleIDs(f"{edge}_{lane}")
            if on_edge:
                veh_pos_on_edge = []
                for veh in on_edge:
                    veh_pos_on_edge.append(traci.vehicle.getLanePosition(veh))
                print(f"step: {step}, lane: {edge}_{lane}, on edge: {on_edge}, pos: {veh_pos_on_edge}, incident pos: {pos}")
                print((np.abs(veh_pos_on_edge - np.array(pos))))
                if np.min(np.abs(veh_pos_on_edge - np.array(pos))) < 7:          # Checking the closest vehicle should be good enough
                    prob_veh = on_edge[np.argmin(np.abs(veh_pos_on_edge - np.array(pos)))]
                    print(f"{prob_veh} is too close, removing it")
                    traci.vehicle.remove(prob_veh)

            # Create the incident blocking the lane
            print(f"run {run_num} step {step} creating block {edge}_{lane}_{pos}_{start_step}")
            traci.route.add(incident_route_id, [edge])
            traci.vehicle.add(vehID=incident_veh_id, routeID=incident_route_id)
            traci.vehicle.moveTo(vehID=incident_veh_id, laneID=f'{edge}_{lane}', position=int(pos)) # Note annoying difference in position or pos between libsum and traci
            traci.vehicle.setSpeed(vehID=incident_veh_id, speed=0)
            traci.vehicle.setLaneChangeMode(vehID=incident_veh_id, laneChangeMode=0) # Again an annoying difference between libSumo and traci
    
    elif step > start_step and (step-start_step)%100==0 and (step-start_step) < duration: # Starts moving block to avoid time out
        for lane in incident_settings.lanes:
            incident_veh_id = f'incident_veh_{edge}_{lane}_{pos}'
            incident_route_id = f"incident_route_{edge}_{lane}_{pos}"
            traci.vehicle.setSpeed(vehID=incident_veh_id, speed=0.101)

    elif step > start_step and (step-start_step)%102==0 and (step-start_step) < duration: # Stops moving block
        for lane in incident_settings.lanes:
            incident_veh_id = f'incident_veh_{edge}_{lane}_{pos}'
            incident_route_id = f"incident_route_{edge}_{lane}_{pos}"
            traci.vehicle.setSpeed(vehID=incident_veh_id, speed=0)

    elif step==(start_step+duration): # Removes block
        for lane in incident_settings.lanes:
            incident_veh_id = f'incident_veh_{edge}_{lane}_{pos}'
            print(f"run {run_num} step {step} removing block {lane}_{pos}_{start_step}")
            traci.vehicle.remove(vehID=incident_veh_id)
    return 