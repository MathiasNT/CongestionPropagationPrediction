#TODO see todos from libsumo version
import numpy as np
import json
import sumolib
import os
import sys

if 'OS' in os.environ.keys():
    import traci
    con_lib = 'traci'
else:
    import libsumo as traci
    con_lib = 'libsumo'

def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()

def create_counterfactual(incident_settings):
    counterfactual = IncidentSettings(run_num=incident_settings.run_num)
    counterfactual.set_incident(start_time=incident_settings.start_time, duration=incident_settings.duration_time)
    return counterfactual

class IncidentSettings():
    def __init__(self, run_num, is_random=False):
        # Currently hardcoded values
        self.slow_zone = 70 
        self.lc_zone = 20
        self.lc_prob_zone = 170
        self.slow_zone_speed = 5 # 13.8 is 50 km/h should work for highway situations.


        self.run_num = run_num
        self.is_incident = False
        self.is_random = is_random
        self.random_seed = np.random.randint(2**31 - 1)
        
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
        object_list = traci.edge.getIDList() 
        edge_list = [edge for edge in object_list if not edge.startswith(':')] # Remove junctions
        edge_list = [edge for edge in edge_list if len(traci.lane.getLinks(f'{edge}_0')) != 0] # Remove last edges
        self.edge = np.random.choice(edge_list)
    
    def random_lanes(self):
        n_lanes = traci.edge.getLaneNumber(self.edge)
        lane_names = np.arange(0, n_lanes)
        n_blocked_lanes = np.random.randint(1, n_lanes + 1)

        if np.random.randint(0,2):
            # Do block from low
            blocked_lanes = lane_names[:n_blocked_lanes]
        else:
            blocked_lanes = lane_names[-n_blocked_lanes:]

        # blocked_lanes = np.random.choice(lane_names, n_blocked_lanes, replace=False) for fully random lanes
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
        return

    def load_incident_dict(self, dict):
        for k, v in dict.items():
            if k not in ['run_num', 'is_random']:
                setattr(self, k, v)


class SUMOIncident():
    def __init__(self, incident_settings):
        self.incident_edge = incident_settings.edge
        self.lanes = incident_settings.lanes
        self.pos = incident_settings.pos
        self.start_step = incident_settings.start_step
        self.duration_steps = incident_settings.duration_steps 
        self.run_num = incident_settings.run_num

        self.slow_zone = incident_settings.slow_zone
        self.lc_zone = incident_settings.lc_zone
        self.lc_prob_zone = incident_settings.lc_prob_zone
        self.slow_zone_speed = incident_settings.slow_zone_speed
        
        

        # These are set in the traci init
        self.incident_edge_lanes = None        
        self.free_lanes = None
        self.upstream_edges = None
        self.upstream_edges_length = None 
        self.upstream_edges_n_lanes = None
        self.upstream_slow_zone = None
        self.upstream_lc_zone = None
        self.upstream_lc_prob_zone = None

        self.last_rerouted_cars = []
        return

    def sim_incident(self, step, reroute=False):
        self.slow_and_change_lanes(step)
        if self.pos < np.max([self.slow_zone, self.lc_prob_zone]):
            self.slow_and_change_upstream(step)
        self.block_lanes(step)        
        if reroute:
            self.reroute_upstream(step)

    def block_lanes(self, step):
        # Logic for maintaining blocked lanes
        if step==self.start_step: # Create block if time, check for collisions first.
            for lane in self.lanes:
                incident_veh_id = f'incident_veh_{self.incident_edge}_{lane}_{self.pos}'
                incident_route_id = f"incident_route_{self.incident_edge}_{lane}"
                #Check for vehicles on the incident lane and remove them if necessary
                on_edge = traci.lane.getLastStepVehicleIDs(f"{self.incident_edge}_{lane}")
                if on_edge:
                    veh_pos_on_edge = []
                    for veh in on_edge:
                        veh_pos_on_edge.append(traci.vehicle.getLanePosition(veh))
                    print(f"step: {step}, lane: {self.incident_edge}_{lane}, on edge: {on_edge}, pos: {veh_pos_on_edge}, incident pos: {self.pos}")
                    print((np.abs(veh_pos_on_edge - np.array(self.pos))))
                    if np.min(np.abs(veh_pos_on_edge - np.array(self.pos))) < 7:          # Checking the closest vehicle should be good enough
                        prob_veh = on_edge[np.argmin(np.abs(veh_pos_on_edge - np.array(self.pos)))]
                        print(f"{prob_veh} is too close, removing it")
                        traci.vehicle.remove(prob_veh)

                # Create the incident blocking the lane
                print(f"run {self.run_num} step {step} creating block {self.incident_edge}_{lane}_{self.pos}_{self.start_step}")
                traci.route.add(incident_route_id, [self.incident_edge, self.downstream_edges[0]])
                traci.vehicle.add(vehID=incident_veh_id, routeID=incident_route_id, typeID='IC')
                
                if con_lib == 'traci':
                    traci.vehicle.moveTo(vehID=incident_veh_id, laneID=f'{self.incident_edge}_{lane}', pos=int(self.pos))
                elif con_lib == 'libsumo':
                    traci.vehicle.moveTo(vehID=incident_veh_id, laneID=f'{self.incident_edge}_{lane}', position=int(self.pos))

                traci.vehicle.setSpeed(vehID=incident_veh_id, speed=0)
                if con_lib == 'traci':
                    traci.vehicle.setLaneChangeMode(vehID=incident_veh_id, lcm=0) # Again an annoying difference between libSumo and traci
                elif con_lib == 'libsumo':
                    traci.vehicle.setLaneChangeMode(vehID=incident_veh_id, laneChangeMode=0) # Again an annoying difference between libSumo and traci
        
        elif step > self.start_step and (step-self.start_step)%100==0 and (step-self.start_step) < self.duration_steps: # Starts moving block to avoid time out
            for lane in self.lanes:
                incident_veh_id = f'incident_veh_{self.incident_edge}_{lane}_{self.pos}'
                incident_route_id = f"incident_route_{self.incident_edge}_{lane}_{self.pos}"
                traci.vehicle.setSpeed(vehID=incident_veh_id, speed=0.101)

        elif step > self.start_step and (step-self.start_step)%102==0 and (step-self.start_step) < self.duration_steps: # Stops moving block
            for lane in self.lanes:
                incident_veh_id = f'incident_veh_{self.incident_edge}_{lane}_{self.pos}'
                incident_route_id = f"incident_route_{self.incident_edge}_{lane}_{self.pos}"
                traci.vehicle.setSpeed(vehID=incident_veh_id, speed=0)

        elif step==(self.start_step+self.duration_steps): # Removes block
            for lane in self.lanes:
                incident_veh_id = f'incident_veh_{self.incident_edge}_{lane}_{self.pos}'
                print(f"run {self.run_num} step {step} removing block {lane}_{self.pos}_{self.start_step}")
                traci.vehicle.remove(vehID=incident_veh_id)
                self.remove_speed_limit()
        return

    def slow_and_change_lanes(self, step):
        # Logic for slowing down traffic around incident
        if step >= self.start_step and step <= (self.start_step + self.duration_steps):
            for lane in self.incident_edge_lanes:
                on_edge = traci.lane.getLastStepVehicleIDs(f"{self.incident_edge}_{lane}")
                cars_on_edge = [car for car in on_edge if 'incident' not in car]
                if cars_on_edge:
                    for veh in cars_on_edge:
                        veh_pos_on_edge = traci.vehicle.getLanePosition(veh)
                        dist_to_incident =  self.pos - veh_pos_on_edge
                        if 0 < dist_to_incident < self.slow_zone:
                            traci.vehicle.setMaxSpeed(veh, self.slow_zone_speed)
                        
                        if self.free_lanes:
                            dist_to_free_lane = np.min(np.abs(lane - np.array(self.free_lanes)))
                            if 0 < dist_to_incident < self.lc_prob_zone * dist_to_free_lane:
                                target_lane = min(self.free_lanes, key=lambda x:abs(x-traci.vehicle.getLaneIndex(veh)))
                                frac_of_prob_zone_left = dist_to_incident / (self.lc_prob_zone - self.lc_zone)
                                if np.random.uniform() > frac_of_prob_zone_left:
                                    traci.vehicle.changeLane(veh, target_lane, 0.1)
                            
                        if veh_pos_on_edge > self.pos:
                            veh_class = traci.vehicle.getVehicleClass(veh)
                            if veh_class == 'passenger':
                                traci.vehicle.setMaxSpeed(veh, 55.55) # 55.55 is default SUMO settings, so should be ok
                            elif veh_class == 'truck':
                                traci.vehicle.setMaxSpeed(veh, 36.11) # 55.55 is default SUMO settings, so should be ok

    def slow_and_change_upstream(self, step):
        for edge in self.upstream_edges:
            if step >= self.start_step and step <= (self.start_step + self.duration_steps):
                for lane in range(self.upstream_edges_n_lanes_dict[edge]):
                    on_edge = traci.lane.getLastStepVehicleIDs(f"{edge}_{lane}")
                    cars_on_edge = [car for car in on_edge if 'incident' not in car]
                    if cars_on_edge:
                        for veh in cars_on_edge:
                            veh_pos_on_edge = traci.vehicle.getLanePosition(veh)
                            dist_to_edge_end =  self.upstream_edges_length_dict[edge] - veh_pos_on_edge
                            if dist_to_edge_end < self.upstream_slow_zone:
                                traci.vehicle.setMaxSpeed(veh, self.slow_zone_speed)

                            if self.free_lanes: 
                                if len(self.incident_edge_lanes) == self.upstream_edges_n_lanes_dict[edge]: # Note this is only implemented if the junction is 1-to-1
                                    dist_to_free_lane = np.min(np.abs(lane - np.array(self.free_lanes)))
                                    if 0 < dist_to_edge_end < self.upstream_lc_prob_zone * dist_to_free_lane:
                                        target_lane = min(self.free_lanes, key=lambda x:abs(x-traci.vehicle.getLaneIndex(veh)))
                                        frac_of_prob_zone_left = (dist_to_edge_end - self.pos) / (self.lc_prob_zone - self.lc_zone)
                                        if np.random.uniform() > frac_of_prob_zone_left:
                                            traci.vehicle.changeLane(veh, target_lane, 0.1)

    def remove_speed_limit(self):
        for lane in self.incident_edge_lanes:
            on_edge = traci.lane.getLastStepVehicleIDs(f"{self.incident_edge}_{lane}")
            cars_on_edge = [car for car in on_edge if 'incident' not in car]
            for veh in cars_on_edge:
                veh_class = traci.vehicle.getVehicleClass(veh)
                if veh_class == 'passenger':
                    traci.vehicle.setMaxSpeed(veh, 55.55) # 55.55 is default SUMO settings, so should be ok
                elif veh_class == 'truck':
                    traci.vehicle.setMaxSpeed(veh, 36.11) # 55.55 is default SUMO settings, so should be ok
        
        if self.pos < np.max([self.slow_zone, self.lc_prob_zone]):
            for edge in self.upstream_edges:
                for lane in range(self.upstream_edges_n_lanes_dict[edge]):
                    on_edge = traci.lane.getLastStepVehicleIDs(f"{edge}_{lane}")
                    cars_on_edge = [car for car in on_edge if 'incident' not in car]
                    for veh in cars_on_edge:
                        veh_class = traci.vehicle.getVehicleClass(veh)
                        if veh_class == 'passenger':
                            traci.vehicle.setMaxSpeed(veh, 55.55) # 55.55 is default SUMO settings, so should be ok
                        elif veh_class == 'truck':
                            traci.vehicle.setMaxSpeed(veh, 36.11) # 55.55 is default SUMO settings, so should be ok

    def reroute_upstream(self, step):
        if step >= self.start_step and step <= (self.start_step + self.duration_steps):
            upstream_cars = []
            for edge in self.upstream_edges:
                for lane in range(self.upstream_edges_n_lanes_dict[edge]):
                    on_edge = traci.lane.getLastStepVehicleIDs(f"{edge}_{lane}")
                    if on_edge:
                        upstream_cars = upstream_cars + list(on_edge)
            
            for car in upstream_cars:
                traci.vehicle.rerouteTraveltime(car, currentTravelTimes=True)

            fully_rerouted_cars = list(set(self.last_rerouted_cars) - set(upstream_cars))
            self.remove_speed_limit_reroute(fully_rerouted_cars)
            
            self.last_rerouted_cars = upstream_cars

    def remove_speed_limit_reroute(self, rerouted_cars):
        for veh in rerouted_cars:
            veh_class = traci.vehicle.getVehicleClass(veh)
            if veh_class == 'passenger':
                traci.vehicle.setMaxSpeed(veh, 55.55) # 55.55 is default SUMO settings, so should be ok
            elif veh_class == 'truck':
                traci.vehicle.setMaxSpeed(veh, 36.11) # 55.55 is default SUMO settings, so should be ok

    def traci_init(self, scenario_folder):
        self.incident_edge_lanes = list(range(traci.edge.getLaneNumber(self.incident_edge)))
        self.free_lanes = list(set(self.incident_edge_lanes) - set(self.lanes)) 
        self.calulate_slow_zone(scenario_folder)

    def calulate_slow_zone(self, scenario_folder):
        net_path = f'{scenario_folder}/Simulations/Base/network.net.xml'
        net = sumolib.net.readNet(net_path)
        i_edge_obj = net.getEdge(self.incident_edge)
        upstream_edges_obj = list(i_edge_obj.getIncoming().keys())
        self.upstream_edges = [edge_obj.getID() for edge_obj in upstream_edges_obj]
        self.upstream_edges_length_dict = {edge_obj.getID():edge_obj.getLength() for edge_obj in upstream_edges_obj} 
        self.upstream_edges_n_lanes_dict = {edge_obj.getID():edge_obj.getLaneNumber() for edge_obj in upstream_edges_obj}
        self.upstream_slow_zone = np.abs(self.slow_zone - self.pos)
        self.upstream_lc_zone = np.abs(self.lc_zone - self.pos) 
        self.upstream_lc_prob_zone = np.abs(self.lc_prob_zone - self.pos) 
            
        downstream_edges_obj = list(i_edge_obj.getOutgoing().keys())
        self.downstream_edges = [edge_obj.getID() for edge_obj in downstream_edges_obj]
        return