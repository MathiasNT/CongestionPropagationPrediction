import traci
import numpy as np

def block_lanes(lane_pos_times, step):
    """Blocks lanes at the specified position and timesteps

    Args:
        lane_pos_times (list of strings): A list of strings with format "{edge}_{lane}_{pos}_{time}_{duration}" that defines the blockages
        step (int): Simulation step for time tracking
        duration (int, optional): Duration of incidents. Defaults to 1200.

    Returns:
        list of strings: A list of strings for blockages that have not started nor finished yet. 
    """
    new_lane_pos_times = []
    for lane_pos_time in lane_pos_times:
        edge, lane, pos, time, duration = lane_pos_time.split('_')
        time = int(time)
        duration = int(duration)
        incident_veh_id = f'incident_veh_{edge}_{lane}_{pos}'
        incident_route_id = f"incident_route_{edge}_{lane}_{pos}"

        if step==time:                                                     # Create block if time, check for collisions first.
            on_edge = traci.lane.getLastStepVehicleIDs(f"{edge}_{lane}")
            if on_edge:
                veh_pos_on_edge = []
                for veh in on_edge:
                    veh_pos_on_edge.append(traci.vehicle.getLanePosition(veh))
                print(f"step: {step}, lane: {edge}_{lane}, on edge: {on_edge}, pos: {veh_pos_on_edge}")
                if np.abs(np.min(veh_pos_on_edge) - int(pos)) < 7:          # Checking the closest vehicle should be good enough
                    prob_veh = on_edge[np.argmin(veh_pos_on_edge)]
                    print(f"{prob_veh} is too close, removing it")
                    traci.vehicle.remove(prob_veh)
            print(f"creating block {lane_pos_time}")
            traci.route.add(incident_route_id, [edge])
            traci.vehicle.add(vehID=incident_veh_id, routeID=incident_route_id)
            traci.vehicle.moveTo(vehID=incident_veh_id, laneID=f'{edge}_{lane}', pos=pos)
            traci.vehicle.setSpeed(vehID=incident_veh_id, speed=0)
            traci.vehicle.setLaneChangeMode(vehID=incident_veh_id, lcm=0)
            new_lane_pos_times.append(lane_pos_time)
        elif step > time and (step-time)%100==0 and (step-time) < duration: # Starts moving block to avoid time out
            traci.vehicle.setSpeed(vehID=incident_veh_id, speed=0.101)
            new_lane_pos_times.append(lane_pos_time)
        elif step > time and (step-time)%102==0 and (step-time) < duration: # Stops moving block
            traci.vehicle.setSpeed(vehID=incident_veh_id, speed=0)
            new_lane_pos_times.append(lane_pos_time)
        elif step==(time+duration): # Removes block
            print(f"removing block {lane_pos_time}")
            traci.vehicle.remove(vehID=incident_veh_id)
        else:                                                               # Keeps blocks that haven't started in list
            new_lane_pos_times.append(lane_pos_time)

    return new_lane_pos_times