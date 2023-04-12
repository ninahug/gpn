import pickle

import torch
import numpy as np
import os

from torch_geometric.data import Data,DataLoader
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
# from torch.utils.data import Data,DataLoader

import json

# CAPACITIES = {10: 20., 20: 30., 50: 40., 100: 50.}
# CAPACITIES = {5: 10., 10: 20., 20: 30., 50: 40., 100: 50.}
# from data_generator import TW_CAPACITIES

CAPACITIES = {5: 10., 10: 20., 20: 30., 50: 40., 100: 50., 120: 50.}
TW_CAPACITIES = {10: 250.,20: 500.,50: 750.,100: 1000.}
max_demand = 9

# def generate_data( device,batch = 10, n_car_each_depot = 15, n_depot = 1, n_customer = 20, capa = 1., seed = None):
# 	if seed is not None:
# 		torch.manual_seed(seed)
# 	n_node = n_depot + n_customer
# 	n_car = n_car_each_depot * n_depot
# 	# assert (9. / CAPACITIES[n_customer]) * n_customer <= capa * n_car, 'infeasible; Customer Demand should be smaller than Vechile Capacity'
# 	assert (max_demand / CAPACITIES[n_customer]) * n_customer <= capa * n_car, 'infeasible; Customer Demand should be smaller than Vechile Capacity'
# 	return {'depot_xy': torch.rand((batch, n_depot, 2), device = device)
# 			,'customer_xy': torch.rand((batch, n_customer, 2), device = device)
# 			# ,'demand': torch.randint(low = 1, high = 10, size = (batch, n_customer), device = device) / CAPACITIES[n_customer]
# 			,'demand': torch.randint(low = 1, high = max_demand+1, size = (batch, n_customer), device = device) / CAPACITIES[n_customer]
# 			# ,'car_start_node': torch.randint(low = 0, high = n_depot, size = (batch, n_car), device = device)
# 			,'car_start_node': torch.arange(n_depot, device = device)[None,:].repeat(batch, n_car_each_depot)
# 			# ,'car_capacity': torch.ones((batch, n_car), device = device)
# 			,'car_capacity': capa * torch.ones((batch, n_car), device = device)
# 			}
def generate_cvrptw_data(device,  n_customer, rnds=None, n_depot=2,n_car=5,
                         service_window=1000,
                         service_duration=10,
                         time_factor=100.0,
                         tw_expansion=3.0,
                         **kwargs):
    """Generate data for CVRP-TW

    Args:
        size (int): size of dataset
        graph_size (int): size of problem instance graph (number of customers without depot)
        rnds : numpy random state
        service_window (int): maximum of time units
        service_duration (int): duration of service
        time_factor (float): value to map from distances in [0, 1] to time units (transit times)
        tw_expansion (float): expansion factor of TW w.r.t. service duration

    Returns:
        List of CVRP-TW instances wrapped in named tuples
    """
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    rnds = np.random if rnds is None else rnds

    # sample locations
    dloc = rnds.uniform(size=(n_depot, 2))  # depot location

    nloc = rnds.uniform(size=(n_customer, 2))  # node locations
    n_nodes = np.concatenate((dloc, nloc), 0)
    edges_index = []
    for i in range(n_nodes.shape[0]):
        for j in range(n_nodes.shape[0]):
            edges_index.append([i, j])
    edges_index = torch.LongTensor(edges_index)
    edges_index = edges_index.transpose(dim0=0, dim1=1)

    edges = np.zeros((n_nodes.shape[0], n_nodes.shape[0], 1))
    for i, (x1) in enumerate(n_nodes):
        for j, (x2) in enumerate(n_nodes):
            d = ((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2) ** 0.5
            edges[i][j][0] = d
    edges = edges.reshape(-1, 1)
    # print(dloc)
    # print(np.min(dloc, 0)[None, :])
    min_t = np.ceil(np.linalg.norm(np.min(dloc, 0)[None, :] * time_factor - nloc * time_factor, axis=-1)) + 1

    # TW end needs to be early enough to perform service and return to depot until end of service window
    max_t = np.ceil(
        np.linalg.norm(np.min(dloc, 0)[None, :] * time_factor - nloc * time_factor, axis=-1) + service_duration) + 1
    # horizon allows for the feasibility of reaching nodes / returning from nodes within the global tw (service window)
    horizon = list(zip(min_t, service_window - max_t))

    epsilon = np.maximum(np.abs(rnds.standard_normal([n_customer])), 1 / time_factor)

    # sample earliest start times a
    a = [rnds.randint(*h) for h in horizon]

    # calculate latest start times b, which is
    # a + service_time_expansion x normal random noise, all limited by the horizon
    # and combine it with a to create the time windows
    tw = [np.transpose(np.vstack((rt,  # a
                                  np.minimum(rt + tw_expansion * time_factor * sd, h[-1]).astype(int)  # b
                                  ))).tolist()
          for rt, sd, h in zip(a, epsilon, horizon)]
    d_c = torch.tensor(
        np.minimum(np.maximum(np.abs(rnds.normal(loc=15, scale=10, size=[n_customer])).astype(int), 1),
                   42).tolist(), device=device)
    d_d = torch.tensor(np.full([n_depot], 0).tolist(), device=device)
    demand = torch.cat((d_d, d_c), 0)
    demand = demand / (torch.max(demand))
    # demend = (demand)

    car_capacity = torch.tensor(
        np.round(np.full((n_depot * n_car), TW_CAPACITIES[n_customer] / (n_depot * n_car))).tolist(),
        device=device)

    depot_tw = torch.tensor([[[0, service_window]] * n_depot], device=device).squeeze(0)

    node_tw = torch.tensor(tw, device=device).squeeze(1)



    t_tw = torch.cat((depot_tw, node_tw), 0)
    t_tw = t_tw / (torch.max(t_tw))
    d_durations = torch.tensor(np.full([n_depot], 0).tolist(), device=device)

    c_durations = torch.tensor(np.full([n_customer], service_duration).tolist(), device=device)

    durations = torch.cat((d_durations, c_durations), 0)
    car_start_node = torch.arange(n_depot)[:].repeat(n_car)



    return 	{"depot_loc": torch.tensor(dloc.tolist(), device = device),
			"node_loc": torch.tensor(nloc.tolist(), device = device),
			"demand": torch.tensor(demand, device = device),
			"capacity": torch.tensor(car_capacity, device = device),
			"tw": torch.tensor(t_tw, device = device),
            "node_tw":torch.tensor(tw, device = device),
            "durations": torch.tensor(durations, device = device),
            "car_start_node": torch.tensor(car_start_node, device = device),
			"edges_index": edges_index,
			"edges": torch.tensor(edges, device=device)

    }


def creat_cvrptw_data(device,  n_customer, rnds=None, n_depot=2,n_car=5,
                         service_window=1000,
                         service_duration=10,
                         time_factor=100.0,
                         tw_expansion=3.0,
                         **kwargs):
    """Generate data for CVRP-TW

    Args:
        size (int): size of dataset
        graph_size (int): size of problem instance graph (number of customers without depot)
        rnds : numpy random state
        service_window (int): maximum of time units
        service_duration (int): duration of service
        time_factor (float): value to map from distances in [0, 1] to time units (transit times)
        tw_expansion (float): expansion factor of TW w.r.t. service duration

    Returns:
        List of CVRP-TW instances wrapped in named tuples
    """
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    rnds = np.random if rnds is None else rnds

    # sample locations
    dloc = rnds.uniform(size=(n_depot, 2))  # depot location

    nloc = rnds.uniform(size=( n_customer, 2))  # node locations
    n_nodes = np.concatenate((dloc, nloc), 0)
    edges_index = []
    for i in range(n_nodes.shape[0]):
        for j in range(n_nodes.shape[0]):
            edges_index.append([i, j])
    edges_index = torch.LongTensor(edges_index)
    edges_index = edges_index.transpose(dim0=0, dim1=1)

    edges = np.zeros(( n_nodes.shape[0], n_nodes.shape[0], 1))
    for i, (x1) in enumerate(n_nodes):
        for j, (x2) in enumerate(n_nodes):
            d = ((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2) ** 0.5
            edges[i][j][0] = d
    edges = edges.reshape(-1, 1)

    min_t = np.ceil(np.linalg.norm(np.min(dloc, 0)[None, :] * time_factor - nloc * time_factor, axis=-1)) + 1

    # TW end needs to be early enough to perform service and return to depot until end of service window
    max_t = np.ceil(
        np.linalg.norm(np.min(dloc, 0)[ None, :] * time_factor - nloc * time_factor, axis=-1) + service_duration) + 1
    # horizon allows for the feasibility of reaching nodes / returning from nodes within the global tw (service window)
    horizon = list(zip(min_t, service_window - max_t))

    epsilon = np.maximum(np.abs(rnds.standard_normal([n_customer])), 1 / time_factor)

    # sample earliest start times a
    a = [rnds.randint(*h) for h in horizon]

    # calculate latest start times b, which is
    # a + service_time_expansion x normal random noise, all limited by the horizon
    # and combine it with a to create the time windows
    tw = [np.transpose(np.vstack((rt,  # a
                                  np.minimum(rt + tw_expansion * time_factor * sd, h[-1]).astype(int)  # b
                                  ))).tolist()
          for rt, sd, h in zip(a, epsilon, horizon)]
    d_c = torch.tensor(
                np.minimum(np.maximum(np.abs(rnds.normal(loc=15, scale=10, size=[n_customer])).astype(int), 1),
                           42).tolist(), device=device)
    d_d = torch.tensor(np.full([n_depot], 0).tolist(), device=device)
    demand = torch.cat((d_d,d_c), 0)
    # demand = demand1 / (torch.max(demand1))
    # print("torch.max(demand)",torch.max(demand1))


    car_capacity =torch.tensor(np.round(np.full((n_depot*n_car), TW_CAPACITIES[n_customer] / (n_depot*n_car))).tolist(),
                 device=device)

    depot_tw = torch.tensor([[[0, service_window]] * n_depot] , device=device).squeeze(0)

    node_tw =  torch.tensor(tw, device=device).squeeze(1)

    t_tw = torch.cat((depot_tw, node_tw),0)

    d_durations = torch.tensor(np.full([n_depot], 0).tolist(), device=device)

    c_durations = torch.tensor(np.full([n_customer], service_duration).tolist(), device=device)

    durations = torch.cat((d_durations,c_durations), 0)
    car_start_node = torch.arange(n_depot)[:].repeat(n_car)


    return n_nodes,t_tw,edges,edges_index,durations, car_capacity,demand,car_start_node,nloc,dloc

def cvrptw(device, size,n_customer, rnds=None, n_depot=2,n_car=5,
                         service_window=1000,
                         service_duration=10,
                         time_factor=100.0,
                         tw_expansion=3.0,
                         **kwargs):
    datas = []

    for i in range(size):
        n_nodes, t_tw, edges, edges_index, durations, car_capacity,demand,car_start_node,nloc,dloc = creat_cvrptw_data(device, n_customer, rnds, n_depot,n_car,
                         service_window,
                         service_duration,
                         time_factor,
                         tw_expansion,
                         **kwargs)

        data = Data(x=torch.from_numpy(n_nodes).float(),  edge_index=edges_index, edge_attr=torch.from_numpy(edges).float(),
                    demand=torch.tensor(demand).unsqueeze(-1).float(),tw =t_tw.float(),durations=torch.tensor(durations).unsqueeze(-1).float(),
                    capcity=torch.tensor(car_capacity).unsqueeze(-1).float(),car_start_node = car_start_node,nloc=torch.from_numpy(nloc).float(),dloc=torch.from_numpy(dloc).float())
        datas.append(data)
    # dl = DataLoader(datas, batch_size=2)
    # # print("data", next(iter(dl)))
    # print(datas)
    return datas




if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch, batch_steps, n_customer = 128, 10, 20
    datas =  creat_cvrptw_data(device,  n_customer, rnds=None, n_depot=2,n_car=5,
                         service_window=1000,
                         service_duration=10,
                         time_factor=100.0,
                         tw_expansion=3.0)
    print("datas",datas)
        # cvrptw(device, size = 3,n_customer=20, seed = 1234,
		# n_depot=2,n_car=3,service_window = 1000, service_duration = 10,
		# time_factor = 100.0,tw_expansion = 3.0)







	# """
	# dataloader = DataLoader(dataset, batch_size = 8, shuffle = True)
	# for i, data in enumerate(dataloader):
	# 	for k, v in data.items():
	# 		print(k, v.size())
	# 		if k == 'demand': print(v[0])
	# 	if i == 0:
	# 		break
	# """
