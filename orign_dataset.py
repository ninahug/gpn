import torch
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import json

# CAPACITIES = {10: 20., 20: 30., 50: 40., 100: 50.}
# CAPACITIES = {5: 10., 10: 20., 20: 30., 50: 40., 100: 50.}
CAPACITIES = {5: 10., 10: 20., 20: 30., 50: 40., 100: 50., 120: 50.}
max_demand = 9

def generate_data(device, batch = 10, n_car_each_depot = 15, n_depot = 1, n_customer = 20, capa = 1., seed = None):
	if seed is not None:
		torch.manual_seed(seed)
	n_node = n_depot + n_customer
	n_car = n_car_each_depot * n_depot
	# assert (9. / CAPACITIES[n_customer]) * n_customer <= capa * n_car, 'infeasible; Customer Demand should be smaller than Vechile Capacity' 
	assert (max_demand / CAPACITIES[n_customer]) * n_customer <= capa * n_car, 'infeasible; Customer Demand should be smaller than Vechile Capacity' 



	return {'depot_xy': torch.rand((batch, n_depot, 2), device = device)
			,'customer_xy': torch.rand((batch, n_customer, 2), device = device)
			# ,'demand': torch.randint(low = 1, high = 10, size = (batch, n_customer), device = device) / CAPACITIES[n_customer]
			,'demand': torch.randint(low = 1, high = max_demand+1, size = (batch, n_customer), device = device) / CAPACITIES[n_customer]
			# ,'car_start_node': torch.randint(low = 0, high = n_depot, size = (batch, n_car), device = device)
			,'car_start_node': torch.arange(n_depot, device = device)[None,:].repeat(batch, n_car_each_depot)
			# ,'car_capacity': torch.ones((batch, n_car), device = device)
			,'car_capacity': capa * torch.ones((batch, n_car), device = device)

			}
# def creat_data(device, n_samples = 5120, n_car_each_depot = 1, n_depot = 1, n_customer = 20, capa = 1., seed = None,batch_size=32):
#     edges_index = []
#     for i in range(n_nodes):
#         for j in range(n_nodes):
#             edges_index.append([i, j])
#     edges_index = torch.LongTensor(edges_index)
#     edges_index = edges_index.transpose(dim0=0,dim1=1)
#
#     datas = []
#
#     for i in range(num_samples):
#         node, edge, demand, capcity = creat_instance(num_samples, n_nodes)
#         data = Data(x=torch.from_numpy(node).float(), edge_index=edges_index,edge_attr=torch.from_numpy(edge).float(),
#                     demand=torch.tensor(demand).unsqueeze(-1).float(),capcity=torch.tensor(capcity).unsqueeze(-1).float())
#         datas.append(data)
#     #print(datas)
#     dl = DataLoader(datas, batch_size=batch_size)
#     return dl



class Generator(Dataset):
	""" https://github.com/utkuozbulak/pytorch-custom-dataset-examples
		 https://github.com/wouterkool/attention-learn-to-route/blob/master/problems/vrp/problem_vrp.py
		 https://github.com/nperlmut31/Vehicle-Routing-Problem/blob/master/dataloader.py
	"""
	def __init__(self, device, n_samples = 5120, n_car_each_depot = 1, n_depot = 1, n_customer = 20, capa = 1., seed = None):
		if seed is not None:
			self.data = generate_data(device, n_samples, n_car_each_depot, n_depot, n_customer, capa, seed)
		self.data = generate_data(device, n_samples, n_car_each_depot, n_depot, n_customer, capa, seed)
		
	def __getitem__(self, idx):
		dic = {}
		for k, v in self.data.items():
			# e.g., dic['depot_xy'] = self.data['depot_xy'][idx]
			dic[k] = v[idx]
		return dic

	def __len__(self):
		return self.data['depot_xy'].size(0)

	
if __name__ == '__main__':
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')	
	batch, batch_steps, n_customer = 128, 10, 20
	dataset = Generator(device, n_samples = batch*batch_steps, 
		n_car_each_depot = 15, n_depot = 1, n_customer = n_customer, capa = 2.)
	data = next(iter(dataset))	
	
	# generate_data(device, batch = 10, n_car = 15, n_depot = 2, n_customer = 20, seed = )
	# data = {}
	# seed = 123
	# for k in ['depot_xy', 'customer_xy', 'demand', 'car_start_node', 'car_capacity']:
	# 	elem = generate_data(device, batch = 1, n_car = 15, n_depot = 2, n_customer = n_customer, seed = seed)[k].squeeze(0)
	# 	data[k] = elem
	
	"""
	dataloader = DataLoader(dataset, batch_size = 8, shuffle = True)
	for i, data in enumerate(dataloader):
		for k, v in data.items():
			print(k, v.size())
			if k == 'demand': print(v[0])
		if i == 0:
			break
	"""
