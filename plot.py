import copy
import random
from time import time
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.cluster import KMeans
# from dataset2 import generate_data
from dataset import  cvrptw
from baseline import load_model
from config import test_parser

import sys

from readdata import creat_data

sys.path.append('../')
from dataclass import TorchJson


def clear_route(arr):
	print("arr", arr)
	dst = []
	for i in range(len(arr)-1):
		print("arr[i]", arr[i])
		if arr[i] != arr[i+1]:
			dst.append(arr[i])
	if len(dst) > 0:
		dst.append(dst[0])

	return dst

def get_dist(n1, n2):
	x1,y1,x2,y2 = n1[0],n1[1],n2[0],n2[1]
	if isinstance(n1, torch.Tensor):
		return torch.sqrt((x2-x1).pow(2)+(y2-y1).pow(2))
	elif isinstance(n1, (list, np.ndarray)):
		return math.sqrt(pow(x2-x1,2)+pow(y2-y1,2))
	else:
		raise TypeError
	
def get_dist_mat(xy): 
	n = len(xy)
	dist_mat = [[0. for i in range(n)] for i in range(n)]
	for i in range(n):
		for j in range(i, n):
			dist = get_dist(xy[i], xy[j])
			dist_mat[i][j] = dist_mat[j][i] = dist#round(float(two), digit)

	return dist_mat

def opt2_swap(route, dist_mat): 
	size = len(route)
	improved = True
	while improved:
		improved = False
		for i in range(size - 2):
			i1 = i + 1
			a = route[i]
			b = route[i1]
			for j in range(i + 2, size):
				j1 = j + 1
				if j == size - 1:
					j1 = 0

				c = route[j]
				d = route[j1]
				if i == 0 and j1 == 0: continue# if i == j1
				if(dist_mat[a][c] + dist_mat[b][d] < dist_mat[a][b] + dist_mat[c][d]):
					""" i i+1 j j+1
						swap(i+1, j)
					"""
					tmp = route[i1:j1]
					route[i1:j1] = tmp[::-1]# tmp in inverse order
					improved = True 
	return route

def apply_2opt(tup):
	routes, depot_xy, customer_xy, demands, xy, cost, pi, idx_in_batch,car_capacity,tw = tup

	dist_mat = get_dist_mat(xy)
	new_routes = []
	for route in routes:# apply 2-opt to each route
		if len(route) > 0: new_routes.append(opt2_swap(route, dist_mat))
		# exchange_path(routes,xy)

	num_depot = len(depot_xy)
	cost = 0.
	fuel_cost = 0.
	fuel_pay = 0.
	pay_cost = 0.
	for i, route in enumerate(new_routes, 1):
		coords = xy[[int(x) for x in route]]
		demand = [demands[i - num_depot] for i in route[1:-1]]
		capitity = 185
		car_no_used =[]
		car_used = []
		for j in range(len(demand)):
			car_left = capitity -demand[j]
			car_no_used.append(car_left)
		car_used = np.append(capitity, car_no_used)
		# Calculate length of each agent loop
		lengths = np.sqrt(np.sum(np.diff(coords, axis = 0) ** 2, axis = 1))
		fuel_cost = car_used * lengths
		fuel_pay = fuel_cost.sum(0)*57.3

		total_length = np.sum(lengths)

		pay_cost = total_length
		cost += pay_cost+ fuel_pay
		# cost += fuel_pay
		# cost += pay_cost


	# cost = cost
	cost = cost+len(new_routes) * 100
	return (new_routes, depot_xy, customer_xy, demands, xy, cost, pi, idx_in_batch,car_capacity,tw)

def get_more_info(cost, pi, idx_in_batch):
	
	# Remove unneeded values
	routes = []
	for pi_of_each_car in pi:
		route = clear_route(pi_of_each_car)
		if len(route) > 0:
			routes.append(route)

	
	# data.keys(), ['depot_xy', 'customer_xy', 'demand', 'car_start_node', 'car_capacity']
	depot_xy = data.dloc.cpu().numpy()
	customer_xy = data.nloc.cpu().numpy()
	demands = data.demand.cpu().numpy()
	xy = np.concatenate([depot_xy, customer_xy], axis = 0)
	car_capacity = data.capcity.cpu().numpy()
	tw = data.tw.cpu().numpy()
	return (routes, depot_xy, customer_xy, demands, xy, cost, pi, idx_in_batch,car_capacity,tw)

def plot_route(tup, title):
	routes, depot_xy, customer_xy, demands, xy, cost, pi, idx_in_batch,car_capacity,tw = tup

	# customer_labels = ['(' + str(demand) + ')' for demand in demands.round(2)]
	path_traces = []
	for i, route in enumerate(routes, 1):
		coords = xy[[int(x) for x in route]]


		# Calculate length of each agent loop
		lengths = np.sqrt(np.sum(np.diff(coords, axis = 0) ** 2, axis = 1))
		total_length = np.sum(lengths)
		
		path_traces.append(go.Scatter(x = coords[:, 0],
									y = coords[:, 1],
									mode = 'markers+lines',
									name = f'Vehicle{i}: Length = {total_length:.3f}',
									opacity = 1.0))

	
	trace_points = go.Scatter(x = customer_xy[:, 0],
							  y = customer_xy[:, 1],
							  mode = 'markers+text', 
							  # name = 'Customer (demand)',
							  # text = customer_labels,
							  textposition = 'top center',
							  marker = dict(size = 7),
							  opacity = 1.0
							  )

	trace_depo = go.Scatter(x = depot_xy[:,0],
							y = depot_xy[:,1],
							# mode = 'markers+text',
							mode = 'markers',
							# name = 'Depot (Capacity = 1.0)',
							name = 'Depot',
							# text = ['1.0'],
							# textposition = 'bottom center',
							marker = dict(size = 23),
							marker_symbol = 'triangle-up'
							)
	
	layout = go.Layout(
		# title=dict(text=f'<b>VRP{customer_xy.shape[0]} depot{depot_xy.shape[0]} {title}, Total cost = {cost:.3f}</b>',
		# 		   x=0.5, y=1, yanchor='bottom', xref='paper', yref='paper', pad=dict(b=10)),
						# title = dict(text = f'<b>VRP{customer_xy.shape[0]} depot{depot_xy.shape[0]} {title}, Total Length = {cost:.3f}</b>', x = 0.5, y = 1, yanchor = 'bottom', yref = 'paper', pad = dict(b = 10)),#https://community.plotly.com/t/specify-title-position/13439/3
						title = dict(text = f'<b>VRP{customer_xy.shape[0]} depot{depot_xy.shape[0]} {title}</b>', x = 0.5, y = 1, yanchor = 'bottom', xref = 'paper', yref = 'paper', pad = dict(b = 10)),
						# xaxis = dict(title = 'X', range = [0, 1], ticks='outside'),
						# yaxis = dict(title = 'Y', range = [0, 1], ticks='outside'),#https://kamino.hatenablog.com/entry/plotly_for_report
						xaxis = dict(title = '<b>X</b>', range = [-1.2, 1.2], linecolor = 'black', showgrid=False, ticks='inside', linewidth=3, mirror=True),
						yaxis = dict(title = '<b>Y</b>', range = [-1.2, 1.2], linecolor = 'black',showgrid=False, ticks='inside', linewidth=3, mirror=True),
						showlegend = True,
						width = 750,
						height = 700,
						autosize = True,
						template = "plotly_white",
						legend = dict(x = 1.05, xanchor = 'left', y =0, yanchor = 'bottom', bordercolor = 'black', borderwidth = 2)
						# legend = dict(x = 1, xanchor = 'right', y =0, yanchor = 'bottom', bordercolor = '#444', borderwidth = 0)
						# legend = dict(x = 0, xanchor = 'left', y =0, yanchor = 'bottom', bordercolor = '#444', borderwidth = 0)
						)

	data = [trace_points, trace_depo] + path_traces
	fig = go.Figure(data = data, layout = layout)
	fig.show()

if __name__ == '__main__':
	# args = test_parser()
	# t1 = time()
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	# pretrained = load_model(device, args.path, embed_dim = 128, n_encode_layers = 3)
	# print(f'model loading time:{time()-t1}s')
	#
	# t2 = time()
	# if args.txt is not None:
	# 	hoge = TorchJson(args.txt)
	# 	data = hoge.load_json(device)# return tensor on GPU
	# 	for k, v in data.items():
	# 		shape = (args.batch, ) + v.size()[1:]
	# 		data[k] = v.expand(*shape).clone()
	# 		# print('k, v', k, *v.size())
	# 		# print(*shape)
	#
	# else:
	# 	data = {}
	# 	for k in ['depot_xy', 'customer_xy', 'demand', 'car_start_node', 'car_capacity']:
	# 		elem = [generate_data(device, batch = 1, n_car = args.n_car, n_depot = args.n_depot, n_customer = args.n_customer, seed = args.seed)[k].squeeze(0) for j in range(args.batch)]
	# 		data[k] = torch.stack(elem, 0)
	#
	# # for k, v in data.items():
	# # 	print('k, v', k, v.size())
	# # 	print(v.type())# dtype of tensor

	dl, nodes, n_nodes, ys_demand, ys_capacity, numdepots, ys_start, ys_end = creat_data('./Data/pr03.txt', num_samples=1,
																						 batch_size=1)
	data = next(iter(dl))
	pretrained = load_model(device,'./Weights/VRP50_epoch14.pt', embed_dim=128, n_encode_layers=3)
	
	# print(f'data generate time:{time()-t1}s')
	pretrained = pretrained.to(device)
	# data = list(map(lambda x: x.to(device), data))
	pretrained.eval()
	with torch.no_grad():
		print('data', data)
		costs, _, pis = pretrained(data, return_pi = True, decode_type='greedy')
		print(" pis",  pis)
	# print('costs:', costs)
	idx_in_batch = torch.argmin(costs, dim = 0)
	cost = costs[idx_in_batch].cpu().numpy()
	# if args.write_csv is not None:
	# 	with open(args.write_csv, 'a') as f:
	# 		f.write(f'{time()-t1},{time()-t2},{cost:.3f}\n')
	# print(f'decode type: {args.decode_type}\nminimum cost(without 2opt): {cost:.3f}\nidx: {idx_in_batch} out of {args.batch} solutions')
	# print(f'\ninference time: {time()-t1}s')
	# print(f'inference time(without loading model): {time()-t2}s')
	
	pi = pis[idx_in_batch].cpu().numpy()
	tup = get_more_info(cost, pi, idx_in_batch)
	print("tup",tup)


	# if args.write_csv is None:
	title = 'pr03'
	# plot_route(tup, title)
	# print('plot time: ', time()-t1)
	# print(f'plot time(without loading model): {time()-t2}s')

	tup = apply_2opt(tup)
	tup = apply_2opt(tup)
	# tup = apply_2opt(tup)
	# tup = apply_2opt(tup)
	# tup = apply_2opt(tup)
	plot_route(tup, title)
	# newClusters = kmeans_exmaple(tup)
	# ind2route(newClusters)
	# cost = tup[-4]
	# if args.write_csv_2opt is not None:
	# 	with open(args.write_csv_2opt, 'a') as f:
	# 		f.write(f'{time()-t1},{time()-t2},{cost:.3f}\n')
	# print(f'minimum cost(without 2opt): {cost:.3f}')
	# print('inference time: ', time()-t1)
	# print(f'inference time(without loading model): {time()-t2}s')
	#
	# if args.write_csv_2opt is None:
	# 	title = 'Pretrained'
	#
	# 	print('plot time: ', time()-t1)
	# 	print(f'plot time(without loading model): {time()-t2}s')