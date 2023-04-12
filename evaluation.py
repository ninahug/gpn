import copy
import os
import random
from time import time
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from readdata import creat_data

from baseline import load_model
from config import test_parser
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import  DataLoader
from tqdm import tqdm
from time import time

from Nets.model import AttentionModel
from baseline import RolloutBaseline
from dataset import  cvrptw
from config import Config, load_pkl, train_parser

import sys
sys.path.append('../')
from dataclass import TorchJson



def evaliuate(path):

    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dl,nodes,n_nodes,ys_demand,ys_capacity,numdepots,ys_start,ys_end=creat_data(path,num_samples=1 ,batch_size=1)

    epoch = torch.arange(1).numpy()
    totalcost = []
    for i in epoch:
        # weight_path = './Weights/VRP20_epoch%s.pt'% (i)
        weight_path = './Weights/VRP100_epoch1.pt'
        agent = load_model(device, weight_path, embed_dim = 128, n_encode_layers = 3)
        agent = agent.to(device)
        agent.eval()
        with torch.no_grad():
            for  t,inputs in enumerate(dl):
                cost, _, pis = agent(inputs, return_pi=True, decode_type='greedy')

    totalcost.append(cost)

    return  totalcost


def plot_route(tup, title):
    routes, depot_xy, customer_xy, demands, xy, cost, pi, idx_in_batch, car_capacity = tup

    customer_labels = ['(' + str(demand) + ')' for demand in demands.round(2)]
    path_traces = []
    for i, route in enumerate(routes, 1):
        coords = xy[[int(x) for x in route]]

        # Calculate length of each agent loop
        lengths = np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1))
        total_length = np.sum(lengths)

        path_traces.append(go.Scatter(x=coords[:, 0],
                                      y=coords[:, 1],
                                      mode='markers+lines',
                                      name=f'Vehicle{i}: Length = {total_length:.3f}',
                                      opacity=1.0))

    trace_points = go.Scatter(x=customer_xy[:, 0],
                              y=customer_xy[:, 1],
                              mode='markers+text',
                              name='Customer (demand)',
                              text=customer_labels,
                              textposition='top center',
                              marker=dict(size=7),
                              opacity=1.0
                              )

    trace_depo = go.Scatter(x=depot_xy[:, 0],
                            y=depot_xy[:, 1],
                            # mode = 'markers+text',
                            mode='markers',
                            # name = 'Depot (Capacity = 1.0)',
                            name='Depot',
                            # text = ['1.0'],
                            # textposition = 'bottom center',
                            marker=dict(size=23),
                            marker_symbol='triangle-up'
                            )

    layout = go.Layout(
        # title = dict(text = f'<b>VRP{customer_xy.shape[0]} depot{depot_xy.shape[0]} {title}, Total Length = {cost:.3f}</b>', x = 0.5, y = 1, yanchor = 'bottom', yref = 'paper', pad = dict(b = 10)),#https://community.plotly.com/t/specify-title-position/13439/3
        title=dict(text=f'<b>VRP{customer_xy.shape[0]} depot{depot_xy.shape[0]} {title}, Total cost = {cost:.3f}</b>',
                   x=0.5, y=1, yanchor='bottom', xref='paper', yref='paper', pad=dict(b=10)),
        # xaxis = dict(title = 'X', range = [0, 1], ticks='outside'),
        # yaxis = dict(title = 'Y', range = [0, 1], ticks='outside'),#https://kamino.hatenablog.com/entry/plotly_for_report
        xaxis=dict(title='X', range=[0, 1], linecolor='black', showgrid=False, ticks='inside', linewidth=2,
                   mirror=True),
        yaxis=dict(title='Y', range=[0, 1], linecolor='black', showgrid=False, ticks='inside', linewidth=2,
                   mirror=True),
        showlegend=True,
        width=750,
        height=700,
        autosize=True,
        template="plotly_white",
        legend=dict(x=1.05, xanchor='left', y=0, yanchor='bottom', bordercolor='black', borderwidth=1)
        # legend = dict(x = 1, xanchor = 'right', y =0, yanchor = 'bottom', bordercolor = '#444', borderwidth = 0)
        # legend = dict(x = 0, xanchor = 'left', y =0, yanchor = 'bottom', bordercolor = '#444', borderwidth = 0)
    )

    data = [trace_points, trace_depo] + path_traces
    fig = go.Figure(data=data, layout=layout)
    fig.show()





def get_imlist(path):
    path_list = os.listdir(path)
    # path_list.sort(key=lambda x:int(x[:]))
    a = []
    for fikena in path_list:
        a.append(os.path.join(path, fikena))
    return a




if __name__ == '__main__':

    fie = './Data/'
    filename = get_imlist( fie )
    print(filename)
    results = []
    times = []
    # cfg = load_pkl(train_parser().path)
    # for path in filename:
    #     result = evaliuate(path)
    #     print("path", path)
    #     print("result",result)
    #
    #     c_path = '%s%s_cost.csv' % ('./Csv/', path[7:10])
    #     with open(c_path, 'w') as f:
    #         for r in result:
    #             f.write('0,%1.4f\n' % (r))
 # for path in filename:
    path= './Data/pr10.txt'
    result = evaliuate(path)
        # print("path", path)
    print("result",result)

    c_path = '%s%s_cost.csv' % ('./Csv/','PR')
    with open(c_path, 'w') as f:
        for r in result:
            f.write('0,%1.4f\n' % (r))
