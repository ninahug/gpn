import csv
import torch
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data,DataLoader
import os
import numpy as np
import pandas as pd
import multiprocessing
from joblib import Parallel, delayed
import time
import copy
from data_classes import Depot, Customer



def load_problem(path):
    global depots, customers
    capcity = 0
    depots = []
    demands = []
    c_durations = []
    nloc = []
    dloc = []
    x_y = []
    s = []
    e =[]
    cart=[]
    
    customers = []

    with open(path) as f:
        _,max_vehicles, num_customers, num_depots = tuple(map(lambda z: int(z), f.readline().strip().split()))
        
        for i in range(num_depots):
            max_duration, max_load = tuple(map(lambda z: int(z), f.readline().strip().split()))

            depots.append(Depot(max_vehicles, max_duration, max_load))
            capcity = max_load


        for i in range(num_customers):
            vals = tuple(map(lambda z: float(z), f.readline().strip().split()))
            cid, x, y, service_duration, demand = (vals[j] for j in range(5))
            start,end = (vals[j] for j in range(11,13))
            # print("start",start)
            # print("end",end)
            customers.append(Customer(cid, x, y, service_duration, demand,start,end))
            x_y.append([x,y])
            demands.append(demand)
            s.append(start)
            e.append(end)
            c_durations.append(service_duration)
            nloc.append([x,y])


        for i in range(num_depots):
            vals = tuple(map(lambda z: float(z), f.readline().strip().split()))
            cid, x, y = (vals[j] for j in range(3))
            start,end = (vals[j] for j in range(7,9))
            depots[i].pos = (x, y)
            x_y.append([x,y])
            demands.append(0)
            s.append(start)
            e.append(end)


            d_durations = np.full([num_depots], 0).tolist()
            dloc.append([x, y])

            

        #demands = demands[0:-num_depots]
        #x_y = x_y[0:-num_depots]

        demands, x_y = np.array(demands), np.array(x_y)
        demands = np.concatenate((demands[-num_depots:],demands[0:-num_depots]))

        x_y = np.concatenate((x_y[-num_depots:], x_y[0:-num_depots]))

        s = np.array(s)
        s = np.concatenate((s[-num_depots:], s[0:-num_depots]))
        e = np.array(e)
        e = np.concatenate((e[-num_depots:], e[0:-num_depots]))
        durations = np.concatenate((d_durations, c_durations))

        nloc = np.array(nloc)
        dloc = np.array(dloc)
        capcity = np.full(num_depots*max_vehicles, capcity).tolist()
        return x_y,demands, capcity,num_depots,s,e,durations,nloc,dloc


def creat_instance(path):
    citys,demand,capcity,numdepots,start,end,durations,nloc,dloc = load_problem(path)


    nodes = citys.copy()
    n_nodes=citys.shape[0]
    d_nodes=dloc.shape[0]

    ys_demand=demand.copy()
    ys_start=start.copy()
    ys_end =end.copy()
    ys_capacity=capcity.copy()

    demand=demand.reshape(-1)
    start=start.reshape(-1)
    end=end.reshape(-1)
    durations=durations.reshape(-1)

#---------------------------------------------------------------------坐标归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    citys = scaler.fit_transform(citys)

    mc = np.max(nodes)
    nloc = nloc/ (mc)
    dloc = dloc/ (mc)

    # citys = citys / (max)
# #----------------------------------------------------------------------需求归一化
    demand_max = np.max(demand)
    ss=start[d_nodes:]
    # print( ss)
    ee = end[d_nodes:]

    demand = demand / (demand_max)
    start_max = np.max(ss)
    start = start / (start_max)
    end_max = np.max(ee)
    end = end / (end_max)
    duration_max = np.max(ee)
    durations = durations / (duration_max)
#
# #----------------------------------------------------------------------容量归一化
    capcity=capcity/demand_max

    def c_dist(x1,x2):
        return ((x1[0]-x2[0])**2+(x1[1]-x2[1])**2)**0.5
    #edges = torch.zeros(n_nodes,n_nodes)
    edges = np.zeros((n_nodes,n_nodes,1))

    for i, (x1, y1) in enumerate(citys):
        for j, (x2, y2) in enumerate(citys):
            d = c_dist((x1, y1), (x2, y2))
            edges[i][j][0]=d
            
            
    edges = edges.reshape(-1, 1)
    edges_index = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            edges_index.append([i, j])
    edges_index = torch.LongTensor(edges_index)
    edges_index = edges_index.transpose(dim0=0, dim1=1)


    return citys,edges,demand,edges_index,capcity,nodes,n_nodes,ys_demand,ys_capacity,numdepots,start,end,ys_start,ys_end,durations,nloc,dloc#demand(num,node) capcity(num)

def creat_data(path,num_samples=4 ,batch_size=4):

    datas = []
    nodes1=[]


    for i in range(num_samples):
        citys,edges,demand,edges_index,capcity,nodes,n_nodes,ys_demand,ys_capacity,numdepots,start,end,ys_start,ys_end,durations,nloc,dloc= creat_instance(path)
        start = torch.tensor(start).unsqueeze(-1)
        end = torch.tensor(end).unsqueeze(-1)
        tw = [start, end]
        tw = torch.cat(tw, 1)

        n_car = int(capcity.shape[0]/dloc.shape[0])

        car_start_node = torch.arange(len(dloc))[:].repeat(n_car)
        # print("nodes",nodes)


        data = Data(x=torch.from_numpy(citys).float(), edge_index=edges_index, edge_attr=torch.from_numpy(edges).float(),
             demand=torch.tensor(demand).unsqueeze(-1).float(), tw=tw.float(),
             durations=torch.tensor(durations).unsqueeze(-1).float(),
             capcity=torch.tensor(capcity).unsqueeze(-1).float(), car_start_node=car_start_node,
             nloc=torch.from_numpy(nloc).float(), dloc=torch.from_numpy(dloc).float())

        datas.append(data)

    #print(datas)
    dl = DataLoader(datas, batch_size=batch_size)

    # print(" dl",  dl)

    return dl,nodes,n_nodes,ys_demand,ys_capacity,numdepots,ys_start,ys_end

if __name__ == '__main__':
    path = './Data/pr03.txt'
    load_problem(path)
    creat_data(path, num_samples=1, batch_size=1)
