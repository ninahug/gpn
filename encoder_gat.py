import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
import copy
import math
from torch.distributions.categorical import Categorical
import sys
sys.path.append('../')
from tupe import TUPEConfig, TUPEEncoder
config  = TUPEConfig()
tupe = TUPEEncoder(config)
# from vrpUpdate import update_mask,update_state
from dataset import  cvrptw
# from dataset2 import Generator
INIT = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GatConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_channels,
                 negative_slope=0.2,dropout=0):
        super(GatConv, self).__init__(aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.fc = nn.Linear(in_channels, out_channels)
        self.attn = nn.Linear(2* out_channels+edge_channels,out_channels)

    def forward(self, x, edge_index, edge_attr, size=None):
        x = self.fc(x)
        return self.propagate(edge_index, size=size, x=x, edge_attr=edge_attr)
        #它调用了propagate函数，这个函数就会去调用 message函数和aggregate函数
    def message(self, edge_index_i, x_i, x_j, size_i, edge_attr):
        x = torch.cat([x_i, x_j, edge_attr], dim=-1)

        alpha = self.attn(x) #2.对节点特征矩阵做线性变换
        alpha = F.leaky_relu(alpha, self.negative_slope)#3.计算节点的归一化系数

        alpha = softmax(alpha, edge_index_i)#4.对节点特征做归一化处理

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha

    def update(self, aggr_out):
        return aggr_out

class Encoder(nn.Module):
    def __init__(self, input_node_dim, embed_dim, input_edge_dim, hidden_edge_dim, conv_layers=3, n_heads=4):
        super(Encoder, self).__init__()
        self.hidden_node_dim = embed_dim
        self.fc_node = nn.Linear(input_node_dim, embed_dim )
        self.bn = nn.BatchNorm1d(embed_dim )
        self.be = nn.BatchNorm1d(hidden_edge_dim)
        self.fc_edge = nn.Linear(input_edge_dim, hidden_edge_dim)  # 1-16

        self.convs1 = nn.ModuleList(
            [GatConv(embed_dim , embed_dim , hidden_edge_dim) for i in range(conv_layers)])
        if INIT:
            for name, p in self.named_parameters():
                if 'weight' in name:
                    if len(p.size()) >= 2:
                        nn.init.orthogonal_(p, gain=1)
                elif 'bias' in name:
                    nn.init.constant_(p, 0)

    def forward(self, data):
        batch_size = 1
        data = data.to(device)

        x = torch.cat([(data.x).to(device),(data.tw).to(device), (data.demand).to(device)], -1)

        x = self.fc_node(x)
        x = self.bn(x)

        x = x.view(batch_size, -1, self.hidden_node_dim)

        x = tupe(x)

        x = x.view(-1, self.hidden_node_dim)
        edge_attr = self.fc_edge(data.edge_attr.to(device))

        edge_attr = self.be(edge_attr.to(device))

        # x = torch.cat([x, edge_attr], dim=-1)

        for conv in self.convs1:
            # x = conv(x,data.edge_index)
            x1 = conv(x, data.edge_index.to(device), edge_attr.to(device))
            x = x + x1

        x = x.reshape((batch_size, -1, self.hidden_node_dim))    #(batch, n_nodes, embed_dim)

        return x, torch.mean(x, dim = 1)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch, batch_steps, n_customer = 10, 1, 20
    # dataset = Generator(device, n_samples = batch*batch_steps,
    # 	n_car_each_depot = 15, n_depot = 2, n_customer = n_customer, capa = 2.)
    # cvrptw = Generator(device, n_samples=batch * batch_steps, n_car_each_depot=15, n_depot=2, n_customer=20, capa=1.,
    #                    service_time=10, depot_end=300, tw_width=10, seed=None)
    datas = cvrptw(device, size = 6,n_customer=20, seed = 1234,
		n_depot=3,n_car=4,service_window = 1000, service_duration = 10,
		time_factor = 100.0,tw_expansion = 3.0)
    dl = DataLoader(datas, batch_size=3)
    data = next(iter(dl))
    # tw = data.tw.view(3,-1,2)
    # car_cur_node = torch.range(0,2)
    # t = torch.gather(input=tw[:,:,1], dim=2, index=car_cur_node[:, :, None])
    print('data',data)


    # for i,data in enumerate(datas):
    #
    #     print(data)
    # # cvrptw = TWGenerator(device, n_samples=batch * batch_steps,
    # #             n_customer=10, seed=1234, n_depot=2, service_window=1000, service_duration=10,
    # #             time_factor=100.0, tw_expansion=3.0)
    # n_nodes = torch.cat((data['depot_loc'], data['node_loc']), 0)
    # tw = torch.cat((data['depot_tw'], data['node_tw']), 0)
    #
    # d = next(iter(cvrptw))
    # data = Data((x=d[]).float(), edge_index=edges_index,edge_attr=torch.from_numpy(edge).float(),
    #                 demand=torch.tensor(demand).unsqueeze(-1).float(),capcity=torch.tensor(capcity).unsqueeze(-1).float())
    # print('data', data)
    # dataset = DataLoader(cvrptw,3)

    # print('dataset', dataset)
    encoder = Encoder(input_node_dim=5, embed_dim=128, input_edge_dim=1, hidden_edge_dim=16, conv_layers=3)
    print('encoder', encoder)
    output = encoder(data)
    print('output.size()', output[0].size())