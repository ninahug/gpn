import torch
import torch.nn as nn

import sys
sys.path.append('../')
from dataset import  cvrptw
# from encoder import GraphAttentionEncoder
# from decoder import DecoderCell

from .encoder_gat import Encoder
from .decoder import DecoderCell
import torch.nn.functional as F


class AttentionModel(nn.Module):
	
    def __init__(self, embed_dim = 128, n_encode_layers = 3, n_heads = 8, tanh_clipping = 10., FF_hidden = 512):
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.Encoder = Encoder(input_node_dim=5,embed_dim=128, input_edge_dim=1, hidden_edge_dim=16, conv_layers=3).to(self.device)
        self.Decoder = DecoderCell(embed_dim, n_heads, tanh_clipping).to(self.device)


    def forward(self, x, return_pi = False, decode_type = 'greedy'):

        encoder_output = self.Encoder(x)
        decoder_output = self.Decoder(x, encoder_output, return_pi = return_pi, decode_type = decode_type)
        if return_pi:
            cost, ll, pi = decoder_output
            return cost, ll, pi
        cost, ll = decoder_output
        return cost, ll
		
# if __name__ == '__main__':

	# critic = StateCritic()
	# model = AttentionModel()
	# critic1 = CriticNetwork()
	# critic.train()
	# model.train()
	# critic1.train()
	# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	# data =  cvrptw(device, size=6, n_customer=20, seed=1234,
	# 			   n_depot=3, n_car=4, service_window=1000, service_duration=10,
	# 			   time_factor=100.0, tw_expansion=3.0)
	# return_pi = False
	# output = model(data, decode_type = 'sampling', return_pi = return_pi)
	# critic_est = critic(data)
	# critic_est1 = critic1(data)
	# advantage = output-critic_est
	# print("critic_est",critic_est.view(-1))
	# print("critic_est1", critic_est1.view(-1))
	# print("output", output)
	# if return_pi:
	# 	cost, ll, pi = output
	# 	print('\ncost: ', cost.size(), cost)
	# 	print('\nll: ', ll.size(), ll)
	# 	print('\npi: ', pi.size(), pi)
	# else:
	# 	print('cost',output[0])# cost: (batch)
	# 	print('ll',output[1])# ll: (batch)
    #
	# cnt = 0
	# for k, v in model.state_dict().items():
	# 	print(k, v.size(), torch.numel(v))
	# 	cnt += torch.numel(v)
	# print('total parameters:', cnt)

	# output[1].mean().backward()
	# print('grad: ', model.Decoder.Wk1.weight.grad[0][0])
	# https://github.com/wouterkool/attention-learn-to-route/blob/master/train.py