import torch
import torch.nn as nn
import math
# from torchsummary import summary

from encoder_layers import MultiHeadAttention
import sys

# from Torch.Nets.TransformerModel import attention, clones

sys.path.append('../')
from dataset import generate_data
# from .AoAModel import MultiHeadedDotAttentionMultiHeadAttention
#
# from .encoder_layers import MultiHeadAttention




class Normalization(nn.Module):

	def __init__(self, embed_dim, normalization = 'batch'):
		super().__init__()

		normalizer_class = {
			'batch': nn.BatchNorm1d,
			'instance': nn.InstanceNorm1d}.get(normalization, None)
		self.normalizer = normalizer_class(embed_dim, affine=True)
		# Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
	# 	self.init_parameters()

	def init_parameters(self):
		for name, param in self.named_parameters():
			stdv = 1. / math.sqrt(param.size(-1))
			# print('param',param)
			param.data.uniform_(-stdv, stdv)

	def forward(self, x):

		if isinstance(self.normalizer, nn.BatchNorm1d):
			# (batch, num_features)
			# https://discuss.pytorch.org/t/batch-normalization-of-linear-layers/20989
			return self.normalizer(x.view(-1, x.size(-1))).view(*x.size())
		
		elif isinstance(self.normalizer, nn.InstanceNorm1d):
			return self.normalizer(x.permute(0, 2, 1)).permute(0, 2, 1)
		else:
			assert self.normalizer is None, "Unknown normalizer type"
			return x


class ResidualBlock_BN(nn.Module):
	def __init__(self, MHA, BN, **kwargs):
		super().__init__(**kwargs)
		self.MHA = MHA
		self.BN = BN

	def forward(self, x, mask = None):
		if mask is None:
			return self.BN(x + self.MHA(x))
		return self.BN(x + self.MHA(x, mask))

class SelfAttention(nn.Module):
	def __init__(self, MHA, **kwargs):
		super().__init__(**kwargs)
		self.MHA = MHA

	def forward(self, x, mask = None):
		return self.MHA([x, x, x], mask = mask)

class EncoderLayer(nn.Module):
	# nn.Sequential):
	def __init__(self, n_heads = 8, FF_hidden = 512, embed_dim = 128, **kwargs):
		super().__init__(**kwargs)
		self.n_heads = n_heads
		self.FF_hidden = FF_hidden
		self.BN1 = Normalization(embed_dim, normalization = 'batch')
		self.BN2 = Normalization(embed_dim, normalization = 'batch')

		self.MHA_sublayer = ResidualBlock_BN(
				SelfAttention(
					MultiHeadAttention(n_heads = self.n_heads, embed_dim = embed_dim, need_W = True)
		            # MultiHeadedDotAttention(n_heads, embed_dim, dropout=0.1, scale=1, project_k_v=1, use_output_layer=1, do_aoa=0, norm_q=0,dropout_aoa=0.3)
				),
			self.BN1
			) #

		self.FF_sublayer = ResidualBlock_BN(
			nn.Sequential(
					nn.Linear(embed_dim, FF_hidden, bias = True),
					nn.ReLU(),
					nn.Linear(FF_hidden, embed_dim, bias = True)
			),
			self.BN2
		)  #self.BN2(self.BN1(x + self.MHA(x, mask)) + nn.Sequential(self.BN1(x + self.MHA(x, mask)))
		
	def forward(self, x, mask=None):
		"""	arg x: (batch, n_nodes, embed_dim)
			return: (batch, n_nodes, embed_dim)
		"""
		return self.FF_sublayer(self.MHA_sublayer(x, mask = mask))
		
class GraphAttentionEncoder(nn.Module):
	def __init__(self, embed_dim = 128, n_heads = 8, n_layers = 3, FF_hidden = 512):
		super().__init__()
		device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.init_W_depot = nn.Linear(2, embed_dim, bias = True).to(device)
		self.init_W = nn.Linear(3, embed_dim, bias = True).to(device)
		self.encoder_layers = nn.ModuleList([EncoderLayer(n_heads, FF_hidden, embed_dim) for _ in range(n_layers)]).to(device)
	
	def forward(self, x, mask = None):
		"""x[0] -- depot_xy: (batch, n_depot, 2) --> embed_depot_xy: (batch, n_depot, embed_dim)
			x[1] -- customer_xy: (batch, n_nodes-n_depot, 2)
			x[2] -- demand: (batch, n_nodes-n_depot)
			--> concated_customer_feature: (batch, n_nodes-n_depot, 3) --> embed_customer_feature: (batch, n_nodes-n_depot, embed_dim)
			embed_x(batch, n_nodes, embed_dim)

			return: (node embeddings(= embedding for all nodes), graph embedding(= mean of node embeddings for graph))
				=((batch, n_nodes, embed_dim), (batch, embed_dim))
		"""
		device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		embed_depot = self.init_W_depot(x['depot_xy']).to(device)
		embed_customer = self.init_W(torch.cat([x['customer_xy'], x['demand'][:,:,None]], dim = -1)).to(device)
		z = torch.cat([embed_depot, embed_customer], dim = 1)
	
		for layer in self.encoder_layers:
			z = layer(z, mask)

		return (z, torch.mean(z, dim = 1))


if __name__ == '__main__':
	batch, n_car, n_depot, n_customer, n_node = 5, 15, 2, 20, 22
	assert n_node == n_depot + n_customer
	encoder = GraphAttentionEncoder(n_layers = 1)
	# data = generate_data(batch = batch, n_customer = n_nodes-1)
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	data = generate_data(device, batch = batch, n_car_each_depot = n_car, n_depot = n_depot, n_customer = n_customer)
	# mask = torch.zeros((batch, n_nodes, 1), dtype = bool)
	output = encoder(data, mask = None)

	print('output[0].shape:', output[0].size())
	print('output[1].shape', output[1].size())
	
	# summary(encoder, [(2), (20,2), (20)])
	cnt = 0
	for i, k in encoder.state_dict().items():
		print(i, k.size(), torch.numel(k))
		cnt += torch.numel(k)
	print(cnt)

	# output[0].mean().backward()
	# print(encoder.init_W_depot.weight.grad)

