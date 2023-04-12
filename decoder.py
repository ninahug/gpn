import torch
import torch.nn as nn
import math
import numpy


import sys

# from regex import F
from torch_geometric.data import DataLoader

sys.path.append('../')
# from dataset import generate_data
from  dataset import  cvrptw
from decoder_layers import MultiHeadAttention, DotProductAttention
# from decoder_utils import TopKSampler, CategoricalSampler, Env
from decoder_env import TopKSampler, CategoricalSampler, Env, UCBSampler


# class Pointer(nn.Module):
#     def __init__(self, hidden_size):
#         super(Pointer, self).__init__()
#
#         self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
#         self.W2 = nn.Linear(hidden_size, hidden_size)
#         self.V = nn.Parameter(torch.zeros((hidden_size, 1), requires_grad=True))
#
#         self.first_h_0 = nn.Parameter(torch.FloatTensor(1, hidden_size), requires_grad=True)
#         self.first_h_0.data.uniform_(-(1. / math.sqrt(hidden_size)), 1. / math.sqrt(hidden_size))
#
#         self.c0 = nn.Parameter(torch.FloatTensor( 1, hidden_size),requires_grad=True)
#         self.c0.data.uniform_(-(1. / math.sqrt(hidden_size)), 1. / math.sqrt(hidden_size))
#
#         self.hidden_0 = (self.first_h_0, self.c0)
#
#         self.lstm = nn.LSTMCell(hidden_size, hidden_size)
#
#
#     def forward(self, input, hidden, enc_outputs, mask):
#
#         hidden = self.lstm(input, hidden)
#         w1e = self.W1(enc_outputs)
#         w2h = self.W2(hidden[0]).unsqueeze(1)
#         u = torch.tanh(w1e + w2h)
#         a = u.matmul(self.V)
#         a = 10*torch.tanh(a).squeeze(2)
#
#         policy = F.softmax(a + mask.float().log(), dim=1)
#
#         return policy, hidden


# class Pointer(nn.Module):
# 	def __init__(self, n_hidden):
# 		super(Pointer, self).__init__()
# 		self.size = 0
# 		self.batch_size = 0
# 		self.dim = n_hidden
#
# 		v = torch.FloatTensor(n_hidden).cuda()
# 		self.v = nn.Parameter(v)
# 		self.v.data.uniform_(-1 / math.sqrt(n_hidden), 1 / math.sqrt(n_hidden))
#
# 		# parameters for pointer attention
# 		self.Wref = nn.Linear(n_hidden, n_hidden)
# 		self.Wq = nn.Linear(n_hidden, n_hidden)
#
# 	def forward(self, q, ref):  # query and reference
# 		self.batch_size = q.size(0)
# 		self.size = int(ref.size(0) / self.batch_size)
# 		q = self.Wq(q)  # (B, dim)
# 		ref = self.Wref(ref)
# 		ref = ref.view(self.batch_size, self.size, self.dim)  # (B, size, dim)
#
# 		q_ex = q.unsqueeze(1).repeat(1, self.size, 1)  # (B, size, dim)
# 		# v_view: (B, dim, 1)
# 		v_view = self.v.unsqueeze(0).expand(self.batch_size, self.dim).unsqueeze(2)
#
# 		# (B, size, dim) * (B, dim, 1)
# 		u = torch.bmm(torch.tanh(q_ex + ref), v_view).squeeze(2)
#
# 		return u, ref

class DecoderCell(nn.Module):
	def __init__(self, embed_dim = 128, n_heads = 8, clip = 10., **kwargs):
		super().__init__(**kwargs)
		self.nb_heads = n_heads
		self.embed_dim = embed_dim
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.Wk1 = nn.Linear(embed_dim, embed_dim, bias = False)
		self.Wv = nn.Linear(embed_dim, embed_dim, bias = False)
		self.Wk2 = nn.Linear(embed_dim, embed_dim, bias = False)
		self.Wq_fixed = nn.Linear(embed_dim, embed_dim, bias = False)
		self.Wout = nn.Linear(embed_dim, embed_dim, bias = False)
		self.Wq_step = nn.Linear(embed_dim+2, embed_dim, bias = False)
		self.Wq_selfatt = nn.Linear(embed_dim, embed_dim)
		self.Wk_selfatt = nn.Linear(embed_dim, embed_dim)
		self.Wv_selfatt = nn.Linear(embed_dim, embed_dim)
		v = torch.FloatTensor(embed_dim)
		self.v = nn.Parameter(v)
		self.v.data.uniform_(-1 / math.sqrt(embed_dim), 1 / math.sqrt(embed_dim))

		self.BN_MLP = nn.LayerNorm(embed_dim)
		
		self.MHA = MultiHeadAttention(n_heads = n_heads, embed_dim = embed_dim, need_W = False)
		self.SHA = DotProductAttention(clip = clip, return_logits = True, head_depth = embed_dim)


		# SHA ==> Single Head Attention, because this layer n_heads = 1 which means no need to spilt heads


	def compute_static(self, node_embeddings, graph_embedding):
		Q_fixed = self.Wq_fixed(graph_embedding[:,None,:])
		self.n_node = node_embeddings.size(1)

		# self.Q_fixed = Q_fixed[:,None,:,:].repeat(1,self.env.n_car,1,1)
		K1 = self.Wk1(node_embeddings)
		# self.K1 = K1[:,None,:,:].repeat(1,self.env.n_car,1,1)
		V = self.Wv(node_embeddings)
		K2 = self.Wk2(node_embeddings)
		self.Q_fixed, self.K1, self.V, self.K2 = list(
			map(lambda x: x[:,None,:,:].repeat(1,self.env.n_car,1,1)
				, [Q_fixed, K1, V, K2]))
		
	def compute_dynamic(self, mask, step_context):
		Q_step = self.Wq_step(step_context)#torch.Size([batch_size, 1,embed_dim])
		self.batch_size = Q_step.size(0)

		Q1 = self.Q_fixed + Q_step
		q = Q_step.repeat(1, 1, self.n_node,1)#torch.Size([batch_size, 1,embed_dim])
		# Q1 = Q1.view(self.env.batch, -1, self.env.embed_dim)
		
		# Q1, self.K1, self.V, self.K2 = list(
		# 	map(lambda x: x.view(self.env.batch, -1, self.env.embed_dim)
		# 		, [Q1, self.K1, self.V, self.K2]))
		# a = self.SHA([Q1 , Q1 , None], mask = mask) #torch.Size([2, 10, 1, 102])

		Q2 = self.MHA([Q1, self.K1, self.V], mask = mask)
		Q2 = self.Wout(Q2)
		Q3 = Q2.squeeze(2)

		v_view = self.v.unsqueeze(0)
		v_view = v_view.unsqueeze(1).expand(self.batch_size, self.env.n_car,self.embed_dim).unsqueeze(3)


		# x = torch.tanh(self.Q_fixed + Q_step)

		u =  torch.matmul(torch.tanh(self.Q_fixed + Q_step), v_view).repeat(1,1,1,self.embed_dim)


		# self.Q_fixed = torch.Size([2, 10, 1, 128])
		logits = self.SHA([u, self.K2, None], mask = mask)

		# logits  = logits  + torch.relu(logits)

		# logits: (batch, n_car, 1, n_node)
		# print("logits",logits.shape)
		logits = logits.view(self.env.batch, -1)
		u = u.view(self.env.batch, -1)
		# print("logits",logits.size())
		return logits

	def forward(self, x, encoder_output, return_pi = False, decode_type = 'sampling'):
		device = self.device
		node_embeddings, graph_embedding = encoder_output
		batch, n_node, embed_dim = node_embeddings.size()
		ba, n_n = graph_embedding.size()
		# PE = self.generate_positional_encoding()

		graph_embedding = graph_embedding.to(device)
		self.env = Env(x, node_embeddings)
		self.compute_static(node_embeddings, graph_embedding)
		mask, step_context = self.env._get_step_t1()

		
		selecter = {'greedy': TopKSampler(), 'sampling': CategoricalSampler(), 'UCB': UCBSampler()}.get(decode_type, None)
		log_ps, tours, cars, idxs = [[] for _ in range(4)]

		def clear_route(arr):
			dst = []
			for i in range(len(arr) - 1):
				if arr[i] != arr[i + 1]:

					dst.append(arr[i])
			if len(dst) > 0:
				dst.append(dst[0])

			return dst

		def get_routes(pi):
			batch, n_car, decoder_step =pi.size()
			# Remove unneeded values
			routes =  [[] for i in range(batch)]
			route_num = []
			for batch_id in range(batch):
				pi_batch = pi[batch_id]
				# print("pi", pi)
				# print("pi_batch", pi_batch)
				for pi_of_each_car in pi_batch:
					# print("pi_of_each_car",pi_of_each_car)
					route = clear_route(pi_of_each_car.cpu().numpy())
					if len(route) > 0:
						routes[batch_id].append(route)
				route_n = len(routes[batch_id])
				route_num.append(route_n)
			return route_num
		for i in range(self.env.n_node * 10):
			logits = self.compute_dynamic(mask, step_context)## logits: (batch, n_car*n_node)

			log_p = torch.log_softmax(logits, dim = -1)

			idx = selecter(log_p)
			next_car = idx // self.env.n_node
			next_node = idx % self.env.n_node
			mask, step_context = self.env._get_step(next_node, next_car)
			# print('next_node[0]', next_node[0])
			# print('next_car[0]', next_car[0])
			tours.append(next_node)
			cars.append(next_car)
			idxs.append(idx)
			log_ps.append(log_p)	
			if self.env.traversed_customer.all():
				break

		assert self.env.traversed_customer.all(), f"not traversed all customer {self.env.traversed_customer} {self.env.D}"

		fuel_cost = self.env.car_fuel.sum(1)*57.3*self.env.Q_max
		print("fuel_cost", fuel_cost)
		print("self.env.Q_max", self.env.Q_max)
		pay_cost = self.env.car_run.sum(1)*6
		print("pay_cost", pay_cost)
		time_cost = self.env.time_cost.sum(1)*0.01
		print("time_cost", time_cost)

		print("numpy.array(get_routes(self.env.pi))", numpy.array(get_routes(self.env.pi)))
		cost = fuel_cost  + pay_cost +torch.from_numpy(numpy.array(get_routes(self.env.pi))).to(device)*20+time_cost

		cost1 = fuel_cost*57.3*self.env.Q_max

		
		"""
		print straightforward pi and selected vehicle 
		select_nodes = torch.stack(tours, 1)
		select_cars = torch.stack(cars, 1)
		print(f'select_nodes.size():{select_nodes.size()}\nselect_cars.size():{select_cars.size()}')
		print('select_nodes[0]:', select_nodes[0])
		print('select_cars[0]:', select_cars[0])
		"""

		_idx  = torch.stack(idxs, 1)
		_log_p = torch.stack(log_ps, 1)
		ll = self.env.get_log_likelihood(_log_p, _idx)
		
		if return_pi:
			return cost, ll, self.env.pi
		return cost, ll

if __name__ == '__main__':

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	batch = 3
	embed_dim = 128
	n_node = 23

	datas = cvrptw(size=6, n_customer=20, seed=1234,
				   n_depot=3, n_car=4, service_window=1000, service_duration=10,
				   time_factor=100.0, tw_expansion=3.0)
	dl = DataLoader(datas, batch_size=3)
	data = next(iter(dl))

	node_embeddings = torch.rand((batch, n_node, embed_dim), dtype=torch.float, device=device)
	graph_embedding = node_embeddings.mean(dim=1)
	encoder_output = (node_embeddings, graph_embedding)
	env = Env(data, node_embeddings)
	decoder = DecoderCell(embed_dim, n_heads=8, clip=10.).to(device)


	decoder.train()
	return_pi = True
	output = decoder(data, encoder_output, return_pi = return_pi, decode_type = 'UCB')
	output1 = decoder(data, encoder_output, return_pi=return_pi, decode_type='greedy')
	output2 = decoder(data, encoder_output, return_pi=return_pi, decode_type='sampling')
	if return_pi:
		"""cost: (batch)
			ll: (batch)
			pi: (batch, n_car, decode_step)
		"""
		cost, ll, pi = output
		cost1, ll1, pi1 = output1
		cost2, ll2, pi2 = output2
		print('\ncost: ', cost.size(), cost)
		print('\ncost1: ', cost1.size(), cost1)
		print('\ncost2: ', cost2.size(), cost2)
		print('\nll: ', ll.size(), ll)
		print('\npi: ', pi.size(), pi)
	else:
		cost, ll = output
		print('\ncost: ', cost.size(), cost)
		print('\nll: ', ll.size(), ll)

	cnt = 0
	for k, v in decoder.state_dict().items():
		print(k, v.size(), torch.numel(v))
		cnt += torch.numel(v)
	print(cnt)

	# ll.mean().backward()
	# print(decoder.Wk1.weight.grad)
	# https://discuss.pytorch.org/t/model-param-grad-is-none-how-to-debug/52634	
