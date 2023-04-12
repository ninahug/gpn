import torch
import torch.nn as nn
import math
import copy
import torch.nn.functional as F

#
# def attention(x, mask=None, dropout=None):
# 	"Compute 'Scaled Dot Product Attention'"
# 	query, key, value = x
# 	d_k = query.size(-1)
# 	scores = torch.matmul(query, key.transpose(-2, -1)) \
# 			 / math.sqrt(d_k)
# 	if mask is not None:
# 		scores = scores.masked_fill(mask == 0, -1e9)
# 	p_attn = F.softmax(scores, dim=-1)
# 	if dropout is not None:
# 		p_attn = dropout(p_attn)
# 	return torch.matmul(p_attn, value), p_attn


# def clones(module, N):
# 	"Produce N identical layers."
# 	return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
#
#
# class MultiHeadedDotAttention(nn.Module):
# 	def __init__(self, h = 8, d_model =128, dropout=0.1, scale=1, project_k_v=1, use_output_layer=1, do_aoa=0, norm_q=0,
# 				 dropout_aoa=0.3):
# 		super(MultiHeadedDotAttention, self).__init__()
# 		assert d_model * scale % h == 0
# 		# We assume d_v always equals d_k
# 		self.d_k = d_model * scale // h
# 		self.h = h
#
# 		# Do we need to do linear projections on K and V?
# 		self.project_k_v = project_k_v
#
# 		# normalize the query?
# 		if norm_q:
# 			self.norm = nn.LayerNorm(d_model)
# 		else:
# 			self.norm = lambda x: x
# 		self.linears = clones(nn.Linear(d_model, d_model * scale), 1 + 2 * project_k_v)
#
# 		# output linear layer after the multi-head attention?
# 		self.output_layer = nn.Linear(d_model * scale, d_model)
#
# 		# apply aoa after attention?
# 		self.use_aoa = do_aoa
# 		if self.use_aoa:
# 			self.aoa_layer = nn.Sequential(nn.Linear((1 + scale) * d_model, 2 * d_model), nn.GLU())
# 			# dropout to the input of AoA layer
# 			if dropout_aoa > 0:
# 				self.dropout_aoa = nn.Dropout(p=dropout_aoa)
# 			else:
# 				self.dropout_aoa = lambda x: x
#
# 		if self.use_aoa or not use_output_layer:
# 			# AoA doesn't need the output linear layer
# 			del self.output_layer
# 			self.output_layer = lambda x: x
#
# 		self.attn = None
# 		self.dropout = nn.Dropout(p=dropout)
#
# 	def forward(self, x, mask=None):
# 		query, value, key = x
# 		if mask is not None:
# 			if len(mask.size()) == 2:
# 				mask = mask.unsqueeze(-2)
# 			# Same mask applied to all h heads.
# 			mask = mask.unsqueeze(1)
#
# 		single_query = 0
# 		if len(query.size()) == 2:
# 			single_query = 1
# 			query = query.unsqueeze(1)
#
# 		nbatches = query.size(0)
#
# 		query = self.norm(query)
#
# 		# Do all the linear projections in batch from d_model => h x d_k
# 		if self.project_k_v == 0:
# 			query_ = self.linears[0](query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
# 			key_ = key.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
# 			value_ = value.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
# 		else:
# 			query_, key_, value_ = \
# 				[l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
# 				 for l, x in zip(self.linears, (query, key, value))]
#
# 		# Apply attention on all the projected vectors in batch.
# 		x, self.attn = attention([query_, key_, value_], mask=mask,
# 								 dropout=self.dropout)
#
# 		# "Concat" using a view
# 		x = x.transpose(1, 2).contiguous() \
# 			.view(nbatches, -1, self.h * self.d_k)
#
# 		if self.use_aoa:
# 			# Apply AoA
# 			x = self.aoa_layer(self.dropout_aoa(torch.cat([x, query], -1)))
# 		x = self.output_layer(x)
#
# 		if single_query:
# 			query = query.squeeze(1)
# 			x = x.squeeze(1)
# 		return x
#
# class AoA_Decoder_Core(nn.Module):
#     def __init__(self, drop_prob_lm=0.1,rnn_size=128,use_multi_head=8,multi_head_scale=16,use_ctx_drop=0):
#
#         super(AoA_Decoder_Core, self).__init__()
#         self.drop_prob_lm = drop_prob_lm
#         self.d_model = rnn_size
#         self.use_multi_head = use_multi_head
#         self.multi_head_scale = multi_head_scale
#         self.use_ctx_drop = getattr('ctx_drop', 0)
#         # self.out_res = getattr(opt, 'out_res', 0)
#         self.decoder_type = getattr('decoder_type', 'AoA')
#         self.att_lstm = nn.LSTMCell(rnn_size+ rnn_size, rnn_size) # we, fc, h^2_t-1
#         self.out_drop = nn.Dropout(drop_prob_lm)
#
#         if self.decoder_type == 'AoA':
#             # AoA layer
#             self.att2ctx = nn.Sequential(nn.Linear(self.d_model * multi_head_scale + rnn_size, 2 * rnn_size), nn.GLU())
#         elif self.decoder_type == 'LSTM':
#             # LSTM layer
#             self.att2ctx = nn.LSTMCell(self.d_model * multi_head_scale + rnn_size, rnn_size)
#         else:
#             # Base linear layer
#             self.att2ctx = nn.Sequential(nn.Linear(self.d_model * multi_head_scale + rnn_size, rnn_size), nn.ReLU())
#
#         # if opt.use_multi_head == 1: # TODO, not implemented for now
#         #     self.attention = MultiHeadedAddAttention(opt.num_heads, opt.d_model, scale=opt.multi_head_scale)
#         # if opt.use_multi_head == 2:
#         self.attention = MultiHeadedDotAttention(num_heads=8, rnn_size=128, project_k_v=0, scale=multi_head_scale, use_output_layer=0, do_aoa=0, norm_q=1)
#         # else:
#         #     self.attention = Attention(opt)
#
#         if self.use_ctx_drop:
#             self.ctx_drop = nn.Dropout(self.drop_prob_lm)
#         else:
#             self.ctx_drop = lambda x :x
#
#     def forward(self, xt, mean_feats, att_feats, p_att_feats, state, att_masks=None):
#         # state[0][1] is the context vector at the last step
#         h_att, c_att = self.att_lstm(torch.cat([xt, mean_feats + self.ctx_drop(state[0][1])], 1), (state[0][0], state[1][0]))
#
#         if self.use_multi_head == 2:
#             att = self.attention(h_att, p_att_feats.narrow(2, 0, self.multi_head_scale * self.d_model), p_att_feats.narrow(2, self.multi_head_scale * self.d_model, self.multi_head_scale * self.d_model), att_masks)
#         else:
#             att = self.attention(h_att, att_feats, p_att_feats, att_masks)
#
#         ctx_input = torch.cat([att, h_att], 1)
#         if self.decoder_type == 'LSTM':
#             output, c_logic = self.att2ctx(ctx_input, (state[0][1], state[1][1]))
#             state = (torch.stack((h_att, output)), torch.stack((c_att, c_logic)))
#         else:
#             output = self.att2ctx(ctx_input)
#             # save the context vector to state[0][1]
#             state = (torch.stack((h_att, output)), torch.stack((c_att, state[1][1])))
#
#         # if self.out_res:
#         #     # add residual connection
#         #     output = output + h_att
#
#         output = self.out_drop(output)
#         return output, state

class DotProductAttention(nn.Module):
	def __init__(self, clip = None, return_logits = False, head_depth = 16, **kwargs):
		super().__init__(**kwargs)
		self.clip = clip
		self.return_logits = return_logits
		self.inf = 1e+10 
		self.scale = math.sqrt(head_depth)
		self.tanh = nn.Tanh

	def get_logits(self, x, mask=None):
		Q, K, V = x
		logits = torch.matmul(Q, K.transpose(-1, -2)) / self.scale

		if self.clip is not None:
			logits = self.clip * torch.tanh(logits)
		# logits = self.clip * self.tanh(logits)

		if self.return_logits:
			if mask is not None:
				# ~ print('mask.size():', mask.size())
				# ~ print('logits.size():', logits.size())
				return logits.masked_fill(mask.transpose(-1, -2) == True, -self.inf)
			return logits

		if mask is not None:
			# ~ print('mask.size():', mask.size())
			# ~ print('logits.size():', logits.size())
			# print('mask: ', mask[:,None,:,:].squeeze(-1).repeat(1,logits.size(1),1,mask.size(1)).size())
			# print('mask expand:', mask[:,None,:,:,:].repeat(1,logits.size(1),1,1,1).transpose(-1,-2).size())
			# logits = logits.masked_fill(mask[:,None,None,:,0].repeat(1,logits.size(1),1,1) == True, -self.inf)
			# logits = logits.masked_fill(mask[:,None,:,:].squeeze(-1).repeat(1,logits.size(1),1,mask.size(1)) == True, -self.inf)
			logits = logits.masked_fill(
				mask[:, None, :, :, :].repeat(1, logits.size(1), 1, 1, 1).transpose(-1, -2) == True, -self.inf)

		probs = torch.softmax(logits, dim=-1)  # [2, 8, 10, 1, 102]   V torch.Size([2, 8, 10, 1, 16])
		return 	torch.matmul(probs,V)
	def forward(self, x, mask = None):
		""" Q: (batch, n_heads, q_seq(=n_nodes or =1), head_depth)
			K: (batch, n_heads, k_seq(=n_nodes), head_depth)
			logits: (batch, n_heads, q_seq(this could be 1), k_seq)
			mask: (batch, n_nodes, 1), e.g. tf.Tensor([[ True], [ True], [False]])
			mask[:,None,None,:,0]: (batch, 1, 1, n_nodes) ==> broadcast depending on logits shape
			[True] -> [1 * -np.inf], [False] -> [logits]
			K.transpose(-1,-2).size() == K.permute(0,1,-1,-2).size()
		"""
		Q, K, V = x
		logits = torch.matmul(Q, K.transpose(-1,-2)) / self.scale

		if self.clip is not None:
			logits = self.clip * torch.tanh(logits)
			# logits = self.clip * self.tanh(logits)
			
		if self.return_logits:
			if mask is not None:
				# ~ print('mask.size():', mask.size())
				# ~ print('logits.size():', logits.size())
				return logits.masked_fill(mask.transpose(-1,-2) == True, -self.inf)
			return logits

		if mask is not None:
			# ~ print('mask.size():', mask.size())
			# ~ print('logits.size():', logits.size())
			# print('mask: ', mask[:,None,:,:].squeeze(-1).repeat(1,logits.size(1),1,mask.size(1)).size())
			# print('mask expand:', mask[:,None,:,:,:].repeat(1,logits.size(1),1,1,1).transpose(-1,-2).size())
			# logits = logits.masked_fill(mask[:,None,None,:,0].repeat(1,logits.size(1),1,1) == True, -self.inf)
			# logits = logits.masked_fill(mask[:,None,:,:].squeeze(-1).repeat(1,logits.size(1),1,mask.size(1)) == True, -self.inf)
			logits = logits.masked_fill(mask[:,None,:,:,:].repeat(1,logits.size(1),1,1,1).transpose(-1,-2) == True, -self.inf)
		probs = torch.softmax(logits, dim = -1)#[2, 8, 10, 1, 102]   V torch.Size([2, 8, 10, 1, 16])
		return torch.matmul(probs, V)

class MultiHeadAttention(nn.Module):
	def __init__(self, n_heads = 8, embed_dim = 128, clip = None, return_logits = None, need_W = None):
		super().__init__()
		self.n_heads = n_heads
		self.embed_dim = embed_dim
		self.head_depth = self.embed_dim // self.n_heads
		if self.embed_dim % self.n_heads != 0:
			raise ValueError("embed_dim = n_heads * head_depth")
		
		self.need_W = need_W 
		self.attention = DotProductAttention(clip = clip, return_logits = return_logits, head_depth = self.head_depth)
		if self.need_W:
			self.Wk = nn.Linear(embed_dim, embed_dim, bias = False)
			self.Wv = nn.Linear(embed_dim, embed_dim, bias = False)
			self.Wq = nn.Linear(embed_dim, embed_dim, bias = False)
			self.Wout = nn.Linear(embed_dim, embed_dim, bias = False)
		self.init_parameters()
	
	def init_parameters(self):
		for name, param in self.named_parameters():
			if name == 'Wout.weight':
				stdv = 1. / math.sqrt(param.size(-1))
			elif name in ['Wk.weight', 'Wv.weight', 'Wq.weight']:
				stdv = 1. / math.sqrt(self.head_depth)
			else:
				raise ValueError
			param.data.uniform_(-stdv, stdv)

	def split_heads(self, T):
		""" https://qiita.com/halhorn/items/c91497522be27bde17ce
			T: (batch, n_car, n_nodes(or 1), self.embed_dim)
			T reshaped: (batch, n_car, n_nodes(or 1), self.n_heads, self.head_depth)
			return: (batch, self.n_heads, n_car, n_nodes(or 1), self.head_depth)
			
			https://raishi12.hatenablog.com/entry/2020/04/20/221905
		"""
		shape = T.size()[:-1] + (self.n_heads, self.head_depth)
		T = T.view(*shape)
		return T.permute(0,3,1,2,4)

	def combine_heads(self, T):
		""" T: (batch, self.n_heads, n_car, n_nodes(or 1), self.head_depth)
			T transposed: (batch, n_car, n_nodes(or 1), self.n_heads, self.head_depth)
			return: (batch, n_car, n_nodes(or 1), self.embed_dim)
		"""
		T = T.permute(0,2,3,1,4).contiguous()
		shape = T.size()[:3] + (self.embed_dim, )
		return T.view(*shape)

	def forward(self, x, mask = None):
		"""	q, k, v = x
			encoder arg x: [x, x, x]
			shape of q: (batch, n_nodes, embed_dim)
			output[0] - output[h_heads-1]: (batch, n_nodes, head_depth)
			--> concat output: (batch, n_nodes, head_depth * h_heads)
			return output: (batch, n_nodes, embed_dim)
		"""
		Q, K, V = x
		if self.need_W:
			Q, K, V = self.Wq(Q), self.Wk(K), self.Wv(V)
		Q, K, V = list(map(self.split_heads, [Q, K, V]))
		output = self.attention([Q, K, V], mask = mask)
		# output = torch.matmul(probs, V)
		output = self.combine_heads(output)
		if self.need_W:
			return self.Wout(output)
		return output

if __name__ == '__main__':
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	mha = MultiHeadAttention(n_heads = 8, embed_dim = 128, need_W = False)
	batch, n_car, n_nodes, embed_dim = 5, 15, 21, 128
	mask = torch.zeros((batch, n_car, n_nodes, 1), dtype = torch.bool, device = device)
	# mask = None
	Q = torch.randn((batch, n_car, 1, embed_dim), dtype = torch.float, device = device)
	K = torch.randn((batch, n_car, n_nodes, embed_dim), dtype = torch.float, device = device)
	V = torch.randn((batch, n_car, n_nodes, embed_dim), dtype = torch.float, device = device)
	output = mha([Q,K,V], mask = mask)
	print('output.size()', output.size())# (batch, n_car, 1, embed_dim)

	sha = DotProductAttention(clip = 10., return_logits = True, head_depth = embed_dim)
	output = sha([output,K,None], mask = mask)
	print('output.size()', output.size())
	output = output.squeeze(2).view(batch, -1)
	print('output.size()', output.size())

	logp = torch.log_softmax(output, dim = -1)	
	# probs = torch.softmax(output, dim = 1)
	idx = torch.multinomial(logp.exp(), num_samples = 1)	
	print(idx)
	print('next car:', idx // n_nodes)
	print('next node:', idx % n_nodes)
	# print(a.size(), a[0])
	# idx = torch.argmax(a, dim = 1)
	# print(idx.size(), idx)


