from typing import Union

import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.distributions import Categorical
from torch_geometric.data import Data, DataLoader
import sys
sys.path.append('../')
from dataset import cvrptw


ourlogzero = sys.float_info.min


class Env():
    def __init__(self, data, node_embeddings):
        super().__init__()
        """depot_xy: (batch, n_depot, 2)
			customer_xy: (batch, n_customer, 2)
			--> xy: (batch, n_node, 2); Coordinates of depot + customer nodes
			n_node= n_depot + n_customer
			demand: (batch, n_customer)
			??? --> demand: (batch, n_car, n_customer)
			D(remaining car capacity): (batch, n_car)
			node_embeddings: (batch, n_node, embed_dim)
			--> node_embeddings: (batch, n_car, n_node, embed_dim)

			car_start_node: (batch, n_car); start node index of each car
			car_cur_node: (batch, n_car); current node index of each car
			car_run: (batch, car); distance each car has run 
			pi: (batch, n_car, decoder_step); which index node each car has moved 
			dist_mat: (batch, n_node, n_node); distance matrix
			traversed_nodes: (batch, n_node)
			traversed_customer: (batch, n_customer)
		"""
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.batch, self.n_node, self.embed_dim = node_embeddings.size()
        self.n_depot = data.dloc.view(self.batch, -1, 2).size(1)
        self.demand = data.demand.view(self.batch, -1)[:, self.n_depot:].to(self.device)
        # print("self.demand",self.demand)
        self.xy = data.x.view(self.batch, -1, 2)
        self.tw = data.tw.view(self.batch, -1, 2).to(self.device)

        self.car_start_node, self.D = data.car_start_node.view(self.batch, -1).to(self.device), data.capcity.view(
            self.batch, -1).to(self.device)
        self.Q_max = 1 / (data.capcity.view(self.batch, -1).to(self.device)[0][0])
        self.car_cur_node = self.car_start_node

        self.pi = self.car_start_node.unsqueeze(-1)
        self.duration = data.durations.view(self.batch, -1).to(self.device)

        self.n_depot = data.dloc.view(self.batch, -1, 2).size(1)
        self.n_customer = data.nloc.view(self.batch, -1, 2).size(1)
        self.n_car = self.car_start_node.size(1)
        self.cur_time = torch.zeros((self.batch, self.n_car), dtype=torch.float, device=self.device)
        self.c_tw = self.tw[:, self.n_depot:]

        self.node_embeddings = node_embeddings[:, None, :, :].repeat(1, self.n_car, 1, 1)

        self.demand_include_depot = data.demand.view(self.batch, -1).to(self.device)
        assert self.demand_include_depot.size(1) == self.n_node, 'demand_include_depot'
        # self.demand = demand[:,None,:].repeat(1,self.n_car,1)
        self.car_run = torch.zeros((self.batch, self.n_car), dtype=torch.float, device=self.device)
        self.car_fuel = torch.zeros((self.batch, self.n_car), dtype=torch.float, device=self.device)
        self.time_cost = torch.zeros((self.batch, self.n_car), dtype=torch.float, device=self.device)
        self.dist_mat = self.build_dist_mat().to(self.device)
        self.mask_depot, self.mask_depot_unused = self.build_depot_mask()
        self.traversed_customer = torch.zeros((self.batch, self.n_customer), dtype=torch.bool, device=self.device)
        self._bidx = torch.arange(self.batch, device=self.device)
        self.opening_time_window_idx = self.tw[:, :, 0]
        self.closing_time_window_idx = self.tw[:, :, 1]
        self.durat = data.durations.view(self.batch, -1)

    def build_dist_mat(self):
        xy = self.xy.unsqueeze(1).repeat(1, self.n_node, 1, 1)
        const_xy = self.xy.unsqueeze(2).repeat(1, 1, self.n_node, 1)
        dist_mat = torch.sqrt(((xy - const_xy) ** 2).sum(dim=3))
        return dist_mat

    def build_depot_mask(self):
        a = torch.arange(self.n_depot, device=self.device).reshape(1, 1, -1).repeat(self.batch, self.n_car, 1)

        b = self.car_start_node[:, :, None].repeat(1, 1, self.n_depot)

        depot_one_hot = (a == b).bool()  # .long()

        return depot_one_hot, torch.logical_not(depot_one_hot)



    def get_mask(self, next_node, next_car):
        is_next_depot = (self.car_cur_node == self.car_start_node).bool()  # .long().sum(-1)


        new_traversed_node = torch.eye(self.n_node, device=self.device)[next_node.squeeze(1)]
        # new_traversed_node: (batch, node)
        new_traversed_customer = new_traversed_node[:, self.n_depot:]

        self.traversed_customer = self.traversed_customer | new_traversed_customer.bool()

        selected_demand = torch.gather(input=self.demand_include_depot, dim=1, index=next_node)

        selected_car = torch.eye(self.n_car, device=self.device)[next_car.squeeze(1)]

        car_used_demand = selected_car * selected_demand

        self.D -= car_used_demand

        arrival_time = self.cur_time[:, :, None]

        D_over_customer = self.demand[:, None, :].repeat(1, self.n_car, 1) > self.D[:, :, None].repeat(1, 1,
                                                                                                       self.n_customer)

        T_over_customer1 = arrival_time.repeat(1, 1, self.n_customer) > (
            self.tw[:, :, -1][:, None, :].expand(self.batch, self.n_car, self.n_node).gather(dim=-1,
                                                                                             index=self.car_cur_node[:,
                                                                                                   :, None]).repeat(1,
                                                                                                                    1,
                                                                                                                    self.n_customer))

        t_max = self.c_tw[:, :, -1][0][0]

        T_over_customer2 = arrival_time.repeat(1, 1, self.n_customer) > (
            t_max.repeat(1, self.n_car, self.n_customer)
        )

        mask_customer =  T_over_customer1 | T_over_customer2 | D_over_customer | self.traversed_customer[:, None,
                                                                                :].repeat(1, self.n_car, 1)
        #T_over_customer1 | T_over_customer2 |

        mask_depot = is_next_depot & (
                (mask_customer == False).long().sum(dim=2).sum(dim=1)[:, None].repeat(1, self.n_car) > 0)

        """mask_depot = True --> We cannot choose depot in the next step
            if 1) the vehicle is at the depot in the next step
            or 2) there is a customer node which has not been visited yet
        """

        mask_depot = self.mask_depot & mask_depot.bool()[:, :, None].repeat(1, 1, self.n_depot)

        mask_depot = self.mask_depot_unused | mask_depot

        return torch.cat([mask_depot, mask_customer], dim=-1).unsqueeze(-1)

    def get_mask_t1(self):
        """mask_depot: (batch, n_car, n_depot)
            mask_customer: (batch, n_car, n_customer)
            --> return mask: (batch, n_car, n_node ,1)
        """
        mask_depot_t1 = self.mask_depot | self.mask_depot_unused
        mask_customer_t1 = self.traversed_customer[:, None, :].repeat(1, self.n_car, 1)
        return torch.cat([mask_depot_t1, mask_customer_t1], dim=-1).unsqueeze(-1)

    def generate_step_context(self):
        """D: (batch, n_car)
			-->　D: (batch, n_car, 1, 1)

			each_car_idx: (batch, n_car, 1, embed_dim)
			node_embeddings: (batch, n_car, n_node, embed_dim)
			--> prev_embeddings(initially, depot_embeddings): (batch, n_car, 1, embed)
			node embeddings where car is located

			return step_context: (batch, n_car, 1, embed+1)
		"""
        each_car_idx = self.car_cur_node[:, :, None, None].repeat(1, 1, 1, self.embed_dim)
        prev_embeddings = torch.gather(input=self.node_embeddings, dim=2, index=each_car_idx)
        step_context = torch.cat([prev_embeddings, self.D[:, :, None, None], self.cur_time[:, :, None, None]], dim=-1)
        return step_context

    def _get_step(self, next_node, next_car):
        """next_node **includes depot** : (batch, 1) int(=long), range[0, n_node-1]

			return
			mask: (batch, n_car, n_node ,1)
			step_context: (batch, n_car, 1, embed+1)
		"""
        self.update_node_path(next_node, next_car)
        self.update_car_distance()
        self.update_car_fuel(next_node, next_car)
        self.update_car_time()
        mask = self.get_mask(next_node, next_car)

        step_context = self.generate_step_context()

        #  + PE[next_node].expand(self.batch, self.n_car, 1,self.embed_dim)
        # print('step_context',step_context.size())+PE[next_node-1].expand(self.batch, 1,self.embed_dim)
        return mask, step_context

    def _get_step_t1(self):
        """return
			mask: (batch, n_car, n_node ,1)
			step_context: (batch, n_car, 1, embed+1)
		"""
        mask_t1 = self.get_mask_t1()
        step_context_t1 = self.generate_step_context()
        return mask_t1, step_context_t1

    def update_node_path(self, next_node, next_car):
        self.car_prev_node = self.car_cur_node
        a = torch.arange(self.n_car, device=self.device).reshape(1, -1).repeat(self.batch, 1)
        b = next_car.reshape(self.batch, 1).repeat(1, self.n_car)
        mask_car = (a == b).long()
        new_node = next_node.reshape(self.batch, 1).repeat(1, self.n_car)

        self.car_cur_node = mask_car * new_node + (1 - mask_car) * self.car_cur_node
        self.pi = torch.cat([self.pi, self.car_cur_node.unsqueeze(-1)], dim=-1)

    def update_car_distance(self):
        # self.car_cur_node0 tensor([[0, 1, 43, 1, 0, 1, 23, 1, 56, 1],[0, 98, 70, 75, 0, 17, 0, 86, 0, 94]])
        prev_node_dist_vec = torch.gather(input=self.dist_mat, dim=1,
                                          index=self.car_prev_node[:, :, None].repeat(1, 1, self.n_node))

        dist = torch.gather(input=prev_node_dist_vec, dim=2, index=self.car_cur_node[:, :, None])

        self.car_run += dist.squeeze(-1)



    def update_car_time(self):

        end = self.tw[:, :, -1][:, None, :].expand(self.batch, self.n_car, self.n_node).gather(dim=-1,
                                                                                               index=self.car_cur_node[
                                                                                                     :, :, None])
        start = self.tw[:, :, 0][:, None, :].expand(self.batch, self.n_car, self.n_node).gather(dim=-1,
                                                                                                index=self.car_cur_node[
                                                                                                      :, :, None])
        service = self.duration[:, None, :].expand(self.batch, self.n_car, self.n_node).gather(dim=-1,
                                                                                               index=self.car_cur_node[
                                                                                                     :, :, None])
        # 计算车辆从上一个节点到当前节点的距离
        prev_node_dist_vec = torch.gather(input=self.dist_mat, dim=1,
                                          index=self.car_prev_node[:, :, None].repeat(1, 1, self.n_node))
        dist = torch.gather(input=prev_node_dist_vec, dim=2, index=self.car_cur_node[:, :, None])
        # 计算车辆到达当前节点的时间
        kk = self.cur_time
        arrival_time = kk[:, :, None] + dist + service + ((start - kk[:, :, None]).clamp(min=0))
        self.cur_time = arrival_time.squeeze(-1)
        time_cost = torch.abs((1 / 2 * (start - end)) - arrival_time)
        self.time_cost = time_cost.squeeze(-1)

    def update_car_fuel(self, next_node, next_car):
        """self.demand **excludes depot**: (batch, n_nodes-1)
			selected_demand: (batch, 1)
			if next node is depot, do not select demand
			self.D: (batch, n_car, 1), D denotes "remaining vehicle capacity"
			self.capacity_over_customer **excludes depot**: (batch, n_car, n_customer)
			visited_customer **excludes depot**: (batch, n_customer, 1)
			is_next_depot: (batch, 1), e.g. [[True], [True], ...]

		"""

        D = self.D
        dist = self.car_run
        bsz = D.shape[0]
        zero_to_bsz = torch.arange(bsz)
        for i in zero_to_bsz:
            car_fuel = self.car_fuel
            car_fuel[i] = torch.mul(dist[i], D[i])
        car_fuel = torch.stack([car_fuel[i]], 0)
        self.car_fuel += car_fuel

    def return_depot_all_car(self, next_node, next_car):
        self.pi = torch.cat([self.pi, self.car_start_node.unsqueeze(-1)], dim=-1)
        # .
        self.car_prev_node = self.car_cur_node
        self.car_cur_node = self.car_start_node
        self.update_car_distance()
        self.update_car_fuel(next_node, next_car)
        self.update_car_time()

    def get_log_likelihood(self, _log_p, _idx):
        """_log_p: (batch, decode_step, n_car * n_node)
			_idx: (batch, decode_step, 1), selected index
		"""
        log_p = torch.gather(input=_log_p, dim=2, index=_idx)
        return log_p.squeeze(-1).sum(dim=1)


class Sampler(nn.Module):
    """args; logits: (batch, n_car * n_nodes)
		return; next_node: (batch, 1)
		TopKSampler --> greedy; sample one with biggest probability
		CategoricalSampler --> sampling; randomly sample one from possible distribution based on probability
	"""

    def __init__(self, n_samples=1, **kwargs):
        super().__init__(**kwargs)
        self.n_samples = n_samples


class TopKSampler(Sampler):
    def forward(self, logits):
        # print("torch.topk(logits, self.n_samples, dim=1)[1]", torch.topk(logits, self.n_samples, dim=1)[1])
        return torch.topk(logits, self.n_samples, dim=1)[1]


# torch.argmax(logits, dim = 1).unsqueeze(-1)


class CategoricalSampler(Sampler):
    def forward(self, logits):
        return torch.multinomial(logits.exp(), self.n_samples)


class UCBSampler(Sampler):
    def forward(self, probs):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        ucb_coeff = torch.tensor(0.5).to(device)
        batch_size = probs.shape[0]
        n_nodes = probs.shape[1]
        rewards = torch.zeros(batch_size, n_nodes).to(device)
        n_selections = torch.ones(batch_size, n_nodes).to(device)
        r = []
        for i in range(batch_size):
            for j in range(n_nodes):
                rewards[i][j] = probs[i][j] + ucb_coeff * torch.sqrt(torch.log(torch.tensor(i + 1)) / (1 + torch.sum(n_selections[:i, j])))
                n_selections += 1

            node_idx = torch.argmax(rewards[i])
            r.append(node_idx)

        new_tensor = torch.stack([x.view(-1) for x in r])
        # print("new_tensor", new_tensor)
        return new_tensor




#
# class UCBSampler(Sampler):
#     def forward(self,probs):
#         device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#         ucb_coeff = torch.tensor(0.4).to(device)
#         batch = probs.shape[0]
#         n_nodes = probs.shape[1]
#         rewards = torch.zeros(n_nodes).to(device)
#         n_selections = torch.ones(n_nodes).to(device)
#
#         r = []
#         for i in range(probs.shape[0]):
#
#             rewards += probs[i]/ n_selections + ucb_coeff* torch.sqrt(torch.log(torch.tensor(i + 1)) / n_selections)
#
#             n_selections += 1
#             node_idx = torch.argmax(rewards)
#             r.append(node_idx)
#         new_tensor = torch.stack([x.view(-1) for x in r])
#
#
#
#         # selected_nodes = torch.argsort(rewards, descending=True)[:probs.shape[0]]
#         # print("selected_nodes", selected_nodes)
#
#         return new_tensor
#






if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch = 3
    embed_dim = 128
    n_node = 23
    # dataset = Generator(device, n_samples = batch*batch_steps,
    # 	n_car_each_depot = 15, n_depot = 2, n_customer = n_customer, capa = 2.)
    # cvrptw = Generator(device, n_samples=batch * batch_steps, n_car_each_depot=15, n_depot=2, n_customer=20, capa=1.,
    #                    service_time=10, depot_end=300, tw_width=10, seed=None)
    datas = cvrptw(device, size=6, n_customer=20, seed=1234,
                   n_depot=3, n_car=4, service_window=1000, service_duration=10,
                   time_factor=100.0, tw_expansion=3.0)
    dl = DataLoader(datas, batch_size=3)
    data = next(iter(dl))

    node_embeddings = torch.rand((batch, n_node, embed_dim), dtype=torch.float, device=device)
    graph_embedding = node_embeddings.mean(dim=1)
    encoder_output = (node_embeddings, graph_embedding)
    env = Env(data,node_embeddings)
    bm = env.build_dist_mat()
    print("bm",bm)
    print("env.tw",env.car_start_node.shape)
    selected_tw = torch.gather(input=env.tw, dim=1, index=torch.ones(3,1,2).type(torch.int64))
    print(" selected_tw",  selected_tw[:,:,0])
    next_node= torch.tensor([[5],[6],[7]])
    next_car = torch.tensor([[1],[2],[3]])
    mak = env.update_node_path(next_node, next_car)
    print("mak",env.cur_time)

#     self.tw[:, :, -1][:, None, :]
#     .expand(self.bs, self.n_car, self.n_node)
#     .gather(dim=-1, index=next_node)
# )