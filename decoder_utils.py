import torch
import torch.nn as nn


class Env():
    def __init__(self, x, node_embeddings):
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
        self.demand = x['demand']
        self.xy = torch.cat([x['depot_xy'], x['customer_xy']], 1)
        self.car_start_node, self.D = x['car_start_node'], x['car_capacity']
        self.car_cur_node = self.car_start_node
        self.pi = self.car_start_node.unsqueeze(-1)

        self.n_depot = x['depot_xy'].size(1)
        self.n_customer = x['customer_xy'].size(1)
        self.n_car = self.car_start_node.size(1)
        self.batch, self.n_node, self.embed_dim = node_embeddings.size()
        self.node_embeddings = node_embeddings[:, None, :, :].repeat(1, self.n_car, 1, 1)

        self.demand_include_depot = torch.cat(
            [torch.zeros((self.batch, self.n_depot), dtype=torch.float, device=self.device), self.demand], dim=1)
        assert self.demand_include_depot.size(1) == self.n_node, 'demand_include_depot'
        # self.demand = demand[:,None,:].repeat(1,self.n_car,1)
        self.car_run = torch.zeros((self.batch, self.n_car), dtype=torch.float, device=self.device)
        self.car_fuel = torch.zeros((self.batch, self.n_car), dtype=torch.float, device=self.device)
        self.dist_mat = self.build_dist_mat()
        self.mask_depot, self.mask_depot_unused = self.build_depot_mask()
        self.traversed_customer = torch.zeros((self.batch, self.n_customer), dtype=torch.bool, device=self.device)
        self._bidx = torch.arange(self.batch, device=self.device)
        self.cur_time = torch.zeros(self.batch, self.n_car, device=self.device)
        self.cur_time_to_depot = torch.zeros(self.batch, self.n_car, device=self.device)

    def build_dist_mat(self):
        xy = self.xy.unsqueeze(1).repeat(1, self.n_node, 1, 1)
        const_xy = self.xy.unsqueeze(2).repeat(1, 1, self.n_node, 1)
        dist_mat = torch.sqrt(((xy - const_xy) ** 2).sum(dim=3))

        return dist_mat

    def build_depot_mask(self):
        a = torch.arange(self.n_depot, device=self.device).reshape(1, 1, -1).repeat(self.batch, self.n_car, 1)
        # print('a',a)
        b = self.car_start_node[:, :, None].repeat(1, 1, self.n_depot)
        # print('b', b)
        depot_one_hot = (a == b).bool()  # .long()
        # print('depot_one_hot',depot_one_hot)
        # print('torch.logical_not(depot_one_hot)', torch.logical_not(depot_one_hot))
        return depot_one_hot, torch.logical_not(depot_one_hot)


    def get_mask(self, next_node, next_car):
        """self.demand **excludes depot**: (batch, n_nodes-1)
			selected_demand: (batch, 1)
			if next node is depot, do not select demand
			self.D: (batch, n_car, 1), D denotes "remaining vehicle capacity"
			self.capacity_over_customer **excludes depot**: (batch, n_car, n_customer)
			visited_customer **excludes depot**: (batch, n_customer, 1)
			is_next_depot: (batch, 1), e.g. [[True], [True], ...]
		"""
        previous_node = self.car_prev_node[self._bidx,next_car.squeeze(1)]
        print("self._bidx",self._bidx)
        print("next_car", next_car.squeeze(1))
        print("previous_node",previous_node)
        dist_mat = self.build_dist_mat()
        print("dist_mat", dist_mat.shape)
        print("next_node", (next_node[:,:,None]).shape)
        dist_mat = self.build_dist_mat()
        cur_time_delta = dist_mat[self._bidx, previous_node, next_node.squeeze(1)]
        print("cur_time_delta", cur_time_delta)

        arrival_time = self.cur_time[self._bidx, next_car] + cur_time_delta
        # tm = 0
        # tm += self.dist_mat[bs_idx, prev, nxt]
        # # add waiting time and service time
        # tm += ((self.tw[bs_idx, nxt][0] - tm).clamp_(min=0) + service_time)
        is_next_depot = (self.car_cur_node == self.car_start_node).bool()  # .long().sum(-1)
        # e.g., is_next_depot = next_node == 0 or next_node == 1
        # is_next_depot: (batch, n_car), e.g. [[True], [True], ...]

        new_traversed_node = torch.eye(self.n_node, device=self.device)[next_node.squeeze(1)]
        # new_traversed_node: (batch, node)
        new_traversed_customer = new_traversed_node[:, self.n_depot:]
        # new_traversed_customer: (batch, n_customer)
        self.traversed_customer = self.traversed_customer | new_traversed_customer.bool()

        # traversed_customer: (batch, n_customer) self.demand_include_depot
        selected_demand = torch.gather(input=self.demand_include_depot, dim=1, index=next_node)
        # selected_demand: (batch, 1)
        selected_car = torch.eye(self.n_car, device=self.device)[next_car.squeeze(1)]
        # selected_car: (batch, n_car)
        car_used_demand = selected_car * selected_demand
        # idx_pair = torch.stack((self.car_cur_node, next_node), dim=0)
        print("self.car_cur_node",self.car_cur_node)
        # self.D -= car_used_demand car_used_demand: (batch, n_car)
        self.D -= car_used_demand
        # D: (batch, n_car)
        # self.D = torch.clamp(self.D, min = 0.)
        D_over_customer = self.demand[:, None, :].repeat(1, self.n_car, 1) > self.D[:, :, None].repeat(1, 1,
                                                                                                       self.n_customer)
        d = self.demand[:, None, :].expand(self.batch, self.n_car, -1)


        mask_customer = D_over_customer | self.traversed_customer[:, None, :].repeat(1, self.n_car, 1)
        # print('mask_customer',mask_customer)
        # mask_customer: (batch, n_car, n_customer)

        mask_depot = is_next_depot & (
                    (mask_customer == False).long().sum(dim=2).sum(dim=1)[:, None].repeat(1, self.n_car) > 0)
        # mask_depot: (batch, n_car)
        """mask_depot = True --> We cannot choose depot in the next step 
			if 1) the vehicle is at the depot in the next step
			or 2) there is a customer node which has not been visited yet
		"""

        mask_depot = self.mask_depot & mask_depot.bool()[:, :, None].repeat(1, 1, self.n_depot)
        # mask_depot: (batch, n_car, n_depot)

        mask_depot = self.mask_depot_unused | mask_depot
        """mask_depot: (batch, n_car, n_depot) 
			mask_customer: (batch, n_car, n_customer) 
			--> return mask: (batch, n_car, n_node ,1)
		"""
        # print("mask",torch.cat([mask_depot, mask_customer], dim = -1).unsqueeze(-1))
        # mask 2*10*102*1 tensor([[[[False],[True],[True],...,[True]], [[True], [False], [True], ...,[True]],...,]])
        return torch.cat([mask_depot, mask_customer], dim=-1).unsqueeze(-1)

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
        step_context = torch.cat([prev_embeddings, self.D[:, :, None, None]], dim=-1)
        return step_context

    def _get_step(self, next_node, next_car):
        """next_node **includes depot** : (batch, 1) int(=long), range[0, n_node-1]

			return
			mask: (batch, n_car, n_node ,1)
			step_context: (batch, n_car, 1, embed+1)
		"""
        self.update_node_path(next_node, next_car)
        self.update_car_distance()
        mask = self.get_mask(next_node, next_car)
        # print("mask",mask)

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
        # print("mask_t1",mask_t1)
        step_context_t1 = self.generate_step_context()
        return mask_t1, step_context_t1

    def get_mask_t1(self):
        """mask_depot: (batch, n_car, n_depot)
			mask_customer: (batch, n_car, n_customer)
			--> return mask: (batch, n_car, n_node ,1)
		"""
        mask_depot_t1 = self.mask_depot | self.mask_depot_unused
        mask_customer_t1 = self.traversed_customer[:, None, :].repeat(1, self.n_car, 1)
        return torch.cat([mask_depot_t1, mask_customer_t1], dim=-1).unsqueeze(-1)

    def update_node_path(self, next_node, next_car):
        # car_node: (batch, n_car)
        # pi: (batch, n_car, decoder_step)
        # next_node	tensor([[1],[1]])
        self.car_prev_node = self.car_cur_node
        a = torch.arange(self.n_car, device=self.device).reshape(1, -1).repeat(self.batch, 1)
        # a tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
        b = next_car.reshape(self.batch, 1).repeat(1, self.n_car)
        # b	tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        mask_car = (a == b).long()
        # mask_car tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
        new_node = next_node.reshape(self.batch, 1).repeat(1, self.n_car)
        # next_node tensor([[6],[0]])
        # new_node  tensor([[6, 6, 6, 6, 6, 6, 6, 6, 6, 6],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        # self.car_cur_node0 tensor([[0, 1, 43, 1, 0, 1, 23, 1, 56, 1],[0, 98, 70, 75, 0, 17, 0, 86, 0, 94]])
        self.car_cur_node = mask_car * new_node + (1 - mask_car) * self.car_cur_node
        # self.car_cur_node1 tensor([[ 6,  1, 43,  1,  0,  1, 23,  1, 56,  1],[ 0, 98, 70, 75,  0, 17,  0, 86,  0, 94]])

        # (1-mask_car) keeps the same node for the unused car, mask_car updates new node for the used car
        self.pi = torch.cat([self.pi, self.car_cur_node.unsqueeze(-1)], dim=-1)
        previous_node = self.car_prev_node[self._bidx, next_car.squeeze(1)]
        dist_mat = self.build_dist_mat()


        cur_time_delta = dist_mat[self._bidx, previous_node, next_node.squeeze(1)]
        print("cur_time_delta",self.cur_time[self._bidx, next_car.squeeze(1)])
        arrival_time = self.cur_time[self._bidx, next_car.squeeze(1)] + cur_time_delta

        # cur_time_delta = (
        #         cur_time_delta[mask_car] +
        #         ((self.xy[:, 0] - arrival_time).clamp_(min=0) + self.service_time[self._bidx])[mask_car]
        # )
        self.cur_time[self._bidx, next_car] = self.cur_time[self._bidx, next_car] + cur_time_delta


        print("self.pi",self.pi)

    # self.pi tensor([[[0, 72, 72, ..., 0, 0, 6],[1, 1, 1, ..., 1, 1, 1],...,[]],[[0, 32, 39, ..., 0, 0, 0],...,]])
    def update_car_distance(self):
        # self.car_cur_node0 tensor([[0, 1, 43, 1, 0, 1, 23, 1, 56, 1],[0, 98, 70, 75, 0, 17, 0, 86, 0, 94]])
        prev_node_dist_vec = torch.gather(input=self.dist_mat, dim=1,
                                          index=self.car_prev_node[:, :, None].repeat(1, 1, self.n_node))
        # prev_node_dist_vec.size() torch.Size([2, 10, 102]) self.dist_mat torch.Size([2, 102, 102])
        # self.car_prev_node[:,:,None] tensor([[[10],[ 1],[26],[ 1],[ 0],[ 1],[ 0],[ 1],[ 0],[ 1]],[[20],[ 1],[33],[ 1],[99],[ 1],[54],[ 1],[81],[ 1]]])
        # dist = torch.gather(input = prev_node_dist_vec, dim = 2, index = self.car_cur_node[:,None,:].repeat(1,self.n_car,1))
        dist = torch.gather(input=prev_node_dist_vec, dim=2, index=self.car_cur_node[:, :, None])
        print("dist",dist)
        self.car_run += dist.squeeze(-1)
    def update_car_time(self):
        # self.car_cur_node0 tensor([[0, 1, 43, 1, 0, 1, 23, 1, 56, 1],[0, 98, 70, 75, 0, 17, 0, 86, 0, 94]])
        prev_node_dist_vec = torch.gather(input=self.dist_mat, dim=1,
                                          index=self.car_prev_node[:, :, None].repeat(1, 1, self.n_node))
        # prev_node_dist_vec.size() torch.Size([2, 10, 102]) self.dist_mat torch.Size([2, 102, 102])
        # self.car_prev_node[:,:,None] tensor([[[10],[ 1],[26],[ 1],[ 0],[ 1],[ 0],[ 1],[ 0],[ 1]],[[20],[ 1],[33],[ 1],[99],[ 1],[54],[ 1],[81],[ 1]]])
        # dist = torch.gather(input = prev_node_dist_vec, dim = 2, index = self.car_cur_node[:,None,:].repeat(1,self.n_car,1))
        dist = torch.gather(input=prev_node_dist_vec, dim=2, index=self.car_cur_node[:, :, None])
        dist_time = dist


        self.car_run += dist.squeeze(-1)

    # print(self.car_run[0])
    def update_car_fuel(self, next_node, next_car):
        """self.demand **excludes depot**: (batch, n_nodes-1)
			selected_demand: (batch, 1)
			if next node is depot, do not select demand
			self.D: (batch, n_car, 1), D denotes "remaining vehicle capacity"
			self.capacity_over_customer **excludes depot**: (batch, n_car, n_customer)
			visited_customer **excludes depot**: (batch, n_customer, 1)
			is_next_depot: (batch, 1), e.g. [[True], [True], ...]

		"""
        is_next_depot = (self.car_cur_node == self.car_start_node).bool()  # .long().sum(-1)
        # e.g., is_next_depot = next_node == 0 or next_node == 1
        # is_next_depot: (batch, n_car), e.g. [[True], [True], ...]

        new_traversed_node = torch.eye(self.n_node, device=self.device)[next_node.squeeze(1)]
        # new_traversed_node: (batch, node)
        new_traversed_customer = new_traversed_node[:, self.n_depot:]
        # new_traversed_customer: (batch, n_customer)
        self.traversed_customer = self.traversed_customer | new_traversed_customer.bool()
        # traversed_customer: (batch, n_customer)

        selected_demand = torch.gather(input=self.demand_include_depot, dim=1, index=next_node)
        # selected_demand: (batch, 1)
        selected_car = torch.eye(self.n_car, device=self.device)[next_car.squeeze(1)]
        # selected_car: (batch, n_car)
        car_used_demand = selected_car * selected_demand
        # self.D -= car_used_demand car_used_demand: (batch, n_car)
        self.D -= car_used_demand
        D =  self.D

        prev_node_dist_vec = torch.gather(input=self.dist_mat, dim=1,
                                          index=self.car_prev_node[:, :, None].repeat(1, 1, self.n_node))

        # dist = torch.gather(input = prev_node_dist_vec, dim = 2, index = self.car_cur_node[:,None,:].repeat(1,self.n_car,1))
        dist = torch.gather(input=prev_node_dist_vec, dim=2, index=self.car_cur_node[:, :, None])
        dist = dist.squeeze(2)

        # car_fuel = self.car_fuel
        # car_fuel[0]=torch.mul(dist[0],D[0])
        #
        # print('car_fuel[0]', car_fuel[0].size())
        # car_fuel[1]=torch.mul(dist[1],D[1])
        # print('car_fuel[1]', car_fuel[1])
        # car_fuel = torch.stack([car_fuel[0], car_fuel[1]], 0)
        # self.car_fuel += car_fuel
        bsz = D.shape[0]
        zero_to_bsz = torch.arange(bsz)
        for i in zero_to_bsz:
            car_fuel = self.car_fuel
            car_fuel[i] = torch.mul(dist[i], D[i])
        car_fuel = torch.stack([car_fuel[i]], 0)
        self.car_fuel += car_fuel

    # D: (batch, n_car)
    # self.D = torch.clamp(self.D, min = 0.)
    def return_depot_all_car(self, next_node, next_car):
        self.pi = torch.cat([self.pi, self.car_start_node.unsqueeze(-1)], dim=-1)
        # .
        self.car_prev_node = self.car_cur_node
        self.car_cur_node = self.car_start_node
        self.update_car_distance()
        self.update_car_fuel(next_node, next_car)

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
        return torch.topk(logits, self.n_samples, dim=1)[1]
# torch.argmax(logits, dim = 1).unsqueeze(-1)


class CategoricalSampler(Sampler):
    def forward(self, logits):
        return torch.multinomial(logits.exp(), self.n_samples)