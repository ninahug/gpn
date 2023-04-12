import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.distributions import Categorical

import sys

ourlogzero = sys.float_info.min
class BeamSearch(nn.Module):
    def __init__(self, neuralnet, args):
        super(BeamSearch, self).__init__()

        self.device = args.device
        self.neuralnet = neuralnet
        self.dyn_feat = DynamicFeatures(args)
        self.lookahead = Lookahead(args)
        self.mu = ModelUtils(args)

    def forward(self, inputs, data_scaled, start_time, dist_mat, infer_type, beam_size):
        self.beam_size = beam_size
        _, sequence_size, input_size = inputs.size()

        # first step  - node 0
        bpresent_time = start_time*torch.ones(1, 1, device=self.device)

        mask = torch.ones(1, sequence_size, device=self.device, requires_grad=False, dtype= torch.uint8)
        bpres_actions = torch.zeros(1, dtype=torch.int64,device=self.device)
        beam_idx = torch.arange(0, 1, device=self.device)

        done, mask = self.mu.feasibility_control(inputs.expand(beam_idx.shape[0], -1, -1),
                                                 mask, dist_mat, bpres_actions, bpresent_time,
                                                 torch.arange(0, mask.shape[0], device=self.device),
                                                 first_step=True)
        adj_mask = self.lookahead.adjacency_matrix(inputs.expand(beam_idx.shape[0], -1, -1),
                                                   mask, dist_mat, bpres_actions, bpresent_time)

        h_0, c_0 = self.neuralnet.decoder.hidden_0
        dec_hidden = (h_0.expand(1, -1), c_0.expand(1, -1))

        step = 0

        # encoder first forward pass
        bdata_scaled = data_scaled.expand(1,-1,-1)
        sum_log_probs = torch.zeros(1, device=self.device).float()

        bdyn_inputs = self.dyn_feat.make_dynamic_feat(inputs.expand(1,-1,-1), bpresent_time, bpres_actions, dist_mat, beam_idx)
        emb1 = self.neuralnet.sta_emb(bdata_scaled)
        emb2 = self.neuralnet.dyn_emb(bdyn_inputs)
        enc_inputs = torch.cat((emb1, emb2), dim=2)

        _, _, enc_outputs = self.neuralnet(enc_inputs, enc_inputs, adj_mask, enc_inputs, dec_hidden, mask, first_step=True)

        decoder_input = enc_outputs[beam_idx, bpres_actions]

        done, mask = self.mu.feasibility_control(inputs.expand(beam_idx.shape[0], -1, -1),
                                                 mask, dist_mat, bpres_actions, bpresent_time,
                                                 torch.arange(0, mask.shape[0], device=self.device))
        adj_mask = self.lookahead.adjacency_matrix(inputs.expand(beam_idx.shape[0], -1, -1),
                                                   mask, dist_mat, bpres_actions, bpresent_time)

        # encoder/decoder forward pass
        bdyn_inputs = self.dyn_feat.make_dynamic_feat(inputs.expand(beam_idx.shape[0], -1, -1), bpresent_time, bpres_actions, dist_mat, beam_idx)
        emb2 = self.neuralnet.dyn_emb(bdyn_inputs)
        enc_inputs = torch.cat((emb1,emb2), dim=2)

        policy, dec_hidden, enc_outputs = self.neuralnet(enc_inputs, enc_outputs, adj_mask, decoder_input, dec_hidden, mask)

        future_actions, log_probs, beam_idx = self.select_actions(policy, sum_log_probs, mask, infer_type)
        # info update
        h_step = torch.index_select(dec_hidden[0], dim=0, index = beam_idx)
        c_step = torch.index_select(dec_hidden[1], dim=0, index = beam_idx)
        dec_hidden = (h_step,c_step)

        mask = torch.index_select(mask, dim=0, index=beam_idx)
        bpresent_time = torch.index_select(bpresent_time, dim=0, index=beam_idx)
        bpres_actions = torch.index_select(bpres_actions, dim=0, index=beam_idx)
        enc_outputs  = torch.index_select(enc_outputs, dim=0, index=beam_idx)
        sum_log_probs = torch.index_select(sum_log_probs, dim=0, index=beam_idx)

        emb1 = torch.index_select(emb1, dim=0, index=beam_idx)

        # initialize buffers
        bllog_probs = torch.zeros(bpres_actions.shape[0], sequence_size, device=self.device).float()
        blactions = torch.zeros(bpres_actions.shape[0], sequence_size, device=self.device).long()

        sum_log_probs += log_probs.squeeze(0).detach()

        blactions[:, step] = bpres_actions

        final_log_probs, final_actions, lstep_mask = [], [], []

        # Starting the trip
        while not done:

            future_actions = future_actions.squeeze(0)

            beam_size = bpres_actions.shape[0]
            bpres_actions, bpresent_time, bstep_mask = \
                self.mu.one_step_update(inputs.expand(beam_size, -1, -1), dist_mat,
                                        bpres_actions, future_actions, bpresent_time,
                                        torch.arange(0,beam_size,device=self.device),
                                        beam_size)

            bllog_probs[:, step] = log_probs
            blactions[:, step+1] = bpres_actions
            step+=1

            done, mask = self.mu.feasibility_control(inputs.expand(beam_idx.shape[0], -1, -1),
                                                     mask, dist_mat, bpres_actions, bpresent_time,
                                                     torch.arange(0, mask.shape[0], device=self.device))
            adj_mask = self.lookahead.adjacency_matrix(inputs.expand(beam_idx.shape[0], -1, -1),
                                                       mask, dist_mat, bpres_actions, bpresent_time)

            active_beam_idx = torch.nonzero(mask[:, -1], as_tuple=False).squeeze(1)
            end_beam_idx = torch.nonzero((mask[:, -1]==0), as_tuple=False).squeeze(1)

            if end_beam_idx.shape[0]>0:

                final_log_probs.append(torch.index_select(bllog_probs, dim=0, index=end_beam_idx))
                final_actions.append(torch.index_select(blactions, dim=0, index=end_beam_idx))

                # ending seq info update
                h_step = torch.index_select(dec_hidden[0], dim=0, index = active_beam_idx)
                c_step = torch.index_select(dec_hidden[1], dim=0, index = active_beam_idx)
                dec_hidden = (h_step,c_step)

                mask = torch.index_select(mask, dim=0, index=active_beam_idx)
                adj_mask = torch.index_select(adj_mask, dim=0, index=active_beam_idx)

                bpresent_time = torch.index_select(bpresent_time, dim=0, index=active_beam_idx)
                bpres_actions = torch.index_select(bpres_actions, dim=0, index=active_beam_idx)
                enc_outputs  = torch.index_select(enc_outputs, dim=0, index=active_beam_idx)

                emb1 = torch.index_select(emb1, dim=0, index=active_beam_idx)

                blactions = torch.index_select(blactions, dim=0, index=active_beam_idx)
                bllog_probs = torch.index_select(bllog_probs, dim=0, index=active_beam_idx)
                sum_log_probs = torch.index_select(sum_log_probs, dim=0, index=active_beam_idx)

            if done: break
            decoder_input = enc_outputs[torch.arange(0, bpres_actions.shape[0], device=self.device), bpres_actions]

            bdyn_inputs = self.dyn_feat.make_dynamic_feat(inputs.expand(beam_idx.shape[0], -1, -1), bpresent_time, bpres_actions, dist_mat, active_beam_idx)
            emb2 = self.neuralnet.dyn_emb(bdyn_inputs)
            enc_inputs = torch.cat((emb1,emb2), dim=2)

            policy, dec_hidden, enc_outputs = self.neuralnet(enc_inputs, enc_outputs, adj_mask, decoder_input, dec_hidden, mask)

            future_actions, log_probs, beam_idx = self.select_actions(policy, sum_log_probs, mask, infer_type)

            # info update
            h_step = torch.index_select(dec_hidden[0], dim=0, index = beam_idx)
            c_step = torch.index_select(dec_hidden[1], dim=0, index = beam_idx)
            dec_hidden = (h_step,c_step)

            mask = torch.index_select(mask, dim=0, index=beam_idx)
            adj_mask = torch.index_select(adj_mask, dim=0, index=beam_idx)

            bpresent_time = torch.index_select(bpresent_time, dim=0, index=beam_idx)
            bpres_actions = torch.index_select(bpres_actions, dim=0, index=beam_idx)

            enc_outputs  = torch.index_select(enc_outputs, dim=0, index=beam_idx)

            emb1 = torch.index_select(emb1, dim=0, index=beam_idx)

            blactions = torch.index_select(blactions, dim=0, index=beam_idx)
            bllog_probs = torch.index_select(bllog_probs, dim=0, index=beam_idx)
            sum_log_probs = torch.index_select(sum_log_probs, dim=0, index=beam_idx)

            sum_log_probs += log_probs.squeeze(0).detach()

        return torch.cat(final_actions, dim=0), torch.cat(final_log_probs, dim=0)



    def select_actions(self, policy, sum_log_probs, mask, infer_type = 'stochastic'):

        beam_size, seq_size = policy.size()
        nzn  = torch.nonzero(mask, as_tuple=False).shape[0]
        sample_size = min(nzn,self.beam_size)

        ourlogzero = sys.float_info.min
        lpolicy = policy.masked_fill(mask == 0, ourlogzero).log()
        npolicy = sum_log_probs.unsqueeze(1) + lpolicy
        if infer_type == 'stochastic':
            nnpolicy = npolicy.exp().masked_fill(mask == 0, 0).view(1, -1)

            m = Categorical(nnpolicy)
            gact_ind = torch.multinomial(nnpolicy, sample_size)
            log_select =  m.log_prob(gact_ind)

        elif infer_type == 'greedy':
            nnpolicy = npolicy.exp().masked_fill(mask == 0, 0).view(1, -1)

            _ , gact_ind = nnpolicy.topk(sample_size, dim = 1)
            prob = policy.view(-1)[gact_ind]
            log_select =  prob.log()

        beam_id = torch.floor_divide(gact_ind, seq_size).squeeze(0)
        act_ind = torch.fmod(gact_ind, seq_size)

        return act_ind, log_select, beam_id