# coding: utf-8
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# ############# simple rnn model ####################### #
class TrajPreSimple(nn.Module):
    """baseline rnn model"""

    def __init__(self, parameters):
        super(TrajPreSimple, self).__init__()
        self.loc_size = parameters.loc_size
        self.loc_emb_size = parameters.loc_emb_size
        self.tim_size = parameters.tim_size
        self.tim_emb_size = parameters.tim_emb_size
        self.hidden_size = parameters.hidden_size
        self.use_cuda = parameters.use_cuda
        self.rnn_type = parameters.rnn_type

        self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)
        self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)

        input_size = self.loc_emb_size + self.tim_emb_size

        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size, self.hidden_size, 1)
        self.init_weights()

        self.fc = nn.Linear(self.hidden_size, self.loc_size)
        self.dropout = nn.Dropout(p=parameters.dropout_p)

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for consistency with Keras version
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform(t)
        for t in hh:
            nn.init.orthogonal(t)
        for t in b:
            nn.init.constant(t, 0)

    def forward(self, loc, tim):
        h1 = Variable(torch.zeros(1, 1, self.hidden_size))
        c1 = Variable(torch.zeros(1, 1, self.hidden_size))
        if self.use_cuda:
            h1 = h1.cuda()
            c1 = c1.cuda()

        loc_emb = self.emb_loc(loc)
        tim_emb = self.emb_tim(tim)
        x = torch.cat((loc_emb, tim_emb), 2)
        x = self.dropout(x)

        if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
            out, h1 = self.rnn(x, h1)
        elif self.rnn_type == 'LSTM':
            out, (h1, c1) = self.rnn(x, (h1, c1))
        out = out.squeeze(1)
        out = F.selu(out)
        out = self.dropout(out)

        y = self.fc(out)
        score = F.log_softmax(y)  # calculate loss by NLLoss
        return score


# ############# rnn model with attention ####################### #
class Attn(nn.Module):
    """Attention Module. Heavily borrowed from Practical Pytorch
    https://github.com/spro/practical-pytorch/tree/master/seq2seq-translation"""

    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(self.hidden_size))

    def forward(self, out_state, history):
        seq_len = history.size()[0]
        state_len = out_state.size()[0]
        attn_energies = Variable(torch.zeros(state_len, seq_len)).cuda()
        for i in range(state_len):
            for j in range(seq_len):
                attn_energies[i, j] = self.score(out_state[i], history[j])
        return F.softmax(attn_energies)

    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output)))
            energy = self.other.dot(energy)
            return energy


# ##############long###########################
class TrajPreAttnAvgLongUser(nn.Module):
    """rnn model with long-term history attention"""

    def __init__(self, parameters):
        super(TrajPreAttnAvgLongUser, self).__init__()
        self.loc_size = parameters.loc_size
        self.loc_emb_size = parameters.loc_emb_size
        self.tim_size = parameters.tim_size
        self.tim_emb_size = parameters.tim_emb_size
        self.uid_size = parameters.uid_size
        self.uid_emb_size = parameters.uid_emb_size
        self.hidden_size = parameters.hidden_size
        self.attn_type = parameters.attn_type
        self.rnn_type = parameters.rnn_type
        self.use_cuda = parameters.use_cuda

        self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)
        self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)
        self.emb_uid = nn.Embedding(self.uid_size, self.uid_emb_size)

        input_size = self.loc_emb_size + self.tim_emb_size
        self.attn = Attn(self.attn_type, self.hidden_size)
        self.fc_attn = nn.Linear(input_size, self.hidden_size)

        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size, self.hidden_size, 1)

        self.fc_final = nn.Linear(2 * self.hidden_size + self.uid_emb_size, self.loc_size)
        self.dropout = nn.Dropout(p=parameters.dropout_p)
        self.init_weights()

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for consistency with Keras version
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform(t)
        for t in hh:
            nn.init.orthogonal(t)
        for t in b:
            nn.init.constant(t, 0)

    def forward(self, loc, tim, history_loc, history_tim, history_count, uid, target_len):
        h1 = Variable(torch.zeros(1, 1, self.hidden_size))
        c1 = Variable(torch.zeros(1, 1, self.hidden_size))
        if self.use_cuda:
            h1 = h1.cuda()
            c1 = c1.cuda()

        loc_emb = self.emb_loc(loc)
        tim_emb = self.emb_tim(tim)
        x = torch.cat((loc_emb, tim_emb), 2)
        x = self.dropout(x)

        loc_emb_history = self.emb_loc(history_loc).squeeze(1)
        tim_emb_history = self.emb_tim(history_tim).squeeze(1)
        count = 0
        loc_emb_history2 = Variable(torch.zeros(len(history_count), loc_emb_history.size()[-1])).cuda()
        tim_emb_history2 = Variable(torch.zeros(len(history_count), tim_emb_history.size()[-1])).cuda()
        for i, c in enumerate(history_count):
            if c == 1:
                tmp = loc_emb_history[count].unsqueeze(0)
            else:
                tmp = torch.mean(loc_emb_history[count:count + c, :], dim=0, keepdim=True)
            loc_emb_history2[i, :] = tmp
            tim_emb_history2[i, :] = tim_emb_history[count, :].unsqueeze(0)
            count += c

        history = torch.cat((loc_emb_history2, tim_emb_history2), 1)
        history = F.tanh(self.fc_attn(history))

        if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
            out_state, h1 = self.rnn(x, h1)
        elif self.rnn_type == 'LSTM':
            out_state, (h1, c1) = self.rnn(x, (h1, c1))
        out_state = out_state.squeeze(1)
        # out_state = F.selu(out_state)

        attn_weights = self.attn(out_state[-target_len:], history).unsqueeze(0)
        context = attn_weights.bmm(history.unsqueeze(0)).squeeze(0)
        out = torch.cat((out_state[-target_len:], context), 1)  # no need for fc_attn

        uid_emb = self.emb_uid(uid).repeat(target_len, 1)
        out = torch.cat((out, uid_emb), 1)
        out = self.dropout(out)

        y = self.fc_final(out)
        score = F.log_softmax(y)

        return score


class TrajPreLocalAttnLong(nn.Module):
    """rnn model with long-term history attention"""

    def __init__(self, parameters):
        super(TrajPreLocalAttnLong, self).__init__()
        self.loc_size = parameters.loc_size
        self.loc_emb_size = parameters.loc_emb_size
        self.tim_size = parameters.tim_size
        self.tim_emb_size = parameters.tim_emb_size
        self.hidden_size = parameters.hidden_size
        self.attn_type = parameters.attn_type
        self.use_cuda = parameters.use_cuda
        self.rnn_type = parameters.rnn_type

        self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)
        self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)

        input_size = self.loc_emb_size + self.tim_emb_size
        self.attn = Attn(self.attn_type, self.hidden_size)
        self.fc_attn = nn.Linear(input_size, self.hidden_size)

        if self.rnn_type == 'GRU':
            self.rnn_encoder = nn.GRU(input_size, self.hidden_size, 1)
            self.rnn_decoder = nn.GRU(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'LSTM':
            self.rnn_encoder = nn.LSTM(input_size, self.hidden_size, 1)
            self.rnn_decoder = nn.LSTM(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'RNN':
            self.rnn_encoder = nn.RNN(input_size, self.hidden_size, 1)
            self.rnn_decoder = nn.LSTM(input_size, self.hidden_size, 1)

        self.fc_final = nn.Linear(2 * self.hidden_size, self.loc_size)
        self.dropout = nn.Dropout(p=parameters.dropout_p)
        self.init_weights()

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for consistency with Keras version
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform(t)
        for t in hh:
            nn.init.orthogonal(t)
        for t in b:
            nn.init.constant(t, 0)

    def forward(self, loc, tim, target_len):
        h1 = Variable(torch.zeros(1, 1, self.hidden_size))
        h2 = Variable(torch.zeros(1, 1, self.hidden_size))
        c1 = Variable(torch.zeros(1, 1, self.hidden_size))
        c2 = Variable(torch.zeros(1, 1, self.hidden_size))
        if self.use_cuda:
            h1 = h1.cuda()
            h2 = h2.cuda()
            c1 = c1.cuda()
            c2 = c2.cuda()

        loc_emb = self.emb_loc(loc)
        tim_emb = self.emb_tim(tim)
        x = torch.cat((loc_emb, tim_emb), 2)
        x = self.dropout(x)

        if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
            hidden_history, h1 = self.rnn_encoder(x[:-target_len], h1)
            hidden_state, h2 = self.rnn_decoder(x[-target_len:], h2)
        elif self.rnn_type == 'LSTM':
            hidden_history, (h1, c1) = self.rnn_encoder(x[:-target_len], (h1, c1))
            hidden_state, (h2, c2) = self.rnn_decoder(x[-target_len:], (h2, c2))

        hidden_history = hidden_history.squeeze(1)
        hidden_state = hidden_state.squeeze(1)
        attn_weights = self.attn(hidden_state, hidden_history).unsqueeze(0)
        context = attn_weights.bmm(hidden_history.unsqueeze(0)).squeeze(0)
        out = torch.cat((hidden_state, context), 1)  # no need for fc_attn
        out = self.dropout(out)

        y = self.fc_final(out)
        score = F.log_softmax(y)

        return score
