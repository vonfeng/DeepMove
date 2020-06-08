# coding: utf-8
from __future__ import print_function
from __future__ import division

import torch
from torch.autograd import Variable

import json
import numpy as np
try:
    import cPickle as pickle
except:
    import pickle
from collections import deque, Counter, defaultdict

class RnnParameterData(object):
    def __init__(self, loc_emb_size=500, uid_emb_size=40, voc_emb_size=50, tim_emb_size=10, hidden_size=500,
                 lr=1e-3, lr_step=3, lr_decay=0.1, dropout_p=0.5, L2=1e-5, clip=5.0, optim='Adam',
                 history_mode='avg', attn_type='dot', epoch_max=30, rnn_type='LSTM', model_mode="simple",
                 data_path='../data/', save_path='../results/', data_name='foursquare', use_cuda=True, data_mode='json'):
        self.data_path = data_path
        self.save_path = save_path
        self.data_name = data_name
        try:
            data = load_dataset(self.data_path, self.data_name, mode=data_mode)
        except:
            data = load_dataset(self.data_path, self.data_name, mode='json')
        self.vid_list = data['vid_list']
        self.uid_list = data['uid_list']
        self.data_neural = data['data_neural']

        self.tim_size = 48                 # Number of timeslot bins (48 are 24 hours for weekdays + 24 hours for weekends)
        self.loc_size = len(self.vid_list) # |L| (i.e., #locations)
        self.uid_size = len(self.uid_list) # |U| (i.e., #users)
        self.loc_emb_size = loc_emb_size   # size of location embedding
        self.tim_emb_size = tim_emb_size   # size of time embedding
        self.voc_emb_size = voc_emb_size   # size of ???
        self.uid_emb_size = uid_emb_size   # size of user embedding
        self.hidden_size = hidden_size     # size of hidden layer

        self.epoch = epoch_max             # number of iteration (default: 30)
        self.dropout_p = dropout_p         # dropout_rate (default: 0.5)
        self.use_cuda = use_cuda           # whether using GPU or CPU (default: True --> using GPU)
        self.lr = lr                       # learning rate of optimization algorithm (default: 0.001)
        self.lr_step = lr_step             # number of epochs with no improvement after which learning rate will be reduced (default: 3)
        self.lr_decay = lr_decay           # factor by which the learning rate will be reduced. new_lr = lr * factor (default: 0.1)
        self.optim = optim                 # optimization algorithm used (default: Adam)
        self.L2 = L2                       # l2 regularization weight
        self.clip = clip                   # gradient clip (default: 5.0)

        self.attn_type = attn_type         # attention mode used ['dot', 'general', 'concat'] (default: 'dot')
        self.rnn_type = rnn_type           # RNN algorithm used ['GRU', 'LSTM', 'RNN'] (default: 'LSTM')
        self.history_mode = history_mode   # aggregation method used for historical data ['avg', 'max', 'whole'] (default: 'avg')
        self.model_mode = model_mode       # model used for the training ['simple', 'simple_long', 'attn_avg_long_user', 'attn_local_long']


"""
Utility functions
"""
def load_dataset(data_path, data_name, mode='pickle'):
    if mode == 'json':
        return json.load(open('/'.join([data_path, data_name]) + '.json', 'r'))
    else:
        return pickle.load(open('/'.join([data_path, data_name]) + '.pk', 'rb'))
    
    
def extract_history(data_neural, sessions, train_id, mode, u, c, sort=True):
    ### Put some historical data in the variable "history"
    history = []
    if mode == 'test': # When performing test, first put the history from the training sessions
        test_id = data_neural[u]['train']
        for tt in test_id:
            history.extend([(s[0], s[1]) for s in sessions[str(tt)]])
    for j in range(c):
        history.extend([(s[0], s[1]) for s in sessions[str(train_id[j])]])
    ### Sort based on the time
    if sort:
        history = sorted(history, key=lambda x: x[1], reverse=False)
    return history


def extract_count(history):
    history_tim = [t[1] for t in history]
    temp = Counter()
    ### Count the visit to the location from the historical data
    for x in history_tim:
        temp[x] += 1
    history_count = list(temp.values())
    return history_tim, history_count


def torch_trace(loc_np, tim_np, target, history_loc=None, history_tim=None, history_count=None):
    trace = {}
    trace['loc'] = Variable(torch.LongTensor(loc_np))    # Tensor shape : (None, 1)
    trace['tim'] = Variable(torch.LongTensor(tim_np))    # Tensor shape : (None, 1)
    trace['target'] = Variable(torch.LongTensor(target)) # Tensor shape : (None)
    if history_loc is not None:
        trace['history_loc'] = Variable(torch.LongTensor(history_loc))
    if history_tim is not None:
        trace['history_tim'] = Variable(torch.LongTensor(history_tim))
    if history_count is not None:
        trace['history_count'] = history_count
    return trace

"""
A simple (short) history generation -- using aggregation (average / max) or concatenation

data_neural : data used for generating the training the neural network (Obtained from the dataset).
mode        : training or testing phase ['train', 'test']
mode2       : history_mode used for generating the history ['avg', 'max', 'whole'] (None means 'whole')
candidate   : list of user ids (if None, then use the ids from data_neural.keys())
"""
def generate_input_history(data_neural, mode, mode2=None, candidate=None):
    data_train = {} # Can be either for train / test / val
    train_idx = {}  # Can be either for train / test / val
    if candidate is None:
        candidate = data_neural.keys() # All the user candidates
    for u in candidate: # For each user
        sessions = data_neural[u]['sessions'] # User session
        train_id = data_neural[u][mode]
        data_train[u] = {}
        
        for c, i in enumerate(train_id): # For each session in the "training" / "testing"
            ### Skip the first session in training
            if mode == 'train' and c == 0:
                continue
            session = sessions[str(i)]
            ### Removing the last for training
            loc_np = np.reshape(np.array([s[0] for s in session[:-1]]), (len(session[:-1]), 1)) # Location array, in 2D
            tim_np = np.reshape(np.array([s[1] for s in session[:-1]]), (len(session[:-1]), 1)) # Time array, in 2D
            ### Removing the first for testing (target label)
            target = np.array([s[0] for s in session[1:]])

            ### Put some historical data in the variable "history"
            history = extract_history(data_neural, sessions, train_id, mode, u, c, sort=True)
            history_count = None

            # merge traces with same time stamp
            if mode2 == 'max':
                """
                Putting the list of location visited by the user in the particular timeslot
                """
                history_tmp = defaultdict(list)
                for tr in history:
                    history_tmp[tr[1]].append(tr[0])
                
                history_filter = []
                for t in history_tmp:
                    selected = history_tmp[t]
                    if len(selected) == 1: # If only 1 location on the list
                        history_filter.append((selected[0], t))
                    else:
                        tmp = Counter(selected).most_common() # Find the one that is most common
                        if tmp[0][1] > 1: # If no competition, then just put it
                            history_filter.append((selected[0], t))
                        else: # If there is competition, select randomly
                            ti = np.random.randint(len(tmp))
                            history_filter.append((tmp[ti][0], t))
                ### Replace history with the aggregated version
                history = history_filter
                history = sorted(history, key=lambda x: x[1], reverse=False)
            elif mode2 == 'avg':
                ### Calculate how many times a location visited (to serve as the "average" count found in the historical data)
                history_tim, history_count = extract_count(history)
            ################

            history_loc = np.reshape(np.array([s[0] for s in history]), (len(history), 1))
            history_tim = np.reshape(np.array([s[1] for s in history]), (len(history), 1))
            ### Convert into torch tensor
            trace = torch_trace(loc_np, tim_np, target, history_loc=history_loc, history_tim=history_tim, history_count=history_count)
            data_train[u][i] = trace
        train_idx[u] = train_id
    return data_train, train_idx


"""
Generate a long history using a simple concatenation (i.e., simple_long). 
The history session index are squeezed into a single index which means that 
there is no difference between "short-term" and "long-term" sequence.

data_neural : data used for generating the training the neural network (Obtained from the dataset).
mode        : training or testing phase ['train', 'test']
candidate   : list of user ids (if None, then use the ids from data_neural.keys())
"""
def generate_input_long_history2(data_neural, mode, candidate=None):
    data_train = {}
    train_idx = {}
    if candidate is None:
        candidate = data_neural.keys()
    for u in candidate:
        sessions = data_neural[u]['sessions']
        train_id = data_neural[u][mode]
        data_train[u] = {}

        session = []
        ### Put all sessions altogether to become a long sequence
        for c, i in enumerate(train_id):
            session.extend(sessions[str(i)])
            ### Generate the target variable as the predicted location
            target = np.array([s[0] for s in session[1:]])

            ### Location and time pair used for training
            loc_np = np.reshape(np.array([s[0] for s in session[:-1]]), (len(session[:-1]), 1))
            tim_np = np.reshape(np.array([s[1] for s in session[:-1]]), (len(session[:-1]), 1))
            ### Convert into torch tensor
            trace = torch_trace(loc_np, tim_np, target)
            data_train[u][i] = trace

            if mode == 'train':
                train_idx[u] = [0, i]
            else:
                train_idx[u] = [i]
    return data_train, train_idx


"""
Generate a long history based on all historical + recent data.

data_neural : data used for generating the training the neural network (Obtained from the dataset).
mode        : training or testing phase ['train', 'test']
candidate   : list of user ids (if None, then use the ids from data_neural.keys())
"""
def generate_input_long_history(data_neural, mode, candidate=None):
    data_train = {}
    train_idx = {}
    if candidate is None:
        candidate = data_neural.keys()
    for u in candidate:
        sessions = data_neural[u]['sessions']
        train_id = data_neural[u][mode]
        data_train[u] = {}
        for c, i in enumerate(train_id):
            if mode == 'train' and c == 0:
                continue
            session = sessions[str(i)]
            target = np.array([s[0] for s in session[1:]])

            ### Put some historical data in the variable "history"
            history = extract_history(data_neural, sessions, train_id, mode, u, c, sort=True)
            ### Calculate how many times a location visited (to serve as the "average" count found in the historical data)
            history_tim, history_count = extract_count(history)

            history_loc = np.reshape(np.array([s[0] for s in history]), (len(history), 1))
            history_tim = np.reshape(np.array([s[1] for s in history]), (len(history), 1))

            ### Consider the data from the historical data + current session
            loc_tim = history
            loc_tim.extend([(s[0], s[1]) for s in session[:-1]])
            loc_np = np.reshape(np.array([s[0] for s in loc_tim]), (len(loc_tim), 1))
            tim_np = np.reshape(np.array([s[1] for s in loc_tim]), (len(loc_tim), 1))
            ### Convert into torch tensor
            trace = torch_trace(loc_np, tim_np, target, history_loc=history_loc, history_tim=history_tim, history_count=history_count)
            data_train[u][i] = trace
        train_idx[u] = train_id
    return data_train, train_idx


"""
Generate queue for training / testing data

train_idx   : index of training / testing data
mode        : queue generation method ['random', 'normal']
mode2       : training or testing phase ['train', 'test']
"""
def generate_queue(train_idx, mode, mode2):
    """return a deque. You must use it by train_queue.popleft()"""
    user = train_idx.keys()
    train_queue = deque()
    if mode == 'random':
        initial_queue = {}
        for u in user:
            if mode2 == 'train':
                initial_queue[u] = deque(train_idx[u][1:])
            else:
                initial_queue[u] = deque(train_idx[u])
        queue_left = 1
        while queue_left > 0:
            np.random.shuffle(user)
            for j, u in enumerate(user):
                if len(initial_queue[u]) > 0:
                    train_queue.append((u, initial_queue[u].popleft()))
                if j >= int(0.01 * len(user)):
                    break
            queue_left = sum([1 for x in initial_queue if len(initial_queue[x]) > 0])
    elif mode == 'normal':
        for u in user:
            for i in train_idx[u]:
                train_queue.append((u, i))
    return train_queue


"""
Get accuracy of top-1, top-5, and top-10
"""
def get_acc(target, scores):
    """target and scores are torch cuda Variable"""
    target = target.data.cpu().numpy()
    val, idxx = scores.data.topk(10, 1)
    predx = idxx.cpu().numpy()
    acc = np.zeros((3, 1))
    for i, p in enumerate(predx):
        t = target[i]
        if t in p[:10] and t > 0:
            acc[0] += 1
        if t in p[:5] and t > 0:
            acc[1] += 1
        if t == p[0] and t > 0:
            acc[2] += 1
    return acc

"""
Unused function
"""
# def get_hint(target, scores, users_visited):
#     """target and scores are torch cuda Variable"""
#     target = target.data.cpu().numpy()
#     val, idxx = scores.data.topk(1, 1)
#     predx = idxx.cpu().numpy()
#     hint = np.zeros((3,))
#     count = np.zeros((3,))
#     count[0] = len(target)
#     for i, p in enumerate(predx):
#         t = target[i]
#         if t == p[0] and t > 0:
#             hint[0] += 1
#         if t in users_visited:
#             count[1] += 1
#             if t == p[0] and t > 0:
#                 hint[1] += 1
#         else:
#             count[2] += 1
#             if t == p[0] and t > 0:
#                 hint[2] += 1
#     return hint, count


"""
Evaluating the model

data      : 
run_idx   : train_idx / test_idx
mode      : training / testing phase ['train', 'test']
lr        : learning rate of optimizer (used for gradient clipping)
clip      : gradient clip
model     : model used
optimizer : optimization variable (e.g., Adam)
criterion : evaluation metric (e.g., negative log likelihood)
mode2     : model used for the training ['simple', 'simple_long', 'attn_avg_long_user', 'attn_local_long']
"""
def run_simple(data, run_idx, mode, lr, clip, model, optimizer, criterion, mode2=None):
    """mode=train: return model, avg_loss
       mode=test: return avg_loss, avg_acc, users_rnn_acc"""
    run_queue = None
    if mode == 'train':
        model.train(True)
        run_queue = generate_queue(run_idx, 'random', 'train')
    elif mode == 'test':
        model.train(False)
        run_queue = generate_queue(run_idx, 'normal', 'test')
    total_loss = []
    queue_len = len(run_queue)

    users_acc = {}
    for c in range(queue_len):
        optimizer.zero_grad()
        u, i = run_queue.popleft()
        if u not in users_acc:
            users_acc[u] = [0, 0]
        loc = data[u][i]['loc'].cuda()
        tim = data[u][i]['tim'].cuda()
        target = data[u][i]['target'].cuda()
        uid = Variable(torch.LongTensor([u])).cuda()

        if 'attn' in mode2:
            history_loc = data[u][i]['history_loc'].cuda()
            history_tim = data[u][i]['history_tim'].cuda()

        """
        Forward process
        """
        if mode2 in ['simple', 'simple_long']:
            scores = model(loc, tim)
        elif mode2 == 'attn_avg_long_user':
            history_count = data[u][i]['history_count']
            target_len = target.data.size()[0]
            scores = model(loc, tim, history_loc, history_tim, history_count, uid, target_len)
        elif mode2 == 'attn_local_long':
            target_len = target.data.size()[0]
            scores = model(loc, tim, target_len)

        if scores.data.size()[0] > target.data.size()[0]:
            scores = scores[-target.data.size()[0]:]
        loss = criterion(scores, target)

        if mode == 'train':
            loss.backward()
            # gradient clipping
            try:
                torch.nn.utils.clip_grad_norm(model.parameters(), clip)
                for p in model.parameters():
                    if p.requires_grad:
                        p.data.add_(-lr, p.grad.data)
            except:
                pass
            optimizer.step()
        elif mode == 'test':
            users_acc[u][0] += len(target)
            acc = get_acc(target, scores)
            users_acc[u][1] += acc[2]
        total_loss.append(loss.data.cpu().numpy()[0])

    avg_loss = np.mean(total_loss, dtype=np.float64)
    if mode == 'train':
        return model, avg_loss
    elif mode == 'test':
        users_rnn_acc = {}
        for u in users_acc:
            tmp_acc = users_acc[u][1] / users_acc[u][0]
            users_rnn_acc[u] = tmp_acc.tolist()[0]
        avg_acc = np.mean([users_rnn_acc[x] for x in users_rnn_acc])
        return avg_loss, avg_acc, users_rnn_acc


"""
Evaluation using a simple markov chain
"""
def markov(parameters, candidate):
    validation = {}
    for u in candidate:
        traces = parameters.data_neural[u]['sessions']
        train_id = parameters.data_neural[u]['train']
        test_id = parameters.data_neural[u]['test']
        trace_train = []
        for tr in train_id:
            trace_train.append([t[0] for t in traces[tr]])
        locations_train = []
        for t in trace_train:
            locations_train.extend(t)
        trace_test = []
        for tr in test_id:
            trace_test.append([t[0] for t in traces[tr]])
        locations_test = []
        for t in trace_test:
            locations_test.extend(t)
        validation[u] = [locations_train, locations_test]
    acc = 0
    count = 0
    user_acc = {}
    for u in validation.keys():
        topk = list(set(validation[u][0]))
        transfer = np.zeros((len(topk), len(topk)))

        # train
        sessions = parameters.data_neural[u]['sessions']
        train_id = parameters.data_neural[u]['train']
        for i in train_id:
            for j, s in enumerate(sessions[i][:-1]):
                loc = s[0]
                target = sessions[i][j + 1][0]
                if loc in topk and target in topk:
                    r = topk.index(loc)
                    c = topk.index(target)
                    transfer[r, c] += 1
        for i in range(len(topk)):
            tmp_sum = np.sum(transfer[i, :])
            if tmp_sum > 0:
                transfer[i, :] = transfer[i, :] / tmp_sum

        # validation
        user_count = 0
        user_acc[u] = 0
        test_id = parameters.data_neural[u]['test']
        for i in test_id:
            for j, s in enumerate(sessions[i][:-1]):
                loc = s[0]
                target = sessions[i][j + 1][0]
                count += 1
                user_count += 1
                if loc in topk:
                    pred = np.argmax(transfer[topk.index(loc), :])
                    if pred >= len(topk) - 1:
                        pred = np.random.randint(len(topk))

                    pred2 = topk[pred]
                    if pred2 == target:
                        acc += 1
                        user_acc[u] += 1
        user_acc[u] = user_acc[u] / user_count
    avg_acc = np.mean([user_acc[u] for u in user_acc])
    return avg_acc, user_acc
