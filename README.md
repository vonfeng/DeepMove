# DeepMove
PyTorch implementation of WWW'18  paper-DeepMove: Predicting Human Mobility with Attentional Recurrent Networks [link](https://dl.acm.org/citation.cfm?id=3178876.3186058)

# Datasets
The sample data to evaluate our model can be found in the data folder, which contains 800+ users and ready for directly used. The raw mobility data similar to ours used in the paper can be found in this public [link](https://sites.google.com/site/yangdingqi/home/foursquare-dataset).

# Requirements
- Python 2.7
- [Pytorch](https://pytorch.org/previous-versions/) 0.20

cPickle is used in the project to store the preprocessed data and parameters. While appearing some warnings, pytorch 0.3.0 can also be used.

# Project Structure
- /codes
    - [main.py](https://github.com/vonfeng/DeepMove/blob/master/codes/main.py)
    - [model.py](https://github.com/vonfeng/DeepMove/blob/master/codes/model.py) # define models
    - [sparse_traces.py](https://github.com/vonfeng/DeepMove/blob/master/codes/sparse_traces.py) # foursquare data preprocessing 
    - [train.py](https://github.com/vonfeng/DeepMove/blob/master/codes/train.py) # define tools for train the model
- /pretrain
    - /simple
        - [res.m](https://github.com/vonfeng/DeepMove/blob/master/pretrain/simple/res.m) # pretrained model file
        - [res.json](https://github.com/vonfeng/DeepMove/blob/master/pretrain/simple/res.json) # detailed evaluation results
        - [res.txt](https://github.com/vonfeng/DeepMove/blob/master/pretrain/simple/res.txt) # evaluation results
    - /simple_long
    - /attn_local_long
    - /attn_avg_long_user
- /data # preprocessed foursquare sample data (pickle file)
- /docs # paper and presentation file
- /resutls # the default save path when training the model

# Usage
1. Load a pretrained model:
> ```python
> python main.py --model_mode=attn_avg_long_user --pretrain=1
> ```

The codes contain four network model (simple, simple_long, attn_avg_long_user, attn_local_long) and a baseline model (Markov). The parameter settings for these model can refer to their [res.txt](https://github.com/vonfeng/DeepMove/blob/master/pretrain/simple/res.txt) file.

|model_in_code | model_in_paper | top-1 accuracy (pre-trained)|
:---: |:---:|:---:
|markov | markov | 0.082|
|simple | RNN-short | 0.096|
|simple_long | RNN-long | 0.118|
|attn_avg_long_user | Ours attn-1 | 0.133|
|attn_local_long | Ours attn-2 | 0.145|

2. Train a new model:
> ```python
> python main.py --model_mode=attn_avg_long_user --pretrain=0
> ```

Other parameters (refer to [main.py](https://github.com/vonfeng/DeepMove/blob/master/codes/main.py)):
- for training: 
    - learning_rate, lr_step, lr_decay, L2, clip, epoch_max, dropout_p
- model definition: 
    - loc_emb_size, uid_emb_size, tim_emb_size, hidden_size, rnn_type, attn_type
    - history_mode: avg, avg, whole

# Others
Batch version for this project will come soon.