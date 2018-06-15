# DeepMove
Contains PyTorch implementation of WWW'18  paper-DeepMove: Predicting Human Mobility with Attentional Recurrent Networks [link](https://dl.acm.org/citation.cfm?id=3178876.3186058)

# Datasets
The sample data to evaluate our model can be found in the data folder, which contains 800+ users and ready for directly used. The raw mobility data similar to ours used in the paper can be found in this public [link](https://sites.google.com/site/yangdingqi/home/foursquare-dataset).

# Requirements
- Python 2.7
- [Pytorch](https://pytorch.org/previous-versions/) 0.20
- [setproctitle](https://pypi.org/project/setproctitle/) change process's title
cPickle is used in the project to store the preprocessed data and parameters. While some warnings, Pytorch 0.3.0 can also be used.

# Usage
1. Train a new model:
> ```python
> python main.py --model_mode=attn_avg_long_user --pretrain=0
> ```
2. Load a pretrained model:
> ```python
> python main.py --model_mode=attn_avg_long_user --pretrain=1
> ```

The codes contain four network model (simple, simple_long, attn_avg_long_user, attn_local_long) and a baseline model (Markov). 
model_in_code | model_in_paper | top-1 accuracy (pre-trained)
--- |---|---
markov | markov | 0.082
simple | RNN-short | 0.096
simple_long | RNN-long | 0.118
attn_avg_long_user | Ours attn-1 | 0.133
attn_local_long | Ours attn-2 | 0.145

Other parameter settings for training the model can refer to the main.py file. 

# Architecture
![network architecture](http://deliveryimages.acm.org/10.1145/3190000/3186058/images/www2018-67-fig3.jpg)