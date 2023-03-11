def load_classifier(args, exp_configure, config):

    if exp_configure['model'] == 'GCNv2':
        if args['cls'] == 'linear':
            classifier = nn.Linear(2 * exp_configure['gnn_hidden_feats'], exp_configure['n_tasks'])
        else:
            classifier = nn.Sequential(
                nn.Dropout(config['predictor_dropout']),
                nn.Linear(2 * exp_configure['gnn_hidden_feats'], config['predictor_hidden_feats']),
                nn.LeakyReLU(),
                nn.BatchNorm1d(config['predictor_hidden_feats']),
                nn.Linear(config['predictor_hidden_feats'], exp_configure['n_tasks']),
            )

import torch.nn as nn
class LinearPredictor(nn.Module):
    def __init__(self, in_feats, out_feats, config):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        
    def forward(self, features):
        emb = self.linear(features)
        return emb

class FF_net(nn.Module):
    def __init__(self, _in, hiddens: list, _out):
        super(FF_net, self).__init__()
        self.block = nn.Sequential()
        hiddens.insert(0, _in)
        hiddens.append(_out)
        for i in range(1, len(hiddens)):
            self.block.add_module(f'linear{hiddens[i - 1]}-{hiddens[i]}',
                                  nn.Linear(hiddens[i - 1], hiddens[i]))
            self.block.add_module('relu', nn.ReLU())
        self.linear_shortcut = nn.Linear(_in, _out)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)