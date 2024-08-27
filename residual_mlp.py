# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 10:41:33 2023

@author: s00810671
"""

import torch
from torch import nn

batch_size = 16
feature_num = 100

x = torch.randn(batch_size, feature_num)

class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(feature_num, 1)
            )
        
        self.mlp2 = nn.Sequential(
            nn.Linear(feature_num, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
            )
        
        self.mlp3 = nn.Sequential(
            nn.Linear(feature_num, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
            )
        
        self.mlp4 = nn.Sequential(
            nn.Linear(feature_num, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
            )
        
        self.out_act = nn.Sigmoid()
        
    def forward(self, x):
        output = self.mlp1(x) + self.mlp2(x) + self.mlp3(x) + self.mlp4(x)
        return self.out_act(output)
    

class RecursiveMLPBlock(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_units=[], 
                 output_dim=None,
                 output_activation=None
                 ):
        super().__init__()
        
        all_units = [input_dim] + hidden_units
        hidden_activations = nn.ReLU()
        if len(all_units) > 1:
            dense_layers = []
            dense_layers.append(nn.Linear(all_units[-2], all_units[-1]))
            dense_layers.append(hidden_activations)
            if output_dim is not None:
                self.output_lin = nn.Linear(hidden_units[-1], output_dim)
            else:
                self.output_lin = nn.Identity()
            if output_activation is not None:
                self.output_act = nn.Sigmoid()
            else:
                self.output_act = nn.Identity()
            self.mlp = nn.Sequential(*dense_layers)
            if len(all_units) > 2:
                self.sub_mlp = RecursiveMLPBlock(input_dim, hidden_units[:-1], None, None)
            else:
                self.sub_mlp = None
        else:
            self.mlp = None
            self.sub_mlp = None
    
    def forward(self, inputs):
        if self.mlp:
            if self.sub_mlp:
                prev_output = self.sub_mlp(inputs)
                output = self.mlp(prev_output) + prev_output
                return self.output_act(self.output_lin(output))
            else:
                output = self.mlp(inputs)
                return self.output_act(self.output_lin(output))
    
baseline_model = Baseline()
recursive_model = RecursiveMLPBlock(feature_num, [100, 100, 100],
                                    output_activation = "sigmoid", output_dim = 1)

baseline_model(x)
recursive_model(x)

print(baseline_model(x))
print(recursive_model(x))

        








        