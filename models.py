#!/usr/bin/python3

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, aggr

class SimpleConv(MessagePassing):
  def __init__(self, channels = 256, drop_rate = 0.2):
    super().__init__(aggr = aggr.MeanAggregation())
    self.dense1 = nn.Linear(channels, channels)
    self.gelu = nn.GELU()
    self.dropout1 = nn.Dropout(drop_rate)
    self.dense2 = nn.Linear(channels, channels)
    self.dropout2 = nn.Dropout(drop_rate)
  def forward(self, x, edge_index):
    return self.propagate(edge_index, x = x)
  def message(self, x):
    results = self.dense1(x)
    results = self.gelu(results)
    results = self.dropout1(results)
    return results
  def update(self, aggr_out):
    results = self.dense2(aggr_out)
    results = self.gelu(results)
    results = self.dropout2(results)
    return results

class ConductivityPredictor(nn.Module):
  def __init__(self, channels = 256, layer_num = 4, drop_rate = 0.2):
    super().__init__()
    self.node_embed = nn.Linear(118, channels)
    #self.edge_embed = nn.Linear(22, channels)
    self.convs = nn.ModuleList([SimpleConv(channels, drop_rate) for _ in range(layer_num)])
  def forward(self, data):
    atom_num, edge_index = data.x, data.edge_index # atom_num.shape = (num_node, 118) edge_index.shape = (2, edge_num)
    results = self.node_embed(atom_num) # atom_results.shape = (num_node, channels)
    for conv in self.convs:
      results = conv(results, edge_index) # results.shape = (num_node, channels)
    results = torch.mean(results, dim = 1)
    return results

if __name__ == "__main__":
  from torch_geometric.loader import DataLoader
  from create_datasets import Molecule
  dataset = ConductivityPredictor('dataset.csv')
  loader = DataLoader(dataset, batch_size = 32, shuffle = True)
  model = FeatureExtract()
  for data in loader:
    a = model(data)
    print(a, data.y)
    break
