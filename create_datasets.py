#!/usr/bin/python3

from rdkit import Chem
import numpy as np
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset

class Molecule(Dataset):
  def __init__(self, csv_path):
    self.samples = list()
    with open(csv_path, 'r') as f:
      if line == 0: continue
      smiles, label = row.split(',')
      self.samples.append((smiles, label))
  def __len__(self):
    return len(self.samples)
  def __getitem__(self, index):
    smiles, label = self.samples[index]
    molecule = Chem.MolFromSmiles(smiles)
    nodes = list()
    nodes_num = list()
    edges = list()
    edges_type = list()
    for atom in molecule.GetAtoms():
      idx = atom.GetIdx()
      nodes_num.append(atom.GetAtomicNum())
      nodes.append([idx])
      for neighbor_atom in atom.GetNeighbors():
        neighbor_idx = neighbor_atom.GetIdx()
        bond = molecule.GetBondBetweenAtoms(idx, neighbor_idx)
        edges.append([idx, neighbor_idx])
        edges_type.append(bond.GetBondType())
    x = torch.tensor(nodes, dtype = torch.long)
    node_num = torch.tensor(nodes_num, dtype = torch.long)
    edge_index = torch.tensor(edges, dtype = torch.long).t().contiguous()
    edge_type = torch.tensor(edges_type, dtype = torch.long)
    data = Data(x = x, node_num = node_num, edge_index = edge_index, edge_type = edge_type)
    return data

