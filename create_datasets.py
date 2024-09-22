#!/usr/bin/python3

from rdkit import Chem
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch.utils.data import Dataset

class Molecule(Dataset):
  def __init__(self, csv_path):
    self.samples = list()
    with open(csv_path, 'r') as f:
      for line, row in enumerate(f.readlines()):
        if line == 0: continue
        smiles, label = row.split(',')
        self.samples.append((smiles, label))
  def __len__(self):
    return len(self.samples)
  def __getitem__(self, index):
    smiles, label = self.samples[index]
    molecule = Chem.MolFromSmiles(smiles)
    nodes = list()
    edges = list()
    edges_type = list()
    for idx, atom in enumerate(molecule.GetAtoms()):
      assert idx == atom.GetIdx()
      nodes.append(atom.GetAtomicNum())
      for neighbor_atom in atom.GetNeighbors():
        neighbor_idx = neighbor_atom.GetIdx()
        bond = molecule.GetBondBetweenAtoms(idx, neighbor_idx)
        edges.append([idx, neighbor_idx])
        edges_type.append(bond.GetBondType())
    x = F.one_hot(torch.tensor(nodes, dtype = torch.long),118).to(torch.float32) # x.shape = (node_num, 118)
    edge_index = torch.tensor(edges, dtype = torch.long).t().contiguous() # edge_index.shape = (2, edge_num)
    edge_type = torch.tensor(edges_type, dtype = torch.long) # edge_type.shape = (edge_num)
    data = Data(x = x, edge_index = edge_index, edge_type = edge_type, y = label)
    return data

if __name__ == "__main__":
  molecule = Molecule('dataset.csv')
  for d in molecule:
    print(d.x,d.edge_index,d.edge_type,d.y)
    break
