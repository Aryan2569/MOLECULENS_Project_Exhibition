"""
Smoke test: proves the full MOLECULENS pipeline chains together
(SMILES -> RDKit parsing -> graph construction -> GCN forward pass)
before we build out the real project. Run this first on your machine
to confirm your environment is sane.
"""
from rdkit import Chem
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool


def mol_to_graph(smiles: str, label: float) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")

    atom_features = [
        [atom.GetAtomicNum(), atom.GetDegree(), atom.GetFormalCharge()]
        for atom in mol.GetAtoms()
    ]
    x = torch.tensor(atom_features, dtype=torch.float)

    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])  # undirected: add both directions
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.float))


class TinyGCN(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=16, out_dim=1):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return self.out(x)


if __name__ == "__main__":
    # A handful of real molecules, standing in for a Tox21/ESOL batch
    samples = [
        ("CCO", 1.0),          # ethanol
        ("c1ccccc1", 0.0),     # benzene
        ("CC(=O)Oc1ccccc1C(=O)O", 1.0),  # aspirin
    ]

    graphs = [mol_to_graph(smi, y) for smi, y in samples]

    model = TinyGCN()
    batch = torch.cat([torch.full((g.x.size(0),), i) for i, g in enumerate(graphs)])
    x = torch.cat([g.x for g in graphs], dim=0)

    offset = 0
    edge_chunks = []
    for g in graphs:
        edge_chunks.append(g.edge_index + offset)
        offset += g.x.size(0)
    edge_index = torch.cat(edge_chunks, dim=1)

    preds = model(x, edge_index, batch)
    print("Parsed", len(graphs), "molecules")
    print("Predictions (untrained, will be noise):", preds.squeeze().tolist())
    print("\nPipeline is fully wired: RDKit -> graph -> GCN -> prediction. You're good to go.")
