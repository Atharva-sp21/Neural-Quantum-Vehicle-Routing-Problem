# ==========================================
# FILE: model_def.py (FINAL VERIFIED VERSION)
# ==========================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import pennylane as qml
import numpy as np

# 1. QUANTUM CONFIGURATION
# We use 7 qubits because your Indian Dataset has 7 features:
# [Stock, Sales, Days, Margin, Shelf_Life, Festival_Flag, Credit_Score]
n_qubits = 7  
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    # Embed features onto the quantum state
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    # Entangle them to find hidden patterns (e.g., Festival vs. Stock)
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    # Measure the result
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

class HybridQuantumGNN(nn.Module):
    def __init__(self, in_dim=7, hidden_dim=64):
        super().__init__()
        # 1. Classical Pre-processing (Maps 7 inputs -> 7 Qubits)
        self.pre_net = nn.Linear(in_dim, n_qubits)
        
        # 2. Quantum Layer (The VQC)
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, {"weights": (2, n_qubits)})
        
        # 3. Post-Processing & Graph Neural Network
        self.post_net = nn.Linear(n_qubits, hidden_dim)
        self.classical_bypass = nn.Linear(in_dim, hidden_dim)
        
        # Graph Attention Layers (The "Neighbor Awareness")
        self.gat1 = GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=True, edge_dim=1)
        self.gat2 = GATv2Conv(hidden_dim * 4, 1, heads=1, concat=False, edge_dim=1)

    def forward(self, data):
        # Unpack data directly from the object
        # (Works for both Training Tensors and Live API Input)
        x, ei, ea = data.x, data.edge_index, data.edge_attr
        
        # Quantum Pass
        x_in = torch.sigmoid(self.pre_net(x)) * np.pi 
        x_q = self.post_net(self.q_layer(x_in))
        
        # Classical Pass (Skip Connection)
        x_c = self.classical_bypass(x)
        
        # Combine
        x = F.elu(x_c + 0.1 * x_q)
        
        # Graph Convolution
        edge_weight = ea[:, 0].unsqueeze(-1)
        x = F.elu(self.gat1(x, ei, edge_weight))
        x = torch.sigmoid(self.gat2(x, ei, edge_weight))
        return x