'''

Graph Convolutional Network (GCN) is a kind of deep learning model for
dealing with graph data, which was came up with by Thomas Kipf and Max
Welling in 2016. GCN uses convolution on graph structure, combining feature
information and structure information of nodes in the graph. GCN is widely
used in Bioinformatics.

GCN clusters neighbor information of each node to update their expressions.
Compared to conventional convolutional neural networks and fully connected
neural networks, GCN is able to deal with nun-Euclidean data and keep topological
relatives between nodes.

Also, GCN uses a normalized Laplace matrix as the core of propagation,
making information propagation more stable and controllable, and GCN
is suitable for modern deep learning architecture, supporting end-to-end
learning.

'''


import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid


dataset = Planetoid(root='/tmp/Cora', name='Cora')

class GCN(torch.nn.Module):
    def __init__(self, I_dim, H_dim, O_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(I_dim, H_dim)
        self.conv2 = GCNConv(H_dim, O_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(I_dim=dataset.num_node_features, H_dim=16, O_dim=dataset.num_classes).to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(100):
    optimizer.zero_grad()
    o = model(data)
    loss = F.nll_loss(o[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, loss: {loss.item():.4f}')

model.eval()
_, pred = model(data).max(dim=1)
corr = int((pred[data.test_mask] == data.y[data.test_mask]).sum())
acc = corr / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')