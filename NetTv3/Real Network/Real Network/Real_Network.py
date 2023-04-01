import dgl
import numpy as np
import networkx as nx
import torch 
import torch.nn as nn
import torch.nn.functional as F
import itertools
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
import torch.optim as optim
import argparse, time, math
import pandas as pd
import os


from IPython.display import HTML
from dgl.data import DGLDataset
from dgl.nn import SAGEConv
from matplotlib import rcParams
matplotlib.matplotlib_fname()

# Input Module   

class KarateClubDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='karate_club')

    def process(self):
        nodes_data = pd.read_csv('C:/Users/Frank/Desktop/Data_Set/Main/nodes-Zac.csv')
        edges_data = pd.read_csv('C:/Users/Frank/Desktop/Data_Set/Main/edge.csv')

           # Prepare the belief and bias node features
        print ('')
        features = torch.from_numpy(nodes_data['Belief'].to_numpy())
        print (features)
    
        features1 = torch.from_numpy(nodes_data['Bias'].to_numpy())
        print ('')
        print (features1)
        print ('')

        #prepare the labels of the nodes

        # The "Class" column represents which label of each node 
        node_labels = torch.from_numpy(nodes_data['Class'].astype('category').cat.codes.to_numpy())
        print ('')
        print (node_labels)

        edge_features = torch.from_numpy(edges_data['Weight'].to_numpy())
        edges_src = torch.from_numpy(edges_data['Src'].to_numpy())
        edges_dst = torch.from_numpy(edges_data['Dst'].to_numpy())

        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=nodes_data.shape[0])
        self.graph.ndata['feat'] = features
        self.graph.ndata['boy'] = features1
        #self.graph.ndata['label'] = node_labels
        self.graph.ndata['label'] = node_labels.type(torch.LongTensor)
        self.graph.edata['weight'] = edge_features

        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        n_nodes = nodes_data.shape[0]
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.20)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

dataset = KarateClubDataset()
graph = dataset[0]


graph = dgl.add_self_loop(graph) # allows for users to include their features in aggregate of the fearures of neighbor users


print ('')
print(graph)




print ('')
print ('User O has an in-degree of: %d' % graph.in_degrees(0))
print ('User 1 has an in-degree of: %d' % graph.in_degrees(1))
print ('User 3 has an in-degree of: %d' % graph.in_degrees(3))
print ('User 11 has an in-degree of: %d' % graph.in_degrees(11))




nx_G = graph.to_networkx()

plt.figure(figsize=(14, 8))
pos = nx.kamada_kawai_layout(nx_G)

nx.draw_networkx_nodes(nx_G, pos, node_size = 2000, node_color = '#2F3030', alpha = 0.3)
nx.draw_networkx_edges(nx_G, pos, style = 'solid', alpha = 0.7)

nx.draw_networkx_labels(nx_G, pos, font_size = 20, font_color = 'k', font_family = 'sans-serif', 
                        font_weight='bold', alpha = None, ax = None )

plt.axis('off')
plt.tight_layout()
plt.show()



# Labeling Module




print ('')
print('Node features')
print(graph.ndata)
print('Edge features')
print(graph.edata)



def new_func():
    node_embed = nn.Embedding(34, 8)  # Every node has an embedding of size 8.
    return node_embed

node_embed = new_func()
inputs = node_embed.weight                         # Use the embedding weight as the node features.
nn.init.xavier_uniform_(inputs)
print ('')
print("Tensor Shape")
print(inputs)
print ('')

features1 = graph.ndata['boy']
labeled_nodes = [0, 1]
print('Labels', features1[labeled_nodes])
print ('')
print(graph.ndata['boy'][3])
print(graph.ndata['boy'][6])


print ('')



print(inputs[3])
print(inputs[9])
print(inputs[17])
print(inputs[33])

print ('')

#NN Module

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats, h_feat2, num_classes):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, h_feat2, 'mean')
        self.conv3 = SAGEConv(h_feat2, num_classes, 'mean')
    
    def forward(self, graph, in_feat):
        h1 = self.conv1(graph, in_feat)
        h1 = F.relu(h1)
        h2 = self.conv2(graph, h1)
        h2 = F.relu(h1)
        h3 = self.conv3(graph, h2)
        return h3
    
# Create the model with given dimensions 
nnet = GraphSAGE(8, 16, 16, 2)



# Output Module


optimizer = torch.optim.Adam(itertools.chain(nnet.parameters(), node_embed.parameters()), lr=0.01)
best_train_acc = 0
best_val_acc = 0
best_test_acc = 0

features = graph.ndata['feat']
features1 = graph.ndata['boy']
labels = graph.ndata['label']
train_mask = graph.ndata['train_mask']
val_mask = graph.ndata['val_mask']
test_mask = graph.ndata['test_mask']

all_logits = []
epoch_losses = [] # for losses over the epoch counts
    
for e in range(100):
      # Forward
      epoch_loss = 0
      nnet.train()
      logits = nnet(graph, inputs)

      # Compute prediction
      pred = logits.argmax(1)

      # Compute loss
      # Note that you should only compute the losses of the nodes in the training set.
      loss = F.cross_entropy(logits[train_mask], features1[train_mask])

      # Compute accuracy on training/validation/test
      train_acc = (pred[train_mask] == features1[train_mask]).float().mean()
      val_acc = (pred[val_mask] == features1[val_mask]).float().mean()
      test_acc = (pred[test_mask] == features1[test_mask]).float().mean()

      # Save the best validation accuracy and the corresponding test accuracy.
      if best_train_acc < train_acc:
          best_train_acc = train_acc
          best_val_acc = val_acc
          best_test_acc = test_acc

      # Backward
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      all_logits.append(logits.detach())

      epoch_loss += loss.detach().item()
      
      if e % 5 == 0:
          print('Epoch {}, Loss: {:.4f}, Train Acc: {:.4f} (best {:.4f}), Val Acc: {:.4f} (best {:.4f}), Test Acc: {:.4f} (best {:.3f})'.format(
              e, loss, train_acc, best_train_acc, val_acc, best_val_acc, test_acc, best_test_acc))
    

      epoch_losses.append(epoch_loss)

print ('')


print(inputs[3])
print(inputs[9])
print(inputs[17])
print(inputs[33])

print ('')

pred = torch.argmax(logits, axis=1)
print('Accuracy', (pred == features1).sum().item() / len(pred))

      
      
def view_model(i):
    color1 = '#30731A'
    color2 = '#0B55E8'
    color3 = '#FF00FF'
    color4 = '#800000'
    pos = {}
    colors = []
    for v in range(34):
        pred = all_logits[i].numpy()
        pos[v] = pred[v]
        cls = features[v]
        colors.append(color3 if cls else color4)
    ax.cla()
    ax.axis('off')
    ax.set_title('Epoch: %d' % i)
    #nx.draw_networkx(nx_G, pos, node_color=colors, with_labels=True, node_size=300, ax=ax)

    nx.draw_networkx_nodes(nx_G, pos, node_size = 500, node_color = colors)
    nx.draw_networkx_edges(nx_G, pos, style = 'solid', alpha = 0.7)

    nx.draw_networkx_labels(nx_G, pos, font_size = 15, font_color = 'white', font_family = 'sans-serif', 
                        font_weight='bold', alpha = None, ax = None )

fig = plt.figure(dpi=150)
fig.clf()
ax = fig.subplots()
view_model(0)  # draw the prediction of the epoch numbered - this can be changed
plt.show()


plt.xlabel('Epoch count')
plt.ylabel('Loss')   
plt.title('Cross Entropy over 100 epochs')
plt.plot(epoch_losses)
plt.show()


def init():
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1, 1)
    return ln,

def update(frame):
    xdata.append(frame)
    ydata.append(np.sin(frame))
    ln.set_data(xdata, ydata)
    return ln,


#animation
#ani = animation.FuncAnimation(fig, view_model, frames=len(all_logits), interval=100)
#plt.show()

#ani.save('C:/Users/Frank/Desktop/Data_Set/numer6008.gif', writer='imagemagick', fps=30)
