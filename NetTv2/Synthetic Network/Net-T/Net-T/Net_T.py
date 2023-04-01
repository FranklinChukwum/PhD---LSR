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

from dgl.nn.pytorch import GraphConv
from matplotlib import rcParams
matplotlib.matplotlib_fname()



# 1. Graph Creation:

G = dgl.DGLGraph() 

G.add_nodes(15) # populate the graph and add edges
G.add_edges([2,3,4,5,6,7,8,9,10,11,12,13,14,15], 0) # super user 1
G.add_edges([2,4,5,6,7,8,9,10,11,12,13,14,15], 1) # super user 2
G.add_edges([2,4,5,15], 3) # first mirror agent
G.add_edges([7,8,10,14], 6) # second mirror agent
G.add_edges([8,9,10,11,13,15], 12) # third mirror agent 


G = dgl.add_self_loop(G) # allows for users to include their features in aggregate of the features of neighbor users


# Print out the number of nodes and edges in our newly constructed graph:

print ('')

print('We have %d users.' % G.number_of_nodes())
print('We have %d connections.' % G.number_of_edges())

# For In-degree (Degree Centrality)
nx_G = G.to_networkx()
nx.degree(nx_G)

print ('')
print ('User O has an in-degree of: %d' % G.in_degrees(0))
print ('User 1 has an in-degree of: %d' % G.in_degrees(1))


   
# plot and display initial graph
nx_G = G.to_networkx()

plt.figure(figsize=(14, 8))
pos = nx.random_layout(nx_G)

nx.draw_networkx_nodes(nx_G, pos, node_size = 2500, node_color = '#2F3030', alpha = 0.3)
nx.draw_networkx_edges(nx_G, pos, style = 'solid', alpha = 0.7)

nx.draw_networkx_labels(nx_G, pos, font_size = 20, font_color = 'k', font_family = 'sans-serif', 
                        font_weight='bold', alpha = None, ax = None )

plt.axis('off')
plt.tight_layout()
plt.show()

# 2. User Features/Profiling:

embed = nn.Embedding(16, 8)  # 16 nodes with embedding dimension equal to 8
G.ndata['feat' 'boy'] = embed.weight

# print a node input feature - node 7
print(G.ndata['feat' 'boy'][7])



# 3. Define GCN

class GCN(nn.Module):
    def __init__(self, input_features, hidden_size, hidden_size2, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(input_features, hidden_size)
        self.conv2 = GraphConv(hidden_size, hidden_size2)
        self.conv3 = GraphConv(hidden_size2, hidden_size2)
        self.classify = nn.Linear(hidden_size2, num_classes) # output layer

        print(f"User Classes:  {num_classes}")

    def forward(self, g, inputs):
        x = F.relu(self.conv1(g, inputs))
        x = F.relu(self.conv2(g, x))
        x = F.relu(self.conv3(g,x)) 
        #print(f"x.size, {x.size()}")
        return self.classify(x)

nnet = GCN(8, 8, 8, 2) # output layer feature of size 2
print (nnet)


# 4. Initialization

inputs = embed.weight
profile_init = torch.tensor([0, 1, 3, 6, 12])  # users with profiles
profiles = torch.tensor([1, 0, 0, 0, 1])  # their profiles are different



# 5 Network Training  and Evaluation  

optimizer = torch.optim.Adam(itertools.chain(nnet.parameters(), embed.parameters()), lr=0.01)
all_logits = []
epoch_losses = [] # for losses over the epoch counts
for epoch in range(100):
    epoch_loss = 0
    nnet.train()
    logits = nnet(G, inputs)
   
    all_logits.append(logits.detach())
    logp = F.log_softmax(logits, 1)

    # for users with profiles
    loss = F.nll_loss(logp[profile_init], profiles) # negative log liklihood function as the loss function

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    epoch_loss += loss.detach().item()
   

    if epoch % 5 == 0:
        print('Epoch {:02d} | loss {:.4f}'.format(epoch, epoch_loss))

    epoch_losses.append(epoch_loss)


# view model
def view_model(i):
    color1 = '#30731A'
    color2 = '#0B55E8'
    pos = {}
    colors = []
    for v in range(16):
        pos[v] = all_logits[i][v].numpy()
        cls = pos[v].argmax()
        colors.append(color1 if cls else color2)
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
view_model(74)  # draw the prediction of the epoch numbered - this can be changed
plt.show()

plt.xlabel('Epoch count')
plt.ylabel('Loss')   
plt.title('Likelihood Loss over 100 epochs')
plt.plot(epoch_losses)
plt.show()

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'ro')

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
ani = animation.FuncAnimation(fig, view_model, frames=len(all_logits), interval=100)
plt.show()

ani.save('C:/Users/Frank/Desktop/Data_Set/numer6008.gif', writer='imagemagick', fps=30)


