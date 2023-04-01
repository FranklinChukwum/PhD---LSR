import dgl
import numpy as np
import networkx as nx
import torch 
import torch.nn as nn
import torch.nn.functional as F
import itertools
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch.optim as optim

from matplotlib import rc
from dgl.nn.pytorch import GraphConv



def build_karate_club_graph():
    
    src = np.array([1, 2, 2, 3, 3, 3, 4, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 9, 10, 10,
        10, 11, 12, 12, 13, 13, 13, 13, 16, 16, 17, 17, 19, 19, 21, 21,
        25, 25, 27, 27, 27, 28, 29, 29, 30, 30, 31, 31, 31, 31, 32, 32,
        32, 32, 32, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 33, 33,
        33, 33, 33, 33, 33, 33, 33, 33, 33, 33])
    dst = np.array([0, 0, 1, 0, 1, 2, 0, 0, 0, 4, 5, 0, 1, 2, 3, 0, 2, 2, 0, 4,
        5, 0, 0, 3, 0, 1, 2, 3, 5, 6, 0, 1, 0, 1, 0, 1, 23, 24, 2, 23,
        24, 2, 23, 26, 1, 8, 0, 24, 25, 28, 2, 8, 14, 15, 18, 20, 22, 23,
        29, 30, 31, 8, 9, 13, 14, 15, 18, 19, 20, 22, 23, 26, 27, 28, 29, 30,
        31, 32])
    # Directional edges
    u = np.concatenate([src, dst])
    v = np.concatenate([dst, dst])
  
    return dgl.DGLGraph((u, v))   # Build a DGLGraph

# Print out the number of nodes and edges 

G = build_karate_club_graph()
G = dgl.add_self_loop(G) # allows for effective feature aggregation
print('We have %d nodes.' % G.number_of_nodes())
print('We have %d edges.' % G.number_of_edges())

# For In-degree (Degree Centrality)
nx_G = G.to_networkx()
nx.degree(nx_G)

print ('')

print ('User O has an in-degree of: %d' % G.in_degrees(0))
print ('User 1 has an in-degree of: %d' % G.in_degrees(1))

# plot and display initial graph
nx_G = G.to_networkx()

pos = nx.random_layout(nx_G)

nx.draw_networkx_nodes(nx_G, pos, node_size = 2500, node_color = '#2F3030', alpha = 0.3)
nx.draw_networkx_edges(nx_G, pos, style = 'solid', alpha = 0.7)

nx.draw_networkx_labels(nx_G, pos, font_size = 20, font_color = 'k', font_family = 'arial', 
                        font_weight='bold', alpha = None, ax = None )

plt.axis('off')
plt.tight_layout()
plt.show()

# 2. User Features/Profiling:

embed = nn.Embedding(34, 8)  # 34 nodes with embedding dimension equal to 8
print (embed.weight)

G.ndata['radar'] = embed.weight

# print a node input feature - node 7
print(G.ndata['radar'][7])


# 3. Define GCN:

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
profile_init = torch.tensor([0, 3, 5, 8, 23, 33])  #  users with profiles
profiles = torch.tensor([1, 0, 0, 0, 1, 0])  # their profiles are different


# 5 Network Training

optimizer = torch.optim.Adam(itertools.chain(nnet.parameters(), embed.parameters()), lr=0.01)

all_logits = []
epoch_losses = []
for epoch in range(100):
    epoch_loss = 0
    nnet.train()
    logits = nnet(G, inputs)
   
    all_logits.append(logits.detach())
    logp = F.log_softmax(logits, 1)

    # for users with profiles
    loss = F.nll_loss(logp[profile_init], profiles)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    epoch_loss += loss.detach().item()
   

    if epoch % 5 == 0:
        print('Epoch {:02d} | loss {:.4f}'.format(epoch, epoch_loss))

    epoch_losses.append(epoch_loss)

plt.xlabel('Epoch count')
plt.ylabel('Loss')   
plt.title('Likelihood Loss over 100 epochs')
plt.plot(epoch_losses)
plt.show()

# view model
def view_model(i):
    color1 = '#30731A'
    color2 = '#0B55E8'
    pos = {}
    colors = []
    for v in range(34):
        pos[v] = all_logits[i][v].numpy()
        cls = pos[v].argmax()
        colors.append(color1 if cls else color2)
    ax.cla()
    ax.axis('off')
    ax.set_title('Epoch: %d' % i)
    #nx.draw_networkx(nx_G, pos, node_color=colors, with_labels=True, node_size=600, ax=ax)

    nx.draw_networkx_nodes(nx_G, pos, node_size = 1000, node_color = colors)
    nx.draw_networkx_edges(nx_G, pos, style = 'solid', alpha = 0.7)

    nx.draw_networkx_labels(nx_G, pos, font_size = 15, font_color = 'white', font_family = 'sans-serif', 
                        font_weight='bold', alpha = None, ax = None )

fig = plt.figure(dpi=150)
fig.clf()
ax = fig.subplots()
view_model(99)  # draw the classification of the epoch numbered - this can be changed
plt.show()



fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [])

def init():
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1, 1)
    return ln,

def update(frame):
    xdata.append(frame)
    ydata.append(np.sin(frame))
    ln.set_data(xdata, ydata)
    return ln,

# animation

ani = animation.FuncAnimation(fig, view_model, frames=len(all_logits), interval=200)
plt.show()

# save animation
ani.save('C:/Users/Frank/Desktop/Data_Set/numer678.gif', writer='imagemagick', fps=30)