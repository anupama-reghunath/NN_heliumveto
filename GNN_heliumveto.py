#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python
# coding: utf-8

import glob
import uproot
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import knn
from torch_geometric.nn import knn_graph
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LogNorm
import math
from sbtveto.model.gnn_model import EncodeProcessDecode
from matplotlib.patches import Patch, Circle
import matplotlib as mpl



# In[3]:



colors={"muDIS+MuinducedBG":'#4daf4a', "neuDIS+MuinducedBG":'#e41a1c',"Signal+MuinducedBG": '#377eb8'}
shapes={"muDIS+MuinducedBG":'^', "neuDIS+MuinducedBG":'*',"Signal+MuinducedBG": 'x'}
mpl.rcParams['font.size'] = 15.0


from pathlib import Path

output_dir = Path("plots")
output_dir.mkdir(exist_ok=True)


# In[4]:


def print_composition_donut_gnn(X, Y, filename='GNN_DONUT.png'):
    """
    Plot the composition of events in donut format for GNN-format data.
    Args:
        X: shape (N_events, 854, N_features), first column is SBT energy
        Y: shape (N_events,), class labels
    """
    # 1) Define class labels and truth‐values
    class_info = [
        ("muDIS+MuinducedBG", 2),
        ("neuDIS+MuinducedBG", 1),
        ("Signal+MuinducedBG", 0),
    ]

    # 2) Compute counts table
    table = []
    for cls_name, cls_val in class_info:
        mask = (Y == cls_val)
        hits = np.sum(X[mask][:, :, 0] > 0, axis=1) >= 1  # SBT hits above zero
        table.append({
            "class":     cls_name,
            "with_SBT":  int(hits.sum()),
            "no_SBT":    int((~hits).sum()),
        })

    # 3) Flatten into plotting lists
    sizes, colors_list, hatches = [], [], []
    for row in table:
        for cond in ("with_SBT", "no_SBT"):
            sizes.append(row[cond])
            colors_list.append(colors[row["class"]])
            hatches.append(".." if cond == "with_SBT" else "")

    # 4) Draw the donut chart
    fig, ax = plt.subplots(figsize=(10,7))
    wedges, _, autotexts = ax.pie(
        sizes, labels=None, colors=colors_list,
        autopct='%1.1f%%', pctdistance=0.75, startangle=90,
        wedgeprops={'width':0.4, 'edgecolor':'white'}
    )

    # White centre
    ax.add_artist(Circle((0,0), 0.60, color='white', linewidth=0))

    # 5) Hatching and text formatting
    for w, h, at in zip(wedges, hatches, autotexts):
        if h: w.set_hatch(h)
        at.set_color('white')
        at.set_fontweight('bold')

    # 6) Class labels positioned around donut
    mids = [(w.theta1 + wedges[i+1].theta2)/2 for i, w in enumerate(wedges[::2])]
    for angle, (cls_name, _) in zip(mids, class_info):
        rad = np.deg2rad(angle)
        x, y = np.cos(rad)*1.15, np.sin(rad)*1.15
        ax.text(x, y, cls_name, rotation=angle-90, rotation_mode='anchor',
                ha='center', va='center', color=colors[cls_name], fontweight='bold')

    # 7) Total count in centre
    ax.text(0.5, 0.5, f"Total\n{sum(sizes)}", transform=ax.transAxes,
            ha='center', va='center', fontsize=18, fontweight='bold')

    # 8) Legend
    hatch_handles = [
        Patch(facecolor='white', edgecolor='black', label='No SBT hits'),
        Patch(facecolor='white', edgecolor='black', hatch='..', label='≥1 SBT hit'),
    ]
    ax.legend(handles=hatch_handles, title='SBT condition',
              loc='center left', bbox_to_anchor=(1.1, 0.5), frameon=False)

    plt.tight_layout()
    plt.savefig(f"plots/{filename}")
    print(f"Saved to plots/{filename}")

    # 9) Print table
    print(f"{'':20}\t| {'noSBT':5}\t{'withSBT':5}\t{'total':5}")
    print("─────────────────────────────────────────────────────")
    for row in table:
        no, yes = row['no_SBT'], row['with_SBT']
        print(f"{row['class']:20}\t| {no:5}\t{yes:5}\t{no + yes:5}")
    comb_no  = sum(r['no_SBT']   for r in table)
    comb_yes = sum(r['with_SBT'] for r in table)
    print("─────────────────────────────────────────────────────")
    print(f"{'Combined':20}\t| {comb_no:5}\t{comb_yes:5}\t{comb_no + comb_yes:5}")


# In[5]:


def plot_SBT(XYZ):

    def GetPhi(x,y):
     
        r=math.sqrt(x*x+y*y)

        if(y>=0):   phi =   math.acos(x / r)
        else:       phi =-1*math.acos(x/r)+2*math.pi
            
        phi=phi*180/ math.pi

        if phi>=360: phi=phi-360

        
        if phi>270: return phi-270
        else:       return phi+90 #offset to start the reading from bottom centre
    
    # Create a figure and axis
    fig, axs6 = plt.subplots(figsize=(6, 4), dpi=200)
    
    fig.tight_layout() 
    axs6.set_title('SBT cell positioning')
    
    detz_full = XYZ[2]
    
    indices = np.arange(len(detz_full))
    
    detphi_full = [GetPhi(x, y) for x, y in zip(XYZ[0], XYZ[1])]

    sc=axs6.scatter(detz_full,detphi_full, c=indices,alpha=1,rasterized=True,edgecolors='black', cmap='jet')
    
    # Annotate each point with its index
    for idx, (zv, phiv) in enumerate(zip(detz_full, detphi_full)):
        axs6.text(zv, phiv, str(idx), fontsize=3, alpha=1,ha='center',va='center',color='black')
    
    # Add colorbar
    cbar = plt.colorbar(sc)
    cbar.set_label("Index")
        
    axs6.axhline(y=33.71,linestyle='--',alpha=0.25,color='black')
    axs6.text(-2500 ,33.71-10,"Bottom")
    axs6.axhline(y=33.71+56.408*2,linestyle='--',alpha=0.25,color='black')
    axs6.text(-2500 ,(33.71+56.408*2)-10,"Left")
    axs6.axhline(y=33.71*3+56.408*2,linestyle='--',alpha=0.25,color='black')
    axs6.text(-2500 ,(33.71*3+56.408*2)-10,"Top")
    axs6.axhline(y=33.71*3+56.408*4,linestyle='--',alpha=0.25,color='black')
    axs6.text(-2500 ,(33.71*3+56.408*4)-10,"Right")
    
    axs6.set_xlim(-2800,2800);axs6.set_ylim(-10,370)
    axs6.set_xlabel('z (cm)');axs6.set_ylabel('\u03C6')
    plt.savefig(f"{output_dir}/GNNSBTpos.png")
    #plt.show()    
    


# In[6]:


def prepare_data(data_path, XYZ, max_files=None):

    def load_and_process(file_paths, label, max_files=None):
        processed_data = []
        signal_info_list = []
        labels = []

        for i, file_path in enumerate(file_paths[:max_files]):
            try:
                file = uproot.open(file_path)
                x = np.array(file['tree;1']['inputmatrix'].array(library="np"))

                energy_dep_sbt = x[:, :854]
                hittime_sbt    = x[:, 854:1708]  
                vertex_pos     = x[:, 1708:1711] # x,y,z
                vertex_time    = x[:, 1711:1712]
                eventweight    = x[:, 1712:1713]
                candidate_info = x[:, 1713:1723]
                ubt_hits       = x[:, 1723:1724]

                # Signal-level info (kept once per event)
                signal_info = np.hstack([vertex_pos, candidate_info, ubt_hits, eventweight])
                signal_info_list.append(signal_info)

                # Prepare repeated node-level features
                N = x.shape[0]
                features = np.vstack([
                    np.expand_dims(energy_dep_sbt, 0),                       # (1, N, 854)
                    #np.expand_dims(hittime_sbt, 0),                         # (1, N, 854)
                    np.expand_dims(np.repeat(XYZ[0:1, :], N, axis=0), 0),     # X
                    np.expand_dims(np.repeat(XYZ[1:2, :], N, axis=0), 0),     # Y
                    np.expand_dims(np.repeat(XYZ[2:3, :], N, axis=0), 0),     # Z
                    #np.expand_dims(np.repeat(vertex_time[:, 0], 854, axis=1), 0),
                    np.expand_dims(np.repeat(vertex_pos[:, 0:1], 854, axis=1), 0),
                    np.expand_dims(np.repeat(vertex_pos[:, 1:2], 854, axis=1), 0),
                    np.expand_dims(np.repeat(vertex_pos[:, 2:3], 854, axis=1), 0),
                    #np.expand_dims(np.repeat(candidate_info[:, 0:1], 854, axis=1), 0),  # IP
                    #np.expand_dims(np.repeat(candidate_info[:, 1:2], 854, axis=1), 0),  # DOCA
                    #np.expand_dims(np.repeat(candidate_info[:, 2:3], 854, axis=1), 0),  # Impact Param
                ])

                features = np.swapaxes(features, 0, 1)  # (N, features, 854)
                features = np.swapaxes(features, 1, 2)  # (N, 854, features)

                processed_data.append(features)
                labels.append(np.full(N, label))

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

        return (
            np.concatenate(processed_data, axis=0),
            np.concatenate(labels),
            np.concatenate(signal_info_list, axis=0)
        )
    
    
    
    #plot_SBT(XYZ)

    neu_files  = glob.glob(f"{data_path}/NNdata_neuDIS_MuBack_batch_0_*.root")
    mu_files   = glob.glob(f"{data_path}/NNdata_muDIS_MuBack_batch_0_*.root")
    embg_files = glob.glob(f"{data_path}/NNdata_signal_MuBack_batch_0_*.root")

    print(f"Number of datafiles available:\n\tneuDIS+MuBack:\t{len(neu_files)}\n\tmuDIS+MuBack:\t{len(mu_files)}\n\tSignal+MuBack:\t{len(embg_files)}")

    embg_X, embg_Y, embg_sig = load_and_process(embg_files, 0, max_files)
    neu_X, neu_Y, neu_sig     = load_and_process(neu_files, 1, max_files)
    mu_X, mu_Y, mu_sig        = load_and_process(mu_files, 2, max_files)

    # Combine
    X = np.concatenate([embg_X, neu_X, mu_X], axis=0)
    Y = np.concatenate([embg_Y, neu_Y, mu_Y], axis=0)
    signal_info = np.concatenate([embg_sig, neu_sig, mu_sig], axis=0)

    print(f"Final data shape: X={X.shape}, Y={Y.shape}, signal_info={signal_info.shape}")
    return X, Y, signal_info


# In[7]:


data_path="/eos/experiment/ship/user/anupamar/NN_data/root_files/wMuonBack"
XYZ = np.load("/afs/cern.ch/user/a/anupamar/Analysis/NNSBTveto/NN_heliumveto/SBT_new_geo_XYZ.npy")
X,Y,signal_info=prepare_data(data_path, XYZ, max_files=None)


# In[8]:


print_composition_donut_gnn(X, Y,filename='GNN_DONUT.png')


# In[9]:


plot_SBT(XYZ)


# In[10]:


# Adjacency helper functions
def adjacency2(n_dau):
    """Generates a fully connected adjacency (all-to-all)."""
    return np.ones((n_dau + 1, n_dau + 1))


# In[11]:


threshold = 45  # in MeV

# Extract the energy deposits (column 0 in each event)#    X shape: (N_events, 854, N_features)
energies = X[:, :, 0]  # Shape: (N_events, 854)

mask = np.any(energies > threshold * 1e-3, axis=1)

X_filtered = X[mask]
Y_filtered = Y[mask]
signal_info_filtered = signal_info[mask]

print(f"Events remaining after {threshold} MeV threshold: {len(X_filtered)}")


# In[12]:


print_composition_donut_gnn(X_filtered, Y_filtered,filename='GNN_DONUT_filtered.png')


# In[12]:


#############################################
# 3D Scatter Plot of sample event
#############################################

event_idx = 1
my_event = X_filtered[event_idx]  # Shape: (854, features)
print("Shape for the selected event:", my_event.shape)


fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection='3d')


# Extract data
hits = my_event[:, 0]
Xcoord = my_event[:, 1]
Ycoord = my_event[:, 2]  
Zcoord = my_event[:, 3]
nonzero_mask = hits > 0

Xcoord = Xcoord[nonzero_mask]
Ycoord = Ycoord[nonzero_mask]
Zcoord = Zcoord[nonzero_mask]
hits    = hits[nonzero_mask]

colors_ = hits  

sc = ax.scatter(Xcoord, Zcoord, Ycoord, c=colors_, cmap='cool', s=5, vmin=threshold* 1e-3, vmax=np.max(hits))
cbar = plt.colorbar(sc, ax=ax, pad=0.1)
cbar.set_label("Energy Deposition(GeV)")

ax.set_xlabel("X")
ax.set_ylabel("Z")
ax.set_zlabel("Y (up)")
ax.set_title(f"3D SBT Layout - Event {event_idx}")
plt.tight_layout()
plt.savefig(f"{output_dir}/GNN_3Dlayout.png")
#plt.show()

#############################################
# 3D with k-NN Edges
#############################################

use_knn = True
if use_knn:
        
    # Filter only fired cells
    nonzero_mask = hits > 0

    Xcoord = Xcoord[nonzero_mask]
    Ycoord = Ycoord[nonzero_mask]
    Zcoord = Zcoord[nonzero_mask]
    hits    = hits[nonzero_mask]

    colors_ = hits  

    positions = torch.tensor(np.stack([Xcoord, Zcoord, Ycoord], axis=1), dtype=torch.float)

    hits_t = torch.tensor(hits, dtype=torch.float)

    edge_index = knn_graph(positions, k=5)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(Xcoord, Zcoord, Ycoord, c=colors_, cmap='cool', s=5, vmin=threshold* 1e-3, vmax=np.max(hits))

    senders = edge_index[0].numpy()
    receivers = edge_index[1].numpy()

    for s, r in zip(senders, receivers):
        xs = [Xcoord[s], Xcoord[r]]
        zs = [Zcoord[s], Zcoord[r]]
        ys = [Ycoord[s], Ycoord[r]]
        ax.plot(xs, zs, ys, color='gray', alpha=0.3, linewidth=0.5)

    cbar = plt.colorbar(sc, ax=ax, pad=0.1)
    cbar.set_label("Energy Deposition(GeV)")

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y (up)")
    ax.set_title(f"Event {event_idx} with k-NN edges (k=5)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/GNN_3Dknnlayout.png")
    #plt.show()


# In[13]:


X_temp, X_test, Y_temp, Y_test, sig_temp, sig_test = train_test_split( X_filtered, Y_filtered, signal_info_filtered, test_size=0.2, random_state=42)


# In[14]:

try:
    print_composition_donut_gnn(X_test,Y_test,"GNN_DONUT_testdata.png")
except:
    print("Unable to print X_test composition donut")

# In[15]:


#split temp to train and validate
X_train, X_val, Y_train, Y_val, sig_train, sig_val = train_test_split( X_temp, Y_temp, sig_temp, test_size=0.25, random_state=42)

del X_temp,Y_temp,sig_temp


# In[16]:

#############################################
# 2) Building Graph Datasets
#############################################

def create_graph_data(X_SBT, X_sig, Y_labels):
    """
    Convert processed arrays into a list of torch_geometric Data objects.
    """
    data_list = []
    global_idx = 0

    for i in range(X_SBT.shape[0]):
        
        Xcon = X_SBT[i][X_SBT[i][:, 0] > 0] # Filter out zero-hits; shape => [N_hits, 4]
        
        phi_column = np.expand_dims(np.arctan2(Xcon[:, 2], Xcon[:, 1]), axis=1) # Add the 'phi' column
        Xcon = np.hstack([Xcon, phi_column])


        if Xcon.shape[0] < 1:
            continue         # Skip if no data

        # Node features
        Xcon_torch = torch.tensor(Xcon, dtype=torch.float)

        # Compute adjacency
        
        if Xcon.shape[0] < 22: # if less than threshold, fully connect them  
            A = adjacency2(Xcon.shape[0] - 1)
            edge_index = torch.tensor(A, dtype=torch.float).nonzero().t().contiguous()
        else:
            # K-NN adjacency
            k = 20
            edge_index = knn(Xcon_torch, Xcon_torch, k)
            # remove self-connections
            edge_index = edge_index[:, edge_index[0] != edge_index[1]]

        # Edge features: r, delta_z, delta_phi
        senders = edge_index[0].numpy()
        receivers = edge_index[1].numpy()

        
        r = np.sqrt(np.sum((Xcon[senders, 1:4] - Xcon[receivers, 1:4]) ** 2, axis=1))
        
        delta_z = Xcon[senders, 3] - Xcon[receivers, 3]
        
        phi_senders = np.arctan2(Xcon[senders, 2], Xcon[senders, 1])
        phi_receivers = np.arctan2(Xcon[receivers, 2], Xcon[receivers, 1])
        delta_phi = phi_senders - phi_receivers

        edge_features = np.vstack([r, delta_z, delta_phi]).T
        edge_features_torch = torch.tensor(edge_features, dtype=torch.float)


        global_features = np.hstack([[Xcon.shape[0]]])
        global_features_torch = torch.tensor(global_features, dtype=torch.float).unsqueeze(0)

        sig_vars_torch = torch.tensor(X_sig[i], dtype=torch.float).unsqueeze(0)


        Y_torch = torch.tensor(Y_labels[i], dtype=torch.long)


        data = Data(
            nodes=Xcon_torch,
            edge_index=edge_index,
            edges=edge_features_torch,
            graph_globals=global_features_torch,
            sig_vars=sig_vars_torch,
            y=Y_torch
        )

        data["receivers"] = data.edge_index[1]
        data["senders"] = data.edge_index[0]


        data_list.append(data)
        global_idx += 1

    return data_list

train_data      = create_graph_data(X_train, sig_train, Y_train)
val_data        = create_graph_data(X_val, sig_val, Y_val)
test_data       = create_graph_data(X_test, sig_test, Y_test)


# Build loaders
train_loader    = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader      = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader     = DataLoader(test_data, batch_size=32, shuffle=False)

print("Number of train samples:", len(train_data))
print("Number of val samples:", len(val_data))


# In[ ]:


#############################################
# 3) Define/Initialize the Model
#############################################

STEP_SIZE = 2
MLP_OUTPUT_SIZE = 8
HIDDEN_CHANNELS = 64
NUM_LAYERS = 4

model = EncodeProcessDecode(
    mlp_output_size=MLP_OUTPUT_SIZE,
    global_op=3,        # 3 output classes
    num_blocks=4        # e.g. 4 message-passing steps
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


#############################################
# 4) Train the Model
#############################################

optimizer_GCN = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

losses = []
vallosses = []
nepochs=25
for epoch in range(1,nepochs+1):
    if epoch == 20:
        optimizer_GCN = torch.optim.Adam(model.parameters(), lr=0.0001)

    model.train()
    epoch_loss = 0

    for b in train_loader:
        b = b.to(device)
        b['receivers'] = b.edge_index[1]
        b['senders'] = b.edge_index[0]
        b['edgepos'] = b.batch[b['senders']]

        optimizer_GCN.zero_grad()

        out = model(b)
        logits = out['graph_globals']  # [batch_size, num_classes]
        loss = criterion(logits, b.y)  # b.y: [batch_size] with class indices

        loss.backward()
        optimizer_GCN.step()

        epoch_loss += loss.item() / len(train_loader)

    model.eval()
    val_epoch_loss = 0

    with torch.no_grad():
        for b in val_loader:
            b = b.to(device)
            b['receivers'] = b.edge_index[1]
            b['senders'] = b.edge_index[0]
            b['edgepos'] = b.batch[b['senders']]

            out = model(b)
            logits = out['graph_globals']
            val_loss = criterion(logits, b.y)
            val_epoch_loss += val_loss.item() / len(val_loader)

    print(f"Epoch {epoch:02d} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f}")
    losses.append(epoch_loss)
    vallosses.append(val_epoch_loss)

model_path = f"GNN_SBTveto_model_newgeo_{threshold}MeV_{nepochs}epochs.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

#############################################
# 5) Evaluate + Confusion Matrix
#############################################

model.eval()
y_true = []
y_pred = []

with torch.no_grad():

    for b in test_loader:
        b = b.to(device)

        b['receivers'] = b.edge_index[1]
        b['senders'] = b.edge_index[0]
        b['edgepos'] = b.batch[b['senders']]

        out = model(b)
        logits = out["graph_globals"]  # shape: [1, num_classes]

        _, predicted = torch.max(logits, dim=1)  # returns scalar label

        y_true.extend(b.y.cpu().tolist())
        y_pred.extend(predicted.cpu().tolist())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Confusion Matrix
labels = ["Signal", r"$\nu$DIS", r"$\mu$DIS"]
cm = confusion_matrix(y_true, y_pred)
cm_norm = cm / cm.sum(axis=1, keepdims=True)  # row-normalized

print("Confusion Matrix (unnormalized):\n", cm)
print("Confusion Matrix (row-normalized):\n", cm_norm)
accuracy = np.sum(y_true == y_pred) / len(y_true)
print(f"Validation Accuracy: {accuracy:.4f}")

# Plot confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(cm_norm, annot=True, fmt=".2f", 
            xticklabels=labels, yticklabels=labels, 
            cmap="Blues", norm=LogNorm())
plt.title("Confusion Matrix (Validation)")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
plt.savefig(f"{output_dir}/GNN_CM.png")
##plt.show()


# In[ ]:


#############################################
# 6) Plot Training Curves
#############################################

epochs_range = range(1, nepochs + 1)
plt.figure(figsize=(6, 4))
plt.plot(epochs_range, losses, label="Train Loss")
plt.plot(epochs_range, vallosses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training & Validation Loss")
plt.tight_layout()
plt.savefig(f"{output_dir}/GNN_Losscurve.png")
#plt.show()

