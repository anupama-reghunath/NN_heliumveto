import glob
import uproot
import numpy as np
from sbtveto.model.nn_model import NN

from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import joblib

import torch
from torch import nn
from torch_geometric.loader import DataLoader
from torch.nn.functional import softmax  

import seaborn as sns

from datetime import datetime
import json

import matplotlib
matplotlib.use("Agg")  # Set non-interactive backend early

import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Patch, Circle

import pandas as pd
colors={"muDIS+MuinducedBG":'#4daf4a', "neuDIS+MuinducedBG":'#e41a1c',"Signal+MuinducedBG": '#377eb8'}
shapes={"muDIS+MuinducedBG":'^', "neuDIS+MuinducedBG":'*',"Signal+MuinducedBG": 'x'}
mpl.rcParams['font.size'] = 15.0


# In[3]:
from pathlib import Path

output_dir = Path("plots")
output_dir.mkdir(exist_ok=True)

def print_composition_donut(inputmatrix, truth,filename='DONUT.png'):
    # 1) Define class labels and truth‐values
    class_info = [
        ("muDIS+MuinducedBG", 2),
        ("neuDIS+MuinducedBG", 1),
        ("Signal+MuinducedBG", 0),
    ]

    # 2) Compute counts table
    table = []
    for cls_name, cls_val in class_info:
        mask = (truth == cls_val)
        hits = np.sum(inputmatrix[mask][:, :854] != 0, axis=1) >= 1
        table.append({
            "class":     cls_name,
            "with_SBT":  int(hits.sum()),
            "no_SBT":    int((~hits).sum()),
        })

    # 3) Flatten into plotting lists
    sizes       = []
    colors_list = []
    hatches     = []
    for row in table:
        for cond in ("with_SBT", "no_SBT"):
            sizes.append(row[cond])
            colors_list.append(colors[row["class"]])
            hatches.append(".." if cond == "with_SBT" else "")

    # 4) Draw the donut
    fig, ax = plt.subplots(figsize=(10,7))
    wedges, _, autotexts = ax.pie(
        sizes,
        labels=None,
        colors=colors_list,
        autopct='%1.1f%%',
        pctdistance=0.75,
        startangle=90,
        wedgeprops={'width':0.4, 'edgecolor':'white'}
    )
    # White centre
    ax.add_artist(Circle((0,0), 0.60, color='white', linewidth=0))

    # 5) Apply hatches and style % texts
    for w, h, at in zip(wedges, hatches, autotexts):
        if h:
            w.set_hatch(h)
        at.set_color('white')
        at.set_fontweight('bold')

    # 6) Place three curved class labels at the mid‐angle of each pair
    mids = []
    for i in range(0, len(wedges), 2):
        a, b = wedges[i].theta1, wedges[i+1].theta2
        mids.append((a+b)/2)
    for angle, (cls_name, _) in zip(mids, class_info):
        rad = np.deg2rad(angle)
        x, y = np.cos(rad)*1.15, np.sin(rad)*1.15
        ax.text(
            x, y, cls_name,
            rotation=angle-90,
            rotation_mode='anchor',
            ha='center', va='center',
            color=colors[cls_name],
            fontweight='bold'
        )

    # 7) Total count in centre
    total = sum(sizes)
    ax.text(
        0.5, 0.5, f"Total\n{total}",
        transform=ax.transAxes,
        ha='center', va='center',
        fontsize=18, fontweight='bold'
    )

     # 8) Simple hatch legend
    hatch_handles = [
        Patch(facecolor='white', edgecolor='black', label='No SBT hits'),
        Patch(facecolor='white', edgecolor='black', hatch='..', label='≥1 SBT hit'),
    ]
    ax.legend(
        handles=hatch_handles,
        title='SBT condition',
        loc='center left',
        bbox_to_anchor=(1.1, 0.5),
        frameon=False
    )

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}")
    #plt.show()
    # 9) Print summary table
    print(f"{'':20}\t| {'noSBT':5}\t{'withSBT':5}\t{'total':5}")
    print("─────────────────────────────────────────────────────")
    for row in table:
        no, yes = row['no_SBT'], row['with_SBT']
        tot      = no + yes
        print(f"{row['class']:20}\t| {no:5}\t{yes:5}\t{tot:5}")
    comb_no  = sum(r['no_SBT']   for r in table)
    comb_yes = sum(r['with_SBT'] for r in table)
    comb_tot = comb_no + comb_yes
    print("─────────────────────────────────────────────────────")
    print(f"{'Combined':20}\t| {comb_no:5}\t{comb_yes:5}\t{comb_tot:5}")


# In[4]:


def prepare_data(data_path):
    """ Prepare Data from ROOT Format"""
    
    def load_and_process_data(file_paths, category, max_files=None):

        processed_data = []#inputmatrix
        labels = []#truth
        #count = 0
        for i, file_path in enumerate(file_paths[:max_files]):
            try:
                #print(f"Loading data from {file_path}")
                file = uproot.open(file_path)
                x = file['tree;1']['inputmatrix'].array(library="np")
                x = np.array(x)
                # Extract relevant features (SBT hits, vertex x, y, z,signal info,len(UBT_hits),weight) #info about timing removed
                energy_dep_sbt=x[:, :854]
                hittime_sbt=x[:, 854:1708]
                vertex_pos =x[:, 1708:1711]
                vertex_time=x[:, 1711:1712]
                eventweight=x[:, 1712:1713]
                candidate_details=x[:, 1713:1723]#IP,DOCAetc
                nhit_ubt   =x[:, 1723:1724]
                x = np.hstack([energy_dep_sbt,
                               vertex_pos ,
                               candidate_details ,
                               nhit_ubt,
                               eventweight])
                processed_data.append(x)
                labels.append(np.full(x.shape[0], category))# Assign labels based on category
                #count += x.shape[0]
            except Exception as e:
                print(f"Skipping {file_path}, Reason:{e}")
                continue
        return np.vstack(processed_data), np.concatenate(labels)

    def load_and_process_data_wtime(file_paths, category, max_files=None):

        processed_data = []#inputmatrix
        labels = []#truth
        #count = 0
        for i, file_path in enumerate(file_paths[:max_files]):
            try:
                #print(f"Loading data from {file_path}")
                file = uproot.open(file_path)
                x = file['tree;1']['inputmatrix'].array(library="np")
                x = np.array(x)
                # Extract relevant features (SBT hits, vertex x, y, z,signal info,len(UBT_hits),weight) #info about timing removed
                energy_dep_sbt=x[:, :854]
                hittime_sbt=x[:, 854:1708]
                vertex_pos =x[:, 1708:1711]
                vertex_time=x[:, 1711:1712]
                eventweight=x[:, 1712:1713]
                candidate_details=x[:, 1713:1723]#IP,DOCAetc
                nhit_ubt   =x[:, 1723:1724]
                x = np.hstack([energy_dep_sbt,
                               hittime_sbt,
                               vertex_pos ,
                               vertex_time,
                               candidate_details ,
                               nhit_ubt,
                               eventweight])
                processed_data.append(x)
                labels.append(np.full(x.shape[0], category))# Assign labels based on category
                #count += x.shape[0]
            except Exception as e:
                print(f"Skipping {file_path}, Reason:{e}")
                continue
        return np.vstack(processed_data), np.concatenate(labels)


    neu_files   = glob.glob(f"{data_path}/NNdata_neuDIS_MuBack_batch_0_*.root")
    mu_files    = glob.glob(f"{data_path}/NNdata_muDIS_MuBack_batch_0_*.root")
    embg_files  = glob.glob(f"{data_path}/NNdata_signal_MuBack_batch_0_*.root") 

    print(f"Number of datafiles available:\n\tneuDIS+MuBack:\t{len(neu_files)}\n\tmuDIS+MuBack:\t{len(mu_files)}\n\tSignal+MuBack:{len(embg_files)}")

    # Load data for each category
    #embg_data, embg_labels  = load_and_process_data(embg_files, category=0) #signal+MuinducedBG
    #neu_data, neu_labels    = load_and_process_data(neu_files, category=1) #neuDIS +MuinducedBG
    #mu_data, mu_labels      = load_and_process_data(mu_files, category=2) #muDIS+MuinducedBG
    embg_data, embg_labels  = load_and_process_data_wtime(embg_files, category=0) #signal+MuinducedBG
    neu_data, neu_labels    = load_and_process_data_wtime(neu_files, category=1) #neuDIS +MuinducedBG
    mu_data, mu_labels      = load_and_process_data_wtime(mu_files, category=2) #muDIS+MuinducedBG


    # Combine all data
    X = np.vstack([neu_data, mu_data, embg_data])
    Y = np.concatenate([neu_labels, mu_labels, embg_labels]).astype(int)


    return X,Y


# In[5]:


def print_features(inputmatrix,truth,prefix="Features_"):
    plt.figure(figsize=(15,7))
    for i,tag in enumerate(["Signal+MuinducedBG","neuDIS+MuinducedBG","muDIS+MuinducedBG"]):
        nonzero_counts = np.sum(inputmatrix[truth==i][:,:854]>0,axis=1)
        plt.hist(nonzero_counts, bins=125,range=[0,700],histtype='step',linewidth=3,label=tag,color=colors[tag])
        plt.ylabel('count')
    plt.legend()
    plt.yscale("log")    
    plt.xlabel('number of SBT cells firing per event')
    plt.savefig('plots/SBT_multiplicity.png')


    # Define the ranges and bins
    x_range = [-600, 600]
    y_range = [-600, 600]
    bins = (200, 200)

    # Create a figure and axes
    fig, axes = plt.subplots(1, 3, figsize=(20, 8), sharex=True, sharey=True)
    fig.suptitle("Vertex position along x,y", fontsize=16)

    # Initialize variables to compute global vmin and vmax
    vmin, vmax = None, None

    # First pass: Calculate vmin and vmax across all histograms
    for i in range(3):
        x_vertex = inputmatrix[truth == i][:, 1709]
        y_vertex = inputmatrix[truth == i][:, 1710]
        hist, _, _ = np.histogram2d(x_vertex, y_vertex, bins=bins, range=[x_range, y_range])
        if vmin is None or vmax is None:
            vmin, vmax = np.min(hist[hist > 0]), np.max(hist)
        else:
            vmin = min(vmin, np.min(hist[hist > 0]))
            vmax = max(vmax, np.max(hist))

    # Second pass: Plot each histogram
    for i, (ax, tag) in enumerate(zip(axes, ["Signal+MuinducedBG", "neuDIS+MuinducedBG", "muDIS+MuinducedBG"])):
        x_vertex = inputmatrix[truth == i][:, 1709]
        y_vertex = inputmatrix[truth == i][:, 1710]
        h = ax.hist2d(
            x_vertex, y_vertex, 
            bins=bins, range=[x_range, y_range], 
            cmap=mpl.cm.plasma, 
            norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
        )
        ax.set_title(tag, fontsize=14)
        ax.set_xlabel("x (cm)", fontsize=12)
        ax.set_ylabel("y (cm)", fontsize=12)

    # Adjust subplot spacing and colorbar
    fig.subplots_adjust(wspace=0.15, hspace=0.2)  # Adjust spacing between subplots
    cbar = fig.colorbar(h[3], ax=axes, location="right", aspect=40, pad=0.02)
    cbar.set_label("Counts (log scale)", fontsize=14)
    plt.savefig(f'{output_dir}/{prefix}Candidatepos_XY.png')


    plt.figure(figsize=(15,7))
    plt.suptitle("Candidate vertex position along z")
    for i,tag in enumerate(["Signal+MuinducedBG","neuDIS+MuinducedBG","muDIS+MuinducedBG"]):
        plt.hist(inputmatrix[truth == i][:,1711],bins=100, range=[-2800, 2800], histtype='step',linewidth=3,label=tag,color=colors[tag])
        plt.ylabel('count')
        plt.legend()
        #plt.yscale("log")
    #plt.ylim(1,10000)
    plt.xlabel("z (cm)")
    plt.savefig(f'{output_dir}/{prefix}Candidatepos_Z.png')
    plt.close()


    fig, axes = plt.subplots(nrows=3, figsize=(30, 15), sharex=True, sharey=True)
    fig.suptitle("Average Energy Deposited Per Cell", fontsize=16)

    global_data = inputmatrix[:, :854] 
    global_mask = global_data != 0 
    max_values = np.max(global_data * global_mask, axis=0)  # Global maximum per cell index

    for i, (ax, tag) in enumerate(zip(axes, ["Signal+MuinducedBG", "neuDIS+MuinducedBG", "muDIS+MuinducedBG"])):
        data = inputmatrix[truth == i][:, :854]
        mask = data != 0  # Mask cells with zero energy deposits

        mean_values = np.sum(data * mask, axis=0) / np.sum(mask, axis=0)
        std_values = np.sqrt(np.sum((data * mask) ** 2, axis=0) / np.sum(mask, axis=0) - mean_values ** 2)

        lower_error = mean_values - std_values
        lower_error[lower_error < 0] = 0  # Set lower bound to 0

        # Plot mean with error bars
        ax.errorbar(
            range(854), mean_values, 
            yerr=[mean_values - lower_error, std_values], 
            fmt=shapes[tag], markersize=5, mew=2, label=f"{tag} (mean ± std)", color=colors[tag], alpha=1,elinewidth=0.25
        )

        ax.set_title(f"{tag}", fontsize=14)
        ax.set_ylabel("Energy Deposited (GeV)", fontsize=12)
        #ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        ax.set_yscale('log')
        ax.axhline(y=0.045,linestyle='--',color='black',label='45MeV')
        ax.legend(fontsize=12)

    axes[-1].set_xlabel("Cell Index", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the main title
    plt.savefig(f'{output_dir}/{prefix}Avgenergy_SBT.png')
    ##plt.show()

    fig, axes = plt.subplots(nrows=3, figsize=(30, 15), sharex=True, sharey=True)
    plt.suptitle("Energy Deposited Per Cell with Global Max", fontsize=16)

    global_data = inputmatrix[:, :854] 
    global_mask = global_data != 0 
    max_values = np.max(global_data * global_mask, axis=0)  # Global maximum per cell index
    index_tag={"Signal+MuinducedBG":0, "neuDIS+MuinducedBG":1, "muDIS+MuinducedBG":2}

    for i, (ax, tag) in enumerate(zip(axes, ["Signal+MuinducedBG", "neuDIS+MuinducedBG", "muDIS+MuinducedBG"])):

        data = inputmatrix[truth == index_tag[tag]][:, :854]
        
        mask = data != 0  # Mask cells with zero energy deposits

        cell_indices = np.tile(np.arange(854), data.shape[0])  # Repeat indices for scatter plotting
        cell_values = data.flatten()  
        non_zero_indices = cell_values > 0  # Mask for non-zero entries
        
        ax.scatter(
            cell_indices[non_zero_indices],  # Cell indices for non-zero entries
            cell_values[non_zero_indices],  # Corresponding non-zero energy values
            label=tag, marker=shapes[tag], color=colors[tag], s=25,alpha=0.5
        )
        ax.set_title(f"{tag}", fontsize=14)
        ax.set_ylabel("Energy Deposited (GeV)", fontsize=12)
        #ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        ax.set_yscale('log')
        ax.axhline(y=0.045,linestyle='--',color='black',label='45MeV')
        ax.legend(fontsize=12)

    axes[-1].set_xlabel("Cell Index", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the main title
    plt.savefig(f'{output_dir}/{prefix}AllEnergy_SBT.png')

    fig, axes = plt.subplots(nrows=3, figsize=(30, 15), sharex=True, sharey=True)
    plt.suptitle("Timing Recorded per Cell (Valid Hits Only)", fontsize=16)

    index_tag = {
        "Signal+MuinducedBG": 0,
        "neuDIS+MuinducedBG": 1,
        "muDIS+MuinducedBG": 2
    }

    for ax, tag in zip(axes, index_tag.keys()):
        data = inputmatrix[truth == index_tag[tag]][:, 854:1708]  # Timing slice (shape: N x 854)

        mask = data != -9999  # Only valid hits
        cell_values = data.flatten()
        valid_mask = cell_values != -9999

        cell_indices = np.tile(np.arange(854), data.shape[0])

        ax.scatter(
            cell_indices[valid_mask],
            cell_values[valid_mask],
            label=tag, marker=shapes[tag],
            color=colors[tag], s=25, alpha=0.4
        )
        ax.set_title(f"{tag}", fontsize=14)
        ax.set_ylabel("Time (ns)", fontsize=12)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(fontsize=12)
        ax.set_yscale('log')

    axes[-1].set_xlabel("Cell Index", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{output_dir}/{prefix}AllTime_SBT.png")
    


# In[21]:


def plot_scalingeffect(inputmatrix,inputmatrix_scaled,truth):

    plt.figure(figsize=(30,10))

    plt.subplot(321)
    plt.title('prescaling')
    for i,tag in enumerate(["Signal+MuinducedBG","neuDIS+MuinducedBG","muDIS+MuinducedBG"]):
        plt.plot(np.std(inputmatrix[truth == i][:,:854], axis=0),color=colors[tag],marker=shapes[tag],label=tag,alpha=0.5)
        plt.ylabel(r'Std($E_{dep}$)')
        plt.xlabel('cell index')
    plt.legend()
                    
    plt.subplot(322)
    plt.title('postscaling')
    for i,tag in enumerate(["Signal+MuinducedBG","neuDIS+MuinducedBG","muDIS+MuinducedBG"]):
        plt.plot(np.std(inputmatrix[truth == i][:,:854], axis=0),color=colors[tag],marker=shapes[tag],label=tag,alpha=0.5)
        plt.ylabel(r'Std($E_{dep}$)')
        plt.xlabel('cell index')
    plt.legend()

    plt.subplot(323)
    for i,tag in enumerate(["Signal+MuinducedBG","neuDIS+MuinducedBG","muDIS+MuinducedBG"]):
        x_vertex = inputmatrix[truth == i][:, 1709]
        y_vertex = inputmatrix[truth == i][:, 1710]
        plt.scatter(x_vertex,y_vertex,color=colors[tag],marker=shapes[tag],label=tag,alpha=0.05)
        plt.ylabel('y [cm]')
        plt.xlabel('x [cm]')
    plt.legend()
                    
    plt.subplot(324)

    for i,tag in enumerate(["Signal+MuinducedBG","neuDIS+MuinducedBG","muDIS+MuinducedBG"]):
        x_vertex = inputmatrix_scaled[truth == i][:, 1709]
        y_vertex = inputmatrix_scaled[truth == i][:, 1710]
        #plt.hist(inputmatrix[truth == i][:,856],bins=100, histtype='step',linewidth=3,label=tag,color=colors[tag])
        plt.scatter(x_vertex,y_vertex,color=colors[tag],marker=shapes[tag],label=tag,alpha=0.05)
        plt.ylabel('y [cm]')
        plt.xlabel('x [cm]')
    plt.legend()

    plt.subplot(325)
    for i,tag in enumerate(["Signal+MuinducedBG","neuDIS+MuinducedBG","muDIS+MuinducedBG"]):
        plt.hist(inputmatrix[truth == i][:,1711],bins=100, histtype='step',linewidth=3,label=tag,color=colors[tag])
        plt.ylabel('count')
        plt.legend()
        plt.yscale("log")

    #plt.ylim(1,10000)
    plt.xlabel("z (cm)")
    #plt.tight_layout()

    plt.subplot(326)

    for i,tag in enumerate(["Signal+MuinducedBG","neuDIS+MuinducedBG","muDIS+MuinducedBG"]):
        plt.hist(inputmatrix_scaled[truth == i][:,1711],bins=100, histtype='step',linewidth=3,label=tag,color=colors[tag])
        plt.ylabel('count')
        plt.legend()
        plt.yscale("log")
    #plt.ylim(1,10000)
    plt.xlabel("z (cm)")

    plt.tight_layout()
    plt.savefig('plots/Scaling_effects.png')
    #plt.show()
    #plt.close()


# In[7]:


def create_dataset(inputmatrix_scaled,truth,signal_info):

    # Convert data to PyTorch tensors
    X_tensor    = torch.tensor(inputmatrix_scaled, dtype=torch.float32)
    Y_tensor    = torch.tensor(truth, dtype=torch.int64)
    signal_info = torch.tensor(signal_info, dtype=torch.float32)

    #split to temp and test
    X_temp, X_test, Y_temp, Y_test, sig_temp, sig_test = train_test_split( X_tensor, Y_tensor, signal_info, test_size=0.2, random_state=42)
    
    if print_plots:
        print("\n\nTest dataset")
        print_composition_donut(X_test.numpy(),Y_test.numpy(),"DONUT_testdata.png")
    
    #split temp to train and validate
    X_train, X_val, Y_train, Y_val, sig_train, sig_val = train_test_split( X_temp, Y_temp, sig_temp, test_size=0.25, random_state=42)
    #60% for training, 20% for validation, and 20% for testing

    # Create TensorDataset objects for each split:
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train, sig_train)
    val_dataset   = torch.utils.data.TensorDataset(X_val, Y_val, sig_val)
    test_dataset  = torch.utils.data.TensorDataset(X_test, Y_test, sig_test)
  
    return train_dataset,val_dataset,test_dataset


# In[8]:


def draw_neural_network(input_nodes, hidden_layers, output_nodes, max_neurons_per_layer=8):
    """
    Draws a feedforward neural network diagram with indices for neurons and dots for skipped neurons.

    Args:
    - input_nodes (int): Number of neurons in the input layer.
    - hidden_layers (list of int): Number of neurons in each hidden layer.
    - output_nodes (int): Number of neurons in the output layer.
    - max_neurons_per_layer (int): Maximum number of neurons to display per layer. Adds '...' for intermediate indices if exceeded.
    """
    # Create a directed graph
    G = nx.DiGraph()

    # Define layer sizes
    layer_sizes = [input_nodes] + hidden_layers + [output_nodes]
    num_layers = len(layer_sizes)

    # Dictionary for node positions
    pos = {}

    # Define spacing
    x_spacing = 2  # Spacing between layers

    def calculate_Y_positions(display_count):
        """
        Calculates the y-coordinates for the displayed neurons in a layer,
        centering them vertically.
        """
        return [i - (display_count - 1) / 2 for i in range(display_count)]

    def add_layer_nodes(layer_idx, num_neurons):
        """
        Adds neurons for a single layer, displaying up to `max_neurons_per_layer`.
        Returns the list of node names and their indices.
        """
        x = layer_idx * x_spacing
        node_names = []

        # If neurons exceed max displayable neurons, split into top and bottom
        if num_neurons > max_neurons_per_layer:
            num_display = max_neurons_per_layer // 2
            top_indices = range(1, num_display + 1)
            bottom_indices = range(num_neurons - num_display + 1, num_neurons + 1)
            y_positions = calculate_y_positions(num_display * 2 + 1)  # Include space for dots
            dotted = True
        else:
            top_indices = range(1, num_neurons + 1)
            bottom_indices = []
            y_positions = calculate_y_positions(len(top_indices))
            dotted = False

        # Add top indices
        for i, idx in enumerate(top_indices):
            y = y_positions[i]
            node_name = f"L{layer_idx}_N{idx}"
            G.add_node(node_name)
            pos[node_name] = (x, y)
            node_names.append((node_name, idx))

        # Add dots if needed
        if dotted:
            dot_node = f"L{layer_idx}_dots"
            G.add_node(dot_node)
            pos[dot_node] = (x, y_positions[len(top_indices)])
            node_names.append((dot_node, "..."))

        # Add bottom indices
        for i, idx in enumerate(bottom_indices):
            y = y_positions[len(top_indices) + 1 + i]
            node_name = f"L{layer_idx}_N{idx}"
            G.add_node(node_name)
            pos[node_name] = (x, y)
            node_names.append((node_name, idx))

        return node_names

    # Add nodes for all layers
    all_nodes = []
    for layer_idx, num_neurons in enumerate(layer_sizes):
        all_nodes.append(add_layer_nodes(layer_idx, num_neurons))

    # Add edges between layers
    for layer_idx in range(num_layers - 1):
        current_layer = [n[0] for n in all_nodes[layer_idx]]
        next_layer = [n[0] for n in all_nodes[layer_idx + 1]]
        for curr_node in current_layer:
            if "dots" in curr_node:
                continue
            for next_node in next_layer:
                if "dots" in next_node:
                    continue
                G.add_edge(curr_node, next_node)

    # Draw the network
    plt.figure(figsize=(20, 10))
    nx.draw(
        G, pos, with_labels=False, node_size=600, node_color="lightblue", 
        edge_color="black", width=0.8, alpha=0.8
    )

    # Add labels for neurons
    for layer_idx, layer_nodes in enumerate(all_nodes):
        for node_name, neuron_idx in layer_nodes:
            x, y = pos[node_name]
            if neuron_idx == "...":
                plt.text(x, y, "...", fontsize=12, ha="center", va="center", fontweight="bold")
            else:
                plt.text(x, y, str(neuron_idx), fontsize=8, ha="center", va="center")

    # Add layer labels based on displayed neurons
    for layer_idx, layer_nodes in enumerate(all_nodes):
        x = layer_idx * x_spacing
        displayed_y_positions = [pos[node_name][1] for node_name, neuron_idx in layer_nodes if neuron_idx != "..."]
        if displayed_y_positions:
            center_y = (min(displayed_y_positions) + max(displayed_y_positions)) / 2
        else:
            center_y = 0

        if layer_idx == 0:
            plt.text(x, center_y + max_neurons_per_layer/2+0.5, f"Input Layer\n({layer_sizes[layer_idx]} nodes)", fontsize=12, ha="center", fontweight="bold")
        elif layer_idx == num_layers - 1:
            plt.text(x, center_y + max_neurons_per_layer/2+0.5, f"Output Layer\n({layer_sizes[layer_idx]} nodes)", fontsize=12, ha="center", fontweight="bold")
        else:
            plt.text(x, center_y + max_neurons_per_layer/2+0.5, f"Hidden Layer {layer_idx}\n({layer_sizes[layer_idx]} neurons)", fontsize=12, ha="center", fontweight="bold")

    plt.axis("off")
    plt.savefig('plots/nn_network_graph.png')
    #plt.show()

def print_model_parameters_markdown_table_auto(model):
            """
            Automatically builds and prints a Markdown table of trainable parameters,
            with friendly names like "Input → Hidden Layer 1 (858 → 32)", etc.
            """
            # 1) Gather in-definition-order only the Linear & BatchNorm1d layers
            layers = []
            def _collect(mod):
                for child in mod._modules.values():
                    if isinstance(child, (nn.Linear, nn.BatchNorm1d)):
                        layers.append(child)
                    # dive into children of containers (Sequential, etc.)
                    if hasattr(child, '_modules') and child._modules:
                        _collect(child)
            _collect(model)

            # 2) Count how many Linears there are (to detect the “last” one)
            total_linears = sum(isinstance(l, nn.Linear) for l in layers)

            # 3) Build a list of (module, friendly_name) in the same order
            rows = []
            lin_idx = 0
            bn_idx  = 0
            for layer in layers:
                if isinstance(layer, nn.Linear):
                    lin_idx += 1
                    in_f, out_f = layer.in_features, layer.out_features

                    if lin_idx == 1:
                        name = f"Input → Hidden Layer 1 ({in_f} → {out_f})"
                    elif lin_idx < total_linears:
                        name = f"Hidden Layer {lin_idx-1} → Hidden Layer {lin_idx} ({in_f} → {out_f})"
                    else:
                        # final Linear
                        name = f"Hidden Layer {lin_idx-1} → Output Layer ({in_f} → {out_f})"

                    w = layer.weight.numel()
                    b = layer.bias.numel() if layer.bias is not None else 0
                    rows.append((name, w, b))

                elif isinstance(layer, nn.BatchNorm1d):
                    bn_idx += 1
                    n = layer.num_features
                    name = f"BatchNorm {bn_idx} ({n} neurons)"
                    w = layer.weight.numel()
                    b = layer.bias.numel() if layer.bias is not None else 0
                    rows.append((name, w, b))

            # 4) Print Markdown table
            total_w = sum(w for _, w, _ in rows)
            total_b = sum(b for _, _, b in rows)

            print("| **Layer**                                    | **Weights** | **Biases** | **Total Parameters** |")
            print("|----------------------------------------------|------------:|-----------:|--------------------:|")
            for name, w, b in rows:
                print(f"| **{name}** | **{w:,}** | **{b:,}** | **{w+b:,}** |")
            print("| **Total Parameters**                         | **" +
                  f"{total_w:,}** | **{total_b:,}** | **{total_w+total_b:,}** |")


# In[9]:


def class_wise_performancemetrics(true_labels,preds):
    # Generate the classification report
    report = classification_report(true_labels, preds, target_names=['Signal', 'NeuDIS', 'MuDIS'], output_dict=True)
    df_report = pd.DataFrame(report).transpose()

    # Extract relevant metrics for each class (excluding avg values)
    df_class_metrics = df_report.iloc[:-3][['precision', 'recall', 'f1-score']]

    # Define distinct colors
    custom_colors = ['pink', 'purple', 'red']

    # Sort classes by F1-score for better readability
    df_class_metrics = df_class_metrics.sort_values(by="f1-score", ascending=False)

    # Plot grouped bar chart
    ax = df_class_metrics.plot(kind='bar', figsize=(12, 6), color=custom_colors, edgecolor='black', alpha=0.85)

    # Plot Accuracy & Macro Avg Lines
    accuracy = report['accuracy']
    macro_f1 = df_report.loc['macro avg']['f1-score']

    plt.axhline(y=accuracy, color='pink', linestyle='--', label=f"Overall Accuracy: {accuracy:.2f}")
    plt.axhline(y=macro_f1, color='black', linestyle='-.', label=f"Macro Avg F1-Score: {macro_f1:.2f}")

    # Add labels and formatting
    plt.grid(True, alpha=0.3)
    plt.title("Class-Wise Performance Metrics", fontsize=14)
    plt.ylabel("Score", fontsize=12)
    plt.ylim(0, 1.4)
    plt.xlabel("Class", fontsize=12)
    plt.xticks(rotation=0, fontsize=10)
    plt.legend(loc="lower right", fontsize=10)

    # Show plot
    print("High accuracy (~1.0) means the model performs well on average.\nHigh precision (~1.0) means few false positives (FP).\nHigh recall (~1.0) means few false negatives (FN).\nHigh F1-score (~1.0) means good overall performance.F1=2×(PrecisionxRecall)/(Precision+Recall)\nOverall Accuracy is the percentage of correctly classified samples out of the total.")
    plt.savefig("plots/class_wise_performancemetrics.png")
    

# In[10]:


def NN_training(model,epochs=30,prefix='Timing_'):
    
    logs = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'gradient_norms': []}
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0
        gradient_norms = []

        for inputs, labels, _ in train_loader: 
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Compute gradient norms
            grad_norms = []
            for name,param in model.named_parameters():
                if param.grad is not None  and ("bn" not in name and "norm" not in name):
                    grad_norms.append(param.grad.norm().item())
            gradient_norms.append(grad_norms)
            
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total
        logs['train_loss'].append(train_loss)
        logs['train_acc'].append(train_acc)

        # Save gradient norms per epoch
        logs['gradient_norms'].append(torch.tensor(gradient_norms).mean(dim=0).cpu().numpy())

        # Validate the model
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total
        logs['val_loss'].append(val_loss)
        logs['val_acc'].append(val_acc)

        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    gradient_norms = np.array(logs['gradient_norms'])

    # Plot Gradient Monitoring
    plt.figure(figsize=(15, 7))
    for i in range(gradient_norms.shape[1]):
        plt.plot(gradient_norms[:, i], label=f"Layer {i+1}")
    #plt.yscale("log") 
    plt.xlabel("Epoch")
    plt.ylabel("Gradient Norm")
    plt.yscale("log")
    plt.title("Gradient Monitoring Per Layer")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/GradientMonitoring.png")
    plt.close()


    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(logs['train_loss'], label="Train Loss")
    plt.plot(logs['val_loss'], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")

    plt.subplot(1, 2, 2)
    plt.plot(logs['train_acc'], label="Train Accuracy")
    plt.plot(logs['val_acc'], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.tight_layout()
    plt.savefig("plots/Accuracy_curve.png")
    plt.close()

    model_path = f"{prefix}SBTveto_model_newgeo_{threshold}MeV_{epochs}epochs.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

    return model_path


# In[28]:


def plot_precision_recall_curve(true_labels,all_probs):
    # ✅ **Step 4: Compute and Plot Precision-Recall Curve for Background Detection**
    # Define background as positive (1), signal as negative (0)
    binary_labels = np.where(np.isin(true_labels, [1, 2]), 1, 0)  # Background = 1, Signal = 0
    background_probs = all_probs[:, 1] + all_probs[:, 2]  # Sum of probabilities for background classes

    # Compute Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(binary_labels, background_probs, pos_label=1)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(8, 8))  # Adjust figure size as needed

    # Plot Precision-Recall Curve
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot(ax=ax)  # Pass the existing axis

    # Customize title and labels
    ax.set_title("Precision-Recall Curve (Background Detection)")
    ax.set_xlabel("Recall (Correctly Identified Background)")
    ax.set_ylabel("Precision (How Many Predicted as Background Were Correct)")
    ax.grid(True)
    plt.tight_layout()
    # Show plot
    #plt.show()

    # Compute AUC for Precision-Recall Curve (Background Detection)

    pr_auc = average_precision_score(binary_labels, background_probs)

    print(f"AUC for Precision-Recall Curve (Background Detection): {pr_auc}")


    # In[ ]:


    # ✅ **Step 4: Compute and Plot Precision-Recall Curve for Background Detection**
    # Define background as positive (1), signal as negative (0)
    binary_labels = np.where(np.isin(true_labels, [1, 2]), 1, 0)  # Background = 1, Signal = 0
    signal_probs = all_probs[:, 0] # Sum of probabilities for background classes

    # Compute Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(binary_labels, signal_probs, pos_label=0)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(8, 8))  # Adjust figure size as needed

    # Plot Precision-Recall Curve
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot(ax=ax)  # Pass the existing axis

    # Customize title and labels
    ax.set_title("Precision-Recall Curve (Signal Detection)")
    ax.set_xlabel("Recall (Correctly Identified Signal)")
    ax.set_ylabel("Precision (How Many Predicted as Signal Were Correct)")
    ax.grid(True)

    # Show plot
    #plt.show()

    # Compute AUC for Precision-Recall Curve (Signal Detection)

    pr_auc = average_precision_score(binary_labels, signal_probs)

    print(f"AUC for Precision-Recall Curve (Signal Detection): {pr_auc}")


# In[12]:


def implement_selections(neuDIS_as,muDIS_as):

    import ROOT
    path='geofile_full.conical.MuonBack-TGeant4.root'#'/eos/experiment/ship/simulation/bkg/MuonBack_2024helium/8070735/job_100/geofile_full.conical.MuonBack-TGeant4.root'

    fgeo = ROOT.TFile.Open(path, "read")
    sGeo   = fgeo.FAIRGeom

    def is_in_fiducial(X,Y,Z):
            """Check if the candidate decay vertex is within the Fiducial Volume."""

            if Z > 2588.0:#self.ship_geo.TrackStation1.z:
                return False
            if Z < -2478.0:#self.veto_geo.z0:
                return False

            # if self.dist2InnerWall(candidate)<=5*u.cm: return False

            vertex_node = ROOT.gGeoManager.FindNode(X,Y,Z)
            vertex_elem = vertex_node.GetVolume().GetName()
            if not vertex_elem.startswith("DecayVacuum_"):
                return False
            return True
    
    def dist_to_innerwall(X,Y,Z):
            """Calculate the minimum distance(in XY plane) of the candidate decay vertex to the inner wall of the decay vessel. If outside the decay volume, or if distance > 100cm,Return 0."""
            
            position = (X,Y,Z)

            nsteps = 8
            dalpha = 2 * ROOT.TMath.Pi() / nsteps
            min_distance = float("inf")

            node = ROOT.gGeoManager.FindNode(*position)
            if not node:
                return 0  # is outside the decay volume

            # Loop over directions in the XY plane
            for n in range(nsteps):
                alpha = n * dalpha
                direction = (
                    ROOT.TMath.Sin(alpha),
                    ROOT.TMath.Cos(alpha),
                    0.0,
                )  # Direction vector in XY plane
                ROOT.gGeoManager.InitTrack(*position, *direction)
                if not ROOT.gGeoManager.FindNextBoundary():
                    continue
                # Get the distance to the boundary and update the minimum distance
                distance = ROOT.gGeoManager.GetStep()
                min_distance = min(min_distance, distance)

            return min_distance if min_distance < 10000 else 0 #100 * u.m 

    def preselection_cut(signals,tag, IP_cut=250):
        
        """
        Umbrella method to apply the pre-selection cuts on the candidate.
        missclassified_signals=misclassified_groups[0][:, i]
        show_table=True tabulates the pre-selection parameters.
        """
        
        _={'X':0,'Y':1,'Z':2,'nParticles':0+3,'Invar. mass':1+3, 'DOCA':2+3, 'Impact_par':3+3,'d1_ch2':4+3,'d2_ch2':5+3,'d1_ndf':6+3,'d2_ndf':7+3,'d1_mom':8+3,'d2_mom':9+3,'UBT_hits':10+3,"eventweight":11+3}
        
        afterbasicselection=[]
        
        #z_afterbasic=[]
        h={}
        for candidate in signals:
            
            flag = True
            
            if candidate[_['nParticles']]!=1: flag = False
            if candidate[_['Impact_par']] >= IP_cut:flag = False
            if candidate[_['DOCA']] >= 1: flag = False
            if (candidate[_['d1_ndf']] <= 25) or (candidate[_['d2_ndf']] <= 25): flag = False
            if (candidate[_['d1_ch2']] >= 5) or (candidate[_['d2_ch2']] >=5):   flag = False
            if (candidate[_['d1_mom']] <=1) or (candidate[_['d2_mom']] <=1):   flag = False            
            if (candidate[_['UBT_hits']]!=0):   flag = False            
            if not ( is_in_fiducial(candidate[_['X']],candidate[_['Y']],candidate[_['Z']]) ):   flag = False
            if dist_to_innerwall(candidate[_['X']],candidate[_['Y']],candidate[_['Z']]) <= 5:   flag = False
            #if dist_to_vesselentrance(candidate[_['X']],candidate[_['Y']],candidate[_['Z']]) <= 100 * u.cm:flag = False
            
            if flag: 
                for signal_info in _:
                    if signal_info not in h:
                        h[signal_info]=[]
                    h[signal_info].append(candidate[_[signal_info]])
                
                #    print(f"{cut}:{candidate[_[cut]]}")         
                #z_afterbasic.append(candidate[_['Z']])
            
            afterbasicselection.append(flag)
                                                    
        return h,afterbasicselection #returns [True,True] if passes preselection 

    h_neuDIS,selection_passed=preselection_cut(neuDIS_as[0],'neuDIS tagged as signal')
    neudis_remaining_afterselection = sum(selection_passed)
    h_muDIS,selection_passed=preselection_cut(muDIS_as[0],'muDIS tagged as signal')
    mudis_remaining_afterselection = sum(selection_passed)
    print(f"before:\t{len(neuDIS_as[0])}\t neudis_remaining_afterselection: {neudis_remaining_afterselection} \n\t{len(muDIS_as[0])} \t mudis_remaining_afterselection: {mudis_remaining_afterselection}")

    return h_neuDIS,h_muDIS


# In[16]:


#def main(print_plots=True,plotNN=False):
print_plots=True,
plotNN=False


# In[19]:


# 1. Load Data
data_path="/eos/experiment/ship/user/anupamar/NN_data/root_files/wMuonBack"

X,Y=prepare_data(data_path)

if print_plots:
    print("Full dataset")
    print_composition_donut(X,Y,filename="Donut_fulldata.png")

threshold = 45 #inMeV
X_withthreshold = X.copy()

# 1. Energy mask: identify hits above threshold
energy_mask = X[:, :854] > threshold * 0.001  # shape: (n_events, 854)
X_withthreshold[:, :854] = np.where(energy_mask, X[:, :854], 0)#Set SBT hit Edep to 0 if less than threshold

# 3. Apply threshold to timing: set to -9999 if corresponding energy is below threshold
X_withthreshold[:, 854:1708] = np.where(energy_mask, X[:, 854:1708], -9999)

#if print_plots:
#    print_composition_donut(X_withthreshold,Y,filename=f"Donut_setEdep{threshold}.png")

mask = np.any(X_withthreshold[:, :854] > 0, axis=1)
inputmatrix_full = X_withthreshold[mask]
truth = Y[mask]
print(f"nEvents with Edep > {threshold} MeV threshold = {len(inputmatrix_full)}")

inputmatrix = inputmatrix_full[:, :1712]         #without signal info,len(UBT_hits),event_weight in 15 years
signal_info = inputmatrix_full[:, 1709:]         #vertex x, y, z,signal info,len(UBT_hits),event_weight in 15 years

if print_plots:

    plt.figure(figsize=(15,7))
    for i,tag in enumerate(["Signal+MuinducedBG","neuDIS+MuinducedBG","muDIS+MuinducedBG"]):
        subset = inputmatrix_full[:, 854:][truth == i]
        w=subset[:,-1]
        plt.hist(w, bins=100,histtype='step',linewidth=3,label=tag,color=colors[tag])
        plt.ylabel('count')
    plt.legend()
    plt.yscale("log")    
    plt.xscale("log")    
    plt.xlabel('$eventweight_{sample}$(scaled to 15 years)')
    plt.savefig(f'{output_dir}/eventweights.png')
    #plt.show()
    print(f"\n\nDataset (above {threshold}MeV)")
    print_composition_donut(inputmatrix,truth,filename="Donut_NNdata.png")
    print_features(inputmatrix,truth,prefix="Features_")


# In[22]:

# Split features
energy_features = inputmatrix[:, :854]
timing_features = inputmatrix[:, 854:1709]
vertex_feature  = inputmatrix[:, 1709:]  # candidate X,Y,Z

# Replace invalid timing values (-9999) with 0 for scaling
timing_features_masked = np.where(timing_features == -9999, 0, timing_features)

# Fit separate scalers
scaler_energy = RobustScaler()
scaler_time   = RobustScaler()

scaled_energy = scaler_energy.fit_transform(energy_features)
scaled_time   = scaler_time.fit_transform(timing_features_masked)

# Concatenate all components
inputmatrix_scaled = np.hstack([scaled_energy, scaled_time, vertex_feature])

# Save scalers
joblib.dump(scaler_energy, "scaler_energy.pkl")
joblib.dump(scaler_time, "scaler_time.pkl")

# Save scaler parameters for reference
scaler_params = {
    'energy_center_': scaler_energy.center_.tolist(),
    'energy_scale_':  scaler_energy.scale_.tolist(),
    'time_center_':   scaler_time.center_.tolist(),
    'time_scale_':    scaler_time.scale_.tolist(),
}
with open('robust_scaler_params.json', 'w') as f:
    json.dump(scaler_params, f)


if print_plots:
    plot_scalingeffect(inputmatrix,inputmatrix_scaled,truth)

#create PyTorch tensors dataset objects
train_dataset,val_dataset,test_dataset=create_dataset(inputmatrix_scaled,truth,signal_info)

# And then create DataLoaders:
train_loader = DataLoader(train_dataset,    batch_size=32, shuffle=True, drop_last=True)
val_loader   = DataLoader(val_dataset,      batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset,     batch_size=32, shuffle=False)

print("\nNumber of training batches:",    len(train_loader))
print("Number of validation batches:",  len(val_loader))
print("Number of test batches:",        len(test_loader))


# In[23]:


# 6. Define Neural Network and Training Parameters
input_dim=1712
output_dim=3
#hidden_sizes=[32, 32, 32, 16, 8]
hidden_sizes = [256, 128, 64, 32, 16]
model = NN(input_dim=input_dim, output_dim=output_dim, hidden_sizes=hidden_sizes, dropout=0.3)

#class_weights   = torch.tensor([2.0, 1.0, 1.0])# Emphasize Signal caution
criterion       = nn.CrossEntropyLoss() #reduction=False (?) loss=torch.mean(weights*loss)
optimizer       = torch.optim.Adam(model.parameters(), lr=0.001)#0.0001 #lr decay
#device          = 'cpu'
device          = 'cuda' if torch.cuda.is_available() else 'cpu'

model.to(device)

#Visualising the network
if print_plots and plotNN:
    draw_neural_network(input_dim, hidden_sizes, output_dim, max_neurons_per_layer=8)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name} | Parameters: {param.numel()}")

    #print_model_parameters_markdown_table_auto(model)

model_path=NN_training(model,epochs=10,prefix='Timing_')# Define hyperparameters

print(f"Training Complete.")
# In[35]:

print(f"Testing starts now.")

test_model_path=model_path
#test_model_path='SBTveto_model_newgeo_45MeV_30epochs.pth'

# In[36]:

# **Step 1: Load and Evaluate the Model**

model.load_state_dict(torch.load(test_model_path, map_location=device))
model.to(device)
model.eval()

print(f"Model successfully loaded from {test_model_path}.")

# Storage for predictions and labels
true_labels, preds, all_probs = [], [], []

neuDIS_as={0: [], 1: [], 2: []}
signal_as={0: [], 1: [], 2: []}
muDIS_as={0: [], 1: [], 2: []}

with torch.no_grad():
    for inputs, labels, signal in test_loader:

        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        
        #threshold = 0.80  # You can tune this based on your false-positive tolerance

        probs = softmax(outputs, dim=1)# Convert logits to probabilities

        #signal_probs = probs[:, 0]  # Class 0 = Signal
        #max_probs, max_indices = torch.max(probs, dim=1)

        # If signal prob is too low, force to predict best background class
        #predicted = max_indices.clone()
        #predicted[signal_probs < threshold] = torch.argmax(probs[signal_probs < threshold, 1:], dim=1) + 1
        
        _, predicted = torch.max(probs, 1)  # Get class with highest probability

        preds.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())  # Store probabilities

        for index, truth_label in enumerate(labels):

            if truth_label.item()==0:
                signal_as[predicted[index].item()].append(signal[index].cpu().numpy())
            if truth_label.item()==1:
                neuDIS_as[predicted[index].item()].append(signal[index].cpu().numpy())
            if truth_label.item()==2:
                muDIS_as[predicted[index].item()].append(signal[index].cpu().numpy())

# Convert stored data to numpy arrays
true_labels = np.array(true_labels)
preds = np.array(preds)
all_probs = np.array(all_probs)  # Shape: (num_samples, num_classes)

# **Step 2: Compute and Plot Confusion Matrix**
cm = confusion_matrix(true_labels, preds)
class_labels = ['Signal', 'nu-DIS', 'mu-DIS']

plt.figure(figsize=(6, 5))
ax = sns.heatmap(cm, cmap="Blues", fmt="d", xticklabels=class_labels, yticklabels=class_labels) #annot=True isnt working correctly 

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        value = cm[i, j]
        text_color = 'white' if value > cm.max() * 0.5 else 'black'
        ax.text(j + 0.5, i + 0.5, str(value),
                ha='center', va='center',
                color=text_color, fontsize=12, fontweight='bold')

plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.tight_layout()  # Avoid overlap/cropping
plt.savefig(f"CM.png")
plt.close()


# Compute accuracy
accuracy = 100 * (1 - np.sum(true_labels != preds) / np.size(true_labels))
print(f"\n\t {accuracy:.2f}% percent classified correctly\n")

# Compute probability of falsely vetoing a signal candidate as background
false_veto_probability = (np.sum(np.logical_and(preds != 0, true_labels == 0)) / np.sum(true_labels == 0)) * 100
print("Probability of falsely vetoing a signal candidate as background:\t",
      np.sum(np.logical_and(preds != 0, true_labels == 0)), "/", np.sum(true_labels == 0),
      "(", round(false_veto_probability, 3), "%)")

# Compute probability of wrongly tagging a nu-DIS candidate as signal
probability_nuDIS = (np.sum(np.logical_and(preds == 0, true_labels == 1)) / np.sum(true_labels == 1)) * 100
print("Probability of wrongly tagging a nu-DIS candidate as signal: \t\t",
      np.sum(np.logical_and(preds == 0, true_labels == 1)), "/", np.sum(true_labels == 1),
      "(", round(probability_nuDIS, 3), "%)")

# Compute probability of wrongly tagging a mu-DIS candidate as signal
probability_muDIS = (np.sum(np.logical_and(preds == 0, true_labels == 2)) / np.sum(true_labels == 2)) * 100
print("Probability of wrongly tagging a mu-DIS candidate as signal: \t\t",
      np.sum(np.logical_and(preds == 0, true_labels == 2)), "/", np.sum(true_labels == 2),
      "(", round(probability_muDIS, 3), "%)")

plot_precision_recall_curve(true_labels,all_probs)

class_wise_performancemetrics(true_labels,preds)


# In[37]:


h_neuDIS,h_muDIS=implement_selections(neuDIS_as,muDIS_as)

h=h_neuDIS

ncoloumns=int(len(h)/3)
fig, axes = plt.subplots(3, ncoloumns, figsize=(22, 15), sharey=False)  # Share y-axis across subplots
fig.suptitle("Features of candidate which pass the selection", fontsize=16)  # Main title

for key, ax in zip(h.keys(), axes.flatten()):  
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=1)
    #print(key,":",h[key])
    ax.hist(h[key], bins=100,range=[min(h[key]),max(h[key])], label=key, color='g')#, alpha=0.5
    ax.legend()
#plt.show()

# Define the ranges and bins
x_range = [-600, 600]
y_range = [-600, 600]
bins = (200, 200)

# Create a figure and axes
fig, axes = plt.subplots(1, 3, figsize=(30, 8), sharex=True, sharey=True)
fig.suptitle("Vertex position along x,y", fontsize=16)

# Initialize variables to compute global vmin and vmax
vmin, vmax = None, None

# First pass: Calculate vmin and vmax across all histograms
for i in range(3):

    x_vertex = inputmatrix[truth == i][:, 854]
    y_vertex = inputmatrix[truth == i][:, 855]
    hist, _, _ = np.histogram2d(x_vertex, y_vertex, bins=bins, range=[x_range, y_range])
    if vmin is None or vmax is None:
        vmin, vmax = np.min(hist[hist > 0]), np.max(hist)
    else:
        vmin = min(vmin, np.min(hist[hist > 0]))
        vmax = max(vmax, np.max(hist))

# Second pass: Plot each histogram
for i, (ax, tag) in enumerate(zip(axes, ["Signal+MuinducedBG", "neuDIS+MuinducedBG", "muDIS+MuinducedBG"])):
    if i==0: continue
    x_vertex = inputmatrix[truth == i][:, 854]
    y_vertex = inputmatrix[truth == i][:, 855]
    h = ax.hist2d(
        x_vertex, y_vertex, 
        bins=bins, range=[x_range, y_range], 
        cmap=mpl.cm.plasma, 
        norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    )
    if tag=="neuDIS+MuinducedBG":
        ax.scatter(h_neuDIS['X'],h_neuDIS['Y'],color='cyan',edgecolors='black',label="survived NN veto+basic selection")
        ax.legend()
    if tag=="muDIS+MuinducedBG":
        ax.scatter(h_muDIS['X'],h_muDIS['Y'],color='cyan',edgecolors='black',label="survived NN veto+basic selection")
        ax.legend()

    ax.set_title(tag, fontsize=14)
    ax.set_xlabel("x (cm)", fontsize=12)
    ax.set_ylabel("y (cm)", fontsize=12)

# Adjust subplot spacing and colorbar
fig.subplots_adjust(wspace=0.15, hspace=0.2)  # Adjust spacing between subplots
cbar = fig.colorbar(h[3], ax=axes, location="right", aspect=40, pad=0.02)
cbar.set_label("Counts (log scale)", fontsize=14)
#plt.show()

fig, axes = plt.subplots(nrows=3, figsize=(30, 15), sharex=True, sharey=True)
plt.suptitle("Candidate position along Z", fontsize=16)

for i, (ax, tag) in enumerate(zip(axes, ["Signal+MuinducedBG","neuDIS+MuinducedBG", "muDIS+MuinducedBG"])):
    ax.hist(inputmatrix[truth == i][:,856],bins=100, range=[-2800, 2800], histtype='step',linewidth=3,label=tag,color=colors[tag])
    ax.set_title(f"{tag}", fontsize=14)
    ax.set_ylabel("count", fontsize=12)
    ax.set_xlabel("Position along z(cm)", fontsize=12)
    if tag=="neuDIS+MuinducedBG":
        ax.hist(h_neuDIS['Z'],color='cyan',label="survived NN veto+basic selection",bins=100, range=[-2800, 2800], histtype='stepfilled',linewidth=3)
        ax.legend()
    if tag=="muDIS+MuinducedBG":
        ax.hist(h_muDIS['Z'],color='cyan',label="survived NN veto+basic selection",bins=100, range=[-2800, 2800], histtype='stepfilled',linewidth=3)
        ax.legend()

    #ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_yscale('log')
    ax.legend(fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the main title
