# SBT Advanced veto
Framework for ML architectures for the SBT veto trained with pytorch.

## Models currently include:
* neural network
* message passing GNN

## Usage

### Ship Veto

The script nnveto_implementation.py provides an example of running the neural network veto within SHIP. The necessary python packages in the Installation section below are first required using ```pip install --user```

The similar example for the gnn will be provided soon.
### Inference

Run example script apply_nn.py and apply_gnn.py to apply trained neural network and GNN to nuDIS events.
### Training 

Currently performed in notebooks GNN_global.ipynb and NN.ipynb however code wiil be factorized in the near future.

## Installation:
cpu: 
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn
pip install dm-tree
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
```
note for gpu:
gpu: 
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
```

Will add a requirements file later.
