# SBT Advanced veto
Framework for ML architectures for the SBT veto trained with pytorch.

## Models currently include:
* neural network
* message passing GNN

## Usage

### Inference

Run example script apply_nn.py to apply trained neural network to nuDIS events.

### Training 

To be implemented 

## Installation:
cpu: 
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
```
note for gpu:
gpu: 
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
```

Will add a requirements file later.
