import torch
import joblib
import glob
import uproot
from sbtveto.model.gnn_model import EncodeProcessDecode
from sbtveto.util.inference import gnn_output
import numpy as np



XYZ =np.load("data/SBT_XYZ.npy")
neu_files = glob.glob("../SBT/ml_dataset/*neuDIS*.root")
file = uproot.open(neu_files[0])
x = np.array(file['tree;1']['inputmatrix'].array())[:,:-2]

print("Input is 104 events of nuDIS")
print("Input shape ", x.shape)
print("Features are E for 2000 SBT cells, signal vertex, x, y, z and UBT hits.")
print("The model will also use SBT x, y, z of cells")






# Load a 4 block GNN model
model = EncodeProcessDecode(mlp_output_size=8, global_op=3,num_blocks=4)
#model.load_state_dict(torch.load('SBT_vacuum_multiclass_4block_GNN.pt', weights_only=True))
#model.eval()

outputs, decisions, classification = gnn_output(model, x, XYZ)

print("The SBT decisions are ", outputs)