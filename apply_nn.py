from sbtveto.model.nn_model import NN
from sbtveto.util.inference import nn_output
import numpy as np
import torch
import joblib
import glob
import uproot


XYZ =np.load("../SBT/SBT_XYZ.npy")
neu_files = glob.glob("../SBT/ml_dataset/*neuDIS*.root")
file = uproot.open(neu_files[0])
x = np.array(file['tree;1']['inputmatrix'].array())[:,:-1]
print("Input is 104 events of nuDIS")
print("Input shape ", x.shape)
print("Features are E for 2000 SBT cells, signal vertex, x, y, z and UBT hits.")
print("The model will also use SBT x, y, z of cells")
scaler_loaded = joblib.load('data/robust_scaler.pkl')
model = NN(8004,3,[32,32,32,16,8], dropout=0)
model.load_state_dict(torch.load('data/SBTveto_vacuum_multiclass_NN.pth', weights_only=True))
model.eval()

outputs, decisions = nn_output(model, x, XYZ, scaler_loaded, device = "cpu")

print("The SBT decisions are ", decisions)

