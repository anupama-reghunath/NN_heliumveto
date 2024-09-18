from sbtveto.model.nn_model import NN
from sbtveto.util.inference import nn_output
import numpy as np
import torch
import joblib
import glob
import uproot


XYZ =np.load("../SBT_XYZ.npy")
neu_files = glob.glob("preprocess/*neuDIS*.root")
file = uproot.open(neu_files[0])
#inputmatrix = np.array(file['tree;1']['inputmatrix'].array())[:,:-1]
data = file['tree;1']['inputmatrix'].array(library='np')
inputmatrix = data[:, :-1]
event_weights = data[:, -1]
print(f"Input is {len(inputmatrix)} events of nuDIS")
print("Input shape ", inputmatrix.shape)
print("Features are E for 2000 SBT cells, signal vertex, x, y, z and UBT hits.")
print("The model will also use SBT x, y, z of cells")
scaler_loaded = joblib.load('data/robust_scaler.pkl')
model = NN(8004,3,[32,32,32,16,8], dropout=0)

# Load the state dictionary into the model
model.load_model('data/SBTveto_vacuum_multiclass_NN.pth')

# Now the model is ready to use
print(model)
model.eval()

outputs, decisions,classification = nn_output(model, inputmatrix, XYZ, scaler_loaded)#returns True if to be removed

print("The SBT decisions are ", decisions)

# Calculate sum of event weights where decision is False
#print(event_weights)
sum_weights_false = np.sum(event_weights[~decisions])
print("Number of signal events weighted over 5 years: ", sum_weights_false)
