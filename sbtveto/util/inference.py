import numpy as np
import torch


def nn_output(model, data, sbt_xyz, scalar, device="cuda"):
    model.to(device)
    X=np.hstack([ data  , np.repeat(sbt_xyz[:1,:],data.shape[0],0),
              np.repeat(sbt_xyz[1:2,:],data.shape[0],0),
              np.repeat(sbt_xyz[2:,:],data.shape[0],0)])
    X = scalar.transform(X)
    X = torch.tensor(X, dtype =torch.float32 ).to(device)
    output = model(X)
    sbt_decision = (torch.max(output, dim = 1).indices == 0)
    return output, sbt_decision


def gnn_output(model, data, sbt_xyz,  device="cuda"):
    return "Working on implementation"