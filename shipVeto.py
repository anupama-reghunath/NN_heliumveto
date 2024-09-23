# utility to simulate response of the veto systems
import ROOT
import shipunit as u
from array import array
import numpy as np
import joblib
import torch
from sbtveto.model.nn_model import NN
from sbtveto.util.inference import nn_output
from sbtveto.model.gnn_model import EncodeProcessDecode
from sbtveto.util.inference import gnn_output
class Task:
 "initialize and give response of the veto systems"
 def __init__(self,t):
  self.SBTefficiency = 0.99  # Surrounding Background tagger: 99% efficiency picked up from TP
  self.SVTefficiency = 0.995 # Straw Veto tagger: guestimate, including dead channels
  self.UBTefficiency = 0.9  # Upstream background tagger
  self.random = ROOT.TRandom()
  ROOT.gRandom.SetSeed(13)
  self.detList  = self.detMap()
  self.sTree = t

 def detMap(self):
  fGeo = ROOT.gGeoManager
  detList = {}
  for v in fGeo.GetListOfVolumes():
   nm = v.GetName()
   i  = fGeo.FindVolumeFast(nm).GetNumber()
   detList[i] = nm
  return detList

 def SBTcell_map(self): #provides a cell map with index in [0,1999] for each cell.
   fGeo = ROOT.gGeoManager
   detList = {}
   LiSC = fGeo.GetTopVolume().GetNode('DecayVolume_1').GetVolume().GetNode('T2_1').GetVolume().GetNode('VetoLiSc_0')
   index = -1
   for LiSc_cell in LiSC.GetVolume().GetNodes():
      index += 1
      name = LiSc_cell.GetName()
      detList[index] = name[-6:]
   return detList

 def SBT_plastic_decision(self,mcParticle=None):
    SBT_decision(self,mcParticle,detector='plastic')
 def SBT_liquid_decision(self,mcParticle=None):
    SBT_decision(self,mcParticle,detector='liquid')

 def SBT_decision(self,mcParticle=None,detector='liquid',threshold=45,advSBT=None,candidate=None):
  # if mcParticle >0, only count hits with this particle
  # if mcParticle <0, do not count hits with this particle
  ################################
  
  #hitSegments = 0
  hitSegments = []
  index = -1
  fdetector = detector=='liquid'

  global fireddetID_list,digihit_index

  
  fireddetID_list={}
  digihit_index={}

  for aDigi in self.sTree.Digi_SBTHits:
     
     index+=1 
     detID    = aDigi.GetDetectorID()
     fireddetID_list[str(aDigi.GetDetectorID())]=aDigi
     digihit_index[str(aDigi.GetDetectorID())]=index

     if fdetector and detID > 999999:continue
     if not fdetector and not detID > 999999:continue 
     if mcParticle:
        found = False
        for mcP in self.sTree.digiSBT2MC[index]: 
         if mcParticle>0 and mcParticle != mcP : found=True
         if mcParticle<0 and abs(mcParticle) == mcP : found=True
        if found: continue
     position = aDigi.GetXYZ()
     ELoss    = aDigi.GetEloss()
     if advSBT:
         if candidate==None:candidate=self.sTree.Particles[0]
         self.t_vtx =self.define_t_vtx(candidate)  
         if ELoss>=threshold*0.001 and self.advSBT_Veto_criteria_check(aDigi,threshold_val=threshold): hitSegments.append(index)#hitSegments+= 1   #does the SBT cell pass the 3 criteria:     

     else:
         if ELoss>=threshold*0.001: hitSegments.append(index)#hitSegments += 1 
         #if aDigi.isValid(): hitSegments += 1 #threshold of 45 MeV per segment
  
  w = (1-self.SBTefficiency)**len(hitSegments)  
  veto = self.random.Rndm() > w

  #print 'SBT :',hitSegments
  return veto, w, hitSegments#hitSegments contain the index of the Digihit that causes the veto
 
 def Veto_decision_NN(self,mcParticle=None,detector='liquid',candidate=None):
   self.inputmatrix = []

   # Load the necessary SBT-related data
   #XYZ = np.load("../SBT_XYZ.npy")  # Load SBT XYZ coordinates
   scaler_loaded = joblib.load('data/robust_scaler.pkl')

   # Define the model architecture and load pretrained weights
   model = NN(2003, 3, [32, 32, 32, 16, 8], dropout=0)
   model.load_model('data/SBTveto_vacuum_multiclass_NN_SBT_E_signal_xyz.pth')
   model.eval()  # Set the model to evaluation mode


   detList = self.SBTcell_map()
   energy_array = np.zeros(2000)
   #time_array = np.full(2000, -9999) #default value is -9999
     
   for aDigi in self.sTree.Digi_SBTHits:
      detID = str(aDigi.GetDetectorID())
      ID_index = [lastname for lastname, firstname in detList.items() if firstname == detID][0]
      energy_array[ID_index] = aDigi.GetEloss()
      #time_array[ID_index] = aDigi.GetTDC()

   nHits=len(self.sTree.UpstreamTaggerPoint) 

   if candidate==None:candidate=self.sTree.Particles[0]

   candidatePos = ROOT.TLorentzVector()
   candidate.ProductionVertex(candidatePos)
   vertexposition = np.array([candidatePos.X(), candidatePos.Y(), candidatePos.Z()])

   self.inputmatrix.append(np.concatenate(
                                       (energy_array,
                                        #time_array,
                                        vertexposition,
                                        #candidate_details,
                                        #np.array(nHits),
                                        #np.array(weight_i)
                                        )
                                        , axis=None
                                        ) )# inputmatrix has shape (1,2003)
   
   # After all data is collected, convert the list to a NumPy array
   self.inputmatrix = np.array(self.inputmatrix, dtype=np.float32)

   outputs, decisions, classification = nn_output(model, self.inputmatrix, scaler_loaded)#returns True if to be vetoed
   
   return decisions,classification #class 0 =Signal, class 1 = neuDIS, class 2 =muonDIS

 def Veto_decision_GNN(self, mcParticle=None, detector='liquid', candidate=None):
     self.inputmatrix = []

     # Load the necessary SBT-related data
     XYZ =np.load("data/SBT_XYZ.npy")

     # Define the model architecture and load pretrained weights
     model = EncodeProcessDecode(mlp_output_size=8, global_op=3,num_blocks=4)



     detList = self.SBTcell_map()
     energy_array = np.zeros(2000)
     # time_array = np.full(2000, -9999) #default value is -9999

     for aDigi in self.sTree.Digi_SBTHits:
         detID = str(aDigi.GetDetectorID())
         ID_index = [lastname for lastname, firstname in detList.items() if firstname == detID][0]
         energy_array[ID_index] = aDigi.GetEloss()
         # time_array[ID_index] = aDigi.GetTDC()

     nHits = len(self.sTree.UpstreamTaggerPoint)

     if candidate == None: candidate = self.sTree.Particles[0]

     candidatePos = ROOT.TLorentzVector()
     candidate.ProductionVertex(candidatePos)
     vertexposition = np.array([candidatePos.X(), candidatePos.Y(), candidatePos.Z()])

     self.inputmatrix.append(np.concatenate(
         (energy_array,
          # time_array,
          vertexposition,
          # candidate_details,
          # np.array(nHits),
          # np.array(weight_i)
          )
         , axis=None
     ))  # inputmatrix has shape (1,2003)

     # After all data is collected, convert the list to a NumPy array
     self.inputmatrix = np.array(self.inputmatrix, dtype=np.float32)

     outputs, decisions, classification = gnn_output(model, x, XYZ)
  # returns True if to be vetoed

     return decisions, classification  # class 0 =Signal, class 1 = neuDIS, class 2 =muonDIS


 def SVT_decision(self,mcParticle=None):
  nHits = 0
  for ahit in self.sTree.strawtubesPoint:
     if mcParticle:
        if mcParticle>0 and mcParticle != ahit.GetTrackID() : continue
        if mcParticle<0 and abs(mcParticle) == ahit.GetTrackID() : continue
     detID   = ahit.GetDetectorID()
     if detID<50000000: continue  # StrawVeto station = 5
     nHits+=1
  w = (1-self.SVTefficiency)**nHits
  veto = self.random.Rndm() > w
  # print 'SVT :',nHits
  return veto,w,nHits
 def RPC_decision(self,mcParticle=None):
  nHits = 0
  mom = ROOT.TVector3()
  for ahit in self.sTree.ShipRpcPoint:
   if mcParticle:
        if mcParticle>0 and mcParticle != ahit.GetTrackID() : continue
        if mcParticle<0 and abs(mcParticle) == ahit.GetTrackID() : continue
   ahit.Momentum(mom)
   if mom.Mag() > 0.1: nHits+=1
  w = 1
  veto = nHits > 0 # 10  change to >0 since neutrino background ntuple not possible otherwise
  if veto: w = 0.
  #print 'RPC :',nHits
  return veto,w,nHits

 def UBT_decision(self, mcParticle=None):
  nHits = 0
  mom = ROOT.TVector3()
  
  for ahit in self.sTree.UpstreamTaggerPoint:
     if mcParticle:
        if mcParticle > 0 and mcParticle != ahit.GetTrackID():
         continue
        if mcParticle < 0 and abs(mcParticle) == ahit.GetTrackID():
         continue
     #ahit.Momentum(mom)
     #if mom.Mag() > 0.1:
     nHits+=1
  
  #w = (1 - self.UBTefficiency) ** nHits
  #veto = self.random.Rndm() > w
  
  if nHits: veto=True     
  else:     veto=False 

  return veto,nHits

 def Track_decision(self,mcParticle=None):
  nMultCon = 0
  k = -1
  for aTrack in self.sTree.FitTracks:
     k+=1
     if mcParticle:
        if mcParticle>0 and mcParticle != ahit.GetTrackID() : continue
        if mcParticle<0 and abs(mcParticle) == ahit.GetTrackID() : continue
     fstatus =  aTrack.getFitStatus()
     if not fstatus.isFitConverged() : continue
     if fstatus.getNdf() < 25: continue
     nMultCon+=1
  w = 1
  veto = nMultCon > 2
  if veto: w = 0.
  return veto,w,nMultCon
 def fiducialCheckSignal(self,n):
  hnl = self.sTree.Particles[n]
  aPoint = ROOT.TVector3(hnl.Vx(),hnl.Vy(),hnl.Vz())
  distmin = self.fiducialCheck(aPoint)
  return distmin
 def fiducialCheck(self,aPoint):
  nav = ROOT.gGeoManager.GetCurrentNavigator()
  phi = 0.
  nSteps = 36
  delPhi = 2.*ROOT.TMath.Pi()/nSteps
  distmin = 1E10
  nav.SetCurrentPoint(aPoint.x(),aPoint.y(),aPoint.z())
  cNode = 'outside'
  aNode = nav.FindNode()
  if aNode:
   cNode = aNode.GetName()
  if cNode not in ('DecayVacuum_block4_0',
                   'DecayVacuum_block5_0',
                   'DecayVacuum_block3_0',
                   'DecayVacuum_block2_0',
                   'DecayVacuum_block1_0'):
   distmin = 0.
  else:
   for n in range(nSteps):
   # set direction
    xDir = ROOT.TMath.Sin(phi)
    yDir = ROOT.TMath.Cos(phi)
    nav.SetCurrentPoint(aPoint.x(),aPoint.y(),aPoint.z())
    cNode = nav.FindNode().GetName()
    nav.SetCurrentDirection(xDir,yDir,0.)
    rc = nav.FindNextBoundaryAndStep()
    x,y  = nav.GetCurrentPoint()[0],nav.GetCurrentPoint()[1]
    if cNode != nav.GetCurrentNode().GetName():
     dist = ROOT.TMath.Sqrt( (aPoint.x()-x)**2 + (aPoint.y()-y)**2)
     if dist < distmin : distmin = dist
    phi+=delPhi
# distance to Tr1_x1
   nav.cd("/Tr1_1")
   shape = nav.GetCurrentNode().GetVolume().GetShape()
   origin = array('d',[0,0,shape.GetDZ()])
   master = array('d',[0,0,0])
   nav.LocalToMaster(origin,master)
   dist = master[2] - aPoint.z()
   if dist < distmin : distmin = dist
# distance to straw veto:
   nav.cd("/Veto_5")
   shape = nav.GetCurrentNode().GetVolume().GetShape()
   origin = array('d',[0,0,shape.GetDZ()])
   master = array('d',[0,0,0])
   nav.LocalToMaster(origin,master)
   dist = aPoint.z() - master[2]
  return distmin

#usage
# import shipVeto
# veto = shipVeto.Task(sTree)
# veto,w = veto.SBT_decision()
# or for plastic veto,w = veto.SBT_decision(detector='plastic')
# if veto: continue # reject event
# or
# continue using weight w
