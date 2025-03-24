#!/usr/bin/env python
"""Script to format neuDIS data file for NN studies."""

import ROOT
import uproot
import numpy as np
from rootpyPickler import Unpickler
import os
import shipunit as u
from argparse import ArgumentParser
import h5py

parser = ArgumentParser(description=__doc__);
parser.add_argument("-i", "--jobDir",dest="jobDir",help="job name of input file",  default='job_2',  type=str)
parser.add_argument("-p", "--path",dest="path",required=False,help="Path to the reconstructed neuDIS simulation folder")
options = parser.parse_args()

class EventDataProcessor:
    def __init__(self, input_file, geo_file, output_dir):
        self.input_file = input_file
        self.geo_file = geo_file
        self.output_dir = output_dir
        self.global_candidate_id = 0
        self.inputmatrix = []
        self.truth = []
        self.detList = None  # Placeholder for cached SBT cell map

        self.load_geofile()
    
    def load_geofile(self):
        """
        Load the geometry file and set the global geometry manager.
        """
        try:
            fgeo = ROOT.TFile(self.geo_file)  
            self.fGeo = fgeo.FAIRGeom  
            ROOT.gGeoManager = self.fGeo  

            upkl    = Unpickler(fgeo)
            self.ShipGeo = upkl.load('ShipGeo')
            print(f"Loaded geometry file: {self.geo_file}")
        except Exception as e:
            raise FileNotFoundError(f"Error loading geo file: {self.geo_file}. Error: {e}")

    def SBTcell_map(self): #provides a cell map with index in [0,853] for each cell.

        if self.detList is not None:
            return  # If the map is already built, no need to rebuild
        try:
            fGeo = ROOT.gGeoManager
            detList = {}
            LiSC = fGeo.GetTopVolume().GetNode('DecayVolume_1').GetVolume().GetNode('T2_1').GetVolume().GetNode('VetoLiSc_0')
            index = -1
            for LiSc_cell in LiSC.GetVolume().GetNodes():
                index += 1
                name = LiSc_cell.GetName()
                detList[index] = name[-6:]
            return detList
        except Exception as e:
            print(e)

    def define_t_vtx(self,sTree,candidate):
          
          t0=sTree.ShipEventHeader.GetEventTime()
          
          candidatePos = ROOT.TLorentzVector()
          candidate.ProductionVertex(candidatePos)

          d1, d2 = candidate.GetDaughter(0), candidate.GetDaughter(1)        
          d1_mc, d2_mc = sTree.fitTrack2MC[d1], sTree.fitTrack2MC[d2]                   

          time_vtx_from_strawhits=[]        

          for hit in sTree.strawtubesPoint:            

             if not (int( str( hit.GetDetectorID() )[:1]) ==1 or int( str( hit.GetDetectorID() )[:1]) ==2) : continue #if hit.GetZ() > ( ShipGeo.TrackStation2.z + 0.5*(ShipGeo.TrackStation3.z - ShipGeo.TrackStation2.z) ): continue #starwhits only from T1 and T2 before the SHiP magnet .

             if not (hit.GetTrackID()==d1_mc or hit.GetTrackID()==d2_mc) : continue
               
             #t_straw    = hit.GetTime()
             t_straw    = sTree.MCTrack[0].GetStartT()/1e4+(hit.GetTime()-sTree.MCTrack[0].GetStartT()) #resolving bug. to be changed for new productions
             
             d_strawhit  = [hit.GetX(),hit.GetY(),hit.GetZ()]

             dist     = np.sqrt( (candidatePos.X()-hit.GetX() )**2+( candidatePos.Y() -hit.GetY())**2+ ( candidatePos.Z()-hit.GetZ() )**2) #distance to the vertex #in cm            

             Mom          = sTree.MCTrack[hit.GetTrackID()].GetP()/u.GeV
             mass         = sTree.MCTrack[hit.GetTrackID()].GetMass()
             v            = u.c_light*Mom/np.sqrt(Mom**2+(mass)**2)

             t_vertex   = t_straw-(dist/v)

             time_vtx_from_strawhits.append(t_vertex)

              
          t_vtx=np.average(time_vtx_from_strawhits)+t0   

          return t_vtx  

    def ImpactParameter(self,point,tPos,tMom):
      t = 0
      if hasattr(tMom,'P'): P = tMom.P()
      else:                 P = tMom.Mag()
      for i in range(3):   t += tMom(i)/P*(point(i)-tPos(i)) 
      dist = 0
      for i in range(3):   dist += (point(i)-tPos(i)-t*tMom(i)/P)**2
      dist = ROOT.TMath.Sqrt(dist)
      return dist #in cm
    
    def define_weight(self,w_DIS,SHiP_running=15,N_gen=100000*98): #Each file has 100k events each change N_gen according to files(1) used for analysis, and 98 successful jobs
        
        nPOTinteraction     =(2.e+20)*(SHiP_running/5)
        nPOTinteraction_perspill =5.e+13

        n_Spill  = nPOTinteraction/nPOTinteraction_perspill #number of spill in SHiP_running(default=15) years
        
        nNu_perspill=4.51e+11       #number of neutrinos in a spill.
        
        N_nu=nNu_perspill*n_Spill   #Expected number of neutrinos in 15 years

        w_nu=nNu_perspill/N_gen     #weight of each neutrino considered scaled to a spill such that sum(w_nu)=(nNu_perspill/N_gen)*N_gen= nNu_perspill = number of neutrinos in a spill.
        
        N_A=6.022*10**23
        E_avg=2.57 #GeV
        sigma_DIS=7*(10**-39)*E_avg*N_A  #cross section cm^2 per mole
        
        return w_DIS*sigma_DIS*w_nu*n_Spill  #(rho_L*N_nu*N_A*neu_crosssection*E_avg)/N_gen     #returns the number of the DIS interaction events of that type in SHiP running(default=5) years.   #DIS_multiplicity=1 here
        
    def make_outputfile(self, filenumber):

        inputmatrix = np.array(self.inputmatrix)
        truth       = np.array(self.truth)
        
        rootfilename    = f"{self.output_dir}datafile_neuDIS_{filenumber}.root"

        file = uproot.recreate(rootfilename)
        file["tree"] = {
                    "inputmatrix": inputmatrix,
                    "truth": truth
                    }

        print(f"\n\nFiles formatted and saved in {rootfilename}")
        h5filename    = f"{self.output_dir}datafile_neuDIS_{filenumber}.h5"
        with h5py.File(h5filename, 'w') as h5file:
            for i in range(inputmatrix.shape[0]):
                event_name = f"event_{i}"
                event_group = h5file.create_group(event_name)
                event_group.create_dataset('data', data=inputmatrix[i])
                event_group.create_dataset('truth', data=truth[i])
        print(f"\n\nFiles formatted and saved in {h5filename}")
        self.inputmatrix = []

    def process_event(self, sTree, eventNr):

        detList = self.SBTcell_map()
        energy_array = np.zeros(854)
        time_array = np.full(854, -9999) #default value is -9999
        
        rho_L    =  sTree.MCTrack[0].GetWeight()
        weight_i =  self.define_weight(rho_L,SHiP_running=15)
        
        t0=sTree.ShipEventHeader.GetEventTime()

        for aDigi in sTree.Digi_SBTHits:
            detID = str(aDigi.GetDetectorID())
            ID_index = [lastname for lastname, firstname in detList.items() if firstname == detID][0]
            energy_array[ID_index] = aDigi.GetEloss()
            #time_array[ID_index] = aDigi.GetTDC()
            correctedTDC=sTree.MCTrack[0].GetStartT()/1e4+(aDigi.GetTDC()-t0-sTree.MCTrack[0].GetStartT()) +t0 #to be changed for new productions after 25.11.2024, resolving existing bug
            time_array[ID_index] = float(correctedTDC)


        #nHits=len(sTree.UpstreamTaggerPoint)

        for signal in sTree.Particles:
            
            signalPos = ROOT.TLorentzVector()
            signal.ProductionVertex(signalPos)
            
            inv_mass = signal.GetMass()
            signalMom = ROOT.TLorentzVector()
            signal.Momentum(signalMom)
            Target_Point = ROOT.TVector3(0, 0, self.ShipGeo.target.z0)
            Impact_par = self.ImpactParameter(Target_Point, signalPos, signalMom)

            track_1, track_2 = signal.GetDaughter(0), signal.GetDaughter(1)

            fitStatus_1 = sTree.FitTracks[track_1].getFitStatus()
            D1_chi2ndf = fitStatus_1.getChi2() / fitStatus_1.getNdf()
            D1_mom = sTree.FitTracks[track_1].getFittedState().getMom().Mag()

            fitStatus_2 = sTree.FitTracks[track_2].getFitStatus()
            D2_chi2ndf = fitStatus_2.getChi2() / fitStatus_2.getNdf()
            D2_mom = sTree.FitTracks[track_2].getFittedState().getMom().Mag()

            candidate_details = np.array([
                len(sTree.Particles),
                signal.GetMass(),
                signal.GetDoca(),
                Impact_par,
                D1_chi2ndf,
                D2_chi2ndf,
                fitStatus_1.getNdf(),
                fitStatus_2.getNdf(),
                D1_mom,
                D2_mom
            ])
            
            vertexposition = np.array([signalPos.X(), signalPos.Y(), signalPos.Z()])

            t_vtx=self.define_t_vtx(sTree,signal)
            
            self.inputmatrix.append(np.concatenate(
                                                   (energy_array,
                                                    time_array,
                                                    vertexposition,
                                                    np.array(t_vtx),#np.array(nHits),
                                                    np.array(weight_i)
                                                    ,candidate_details
                                                    )
                                                    , axis=None
                                                    ) )# inputmatrix has shape (nEvents,size of inputarray)


            self.truth.append(1)
            self.global_candidate_id += 1

    def process_file(self):

        f = ROOT.TFile.Open(self.input_file)
        filenumber = self.input_file.split("job")[1][1:-36]
        sTree = f.cbmsim
        nEvents = sTree.GetEntries()
        
        print(nEvents)

        for eventNr,event in enumerate(sTree):
            
            if hasattr(event, 'Particles') and len(event.Particles) and len(event.Digi_SBTHits):
                print(f"neuDIS {filenumber} {eventNr} {self.global_candidate_id} ---> {len(event.Particles)} reconst. particle(s) in event")
                self.process_event(event, eventNr)
                
            
        self.make_outputfile(filenumber)

    def read_outputdata(self):

        inputmatrixlist,truthlist=[],[]
            
        for datafile in os.listdir(self.output_dir):

            if not datafile.startswith("datafile_neuDIS_"+options.jobDir.split('_', 1)[1]): continue

            if not 'root' in datafile: continue
                        
            tree = uproot.open(self.output_dir+datafile)["tree"]
            data = tree.arrays(['inputmatrix', 'truth'], library='np')
            
            inputmatrix     = data['inputmatrix']
            truth           = data['truth'] 
            
            inputmatrixlist.append(inputmatrix)
            truthlist.append(truth)

        inputmatrix = np.concatenate(inputmatrixlist)
        truth = np.concatenate(truthlist)
        print(f"\n\nNumber of events available:{len(inputmatrix)} ")
        print(f"\nTest print event 0:\n------------------------------------------")
        print("Number of SBThits:", np.count_nonzero(inputmatrix[0][:854]),"/",len(inputmatrix[0][:854]))
        print("\tshould match timing entries:",np.sum(inputmatrix[0][854:1708] != -9999),"/",len(inputmatrix[0][854:1708]))
        print("\nvertexposition",inputmatrix[0][1708:1711])
        print("\nVertex time:",inputmatrix[0][1711],"ns")
        print("\nEvent weight:",inputmatrix[0][1712]," over 15 years")
        
        signal_details=inputmatrix[0][1713:]
        print("\nOther Candidate details:")
        print(f"\tlen(sTree.Particles)\t{signal_details[0]}")
        print(f"\tsignal.GetMass()\t{signal_details[1]}")
        print(f"\tsignal.GetDoca()\t{signal_details[2]}")
        print(f"\tImpact_par\t{signal_details[3]}")
        print(f"\tDaughter1_chi2ndf\t{signal_details[4]}")
        print(f"\tDaughter2_chi2ndf\t{signal_details[5]}")
        print(f"\tfitStatus_Daughter1.getNdf()\t{signal_details[6]}")
        print(f"\tfitStatus_Daughter2.getNdf()\t{signal_details[7]}")
        print(f"\tDaughter1_mom\t{signal_details[8]}")
        print(f"\tDaughter2_mom\t{signal_details[9]}")

# Usage:

if options.path:
    path = options.path
else:
    path = '/eos/experiment/ship/user/Iaroslava/train_sample_N2024_big/'#'/eos/experiment/ship/user/Iaroslava/train_sample_N2024/'<---older smaller sample.

processor = EventDataProcessor(input_file=path+options.jobDir+"/ship.conical.Genie-TGeant4_rec.root" , geo_file=path+options.jobDir+"/geofile_full.conical.Genie-TGeant4.root", output_dir="./")
processor.process_file()

processor.read_outputdata()

