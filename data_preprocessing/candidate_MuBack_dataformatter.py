#!/usr/bin/env python
"""Script to format candidate+Muinduced Background data file for NN studies.
SBT response for each signal candidate is combined with a randomly chosen EMBG file on an MC level (and then digitised) .
"""

import ROOT
import uproot
import numpy as np
from rootpyPickler import Unpickler
import os
import csv
from argparse import ArgumentParser
import shipunit as u
import random
import h5py
import glob

pdg = ROOT.TDatabasePDG.Instance()

#CHANGE signale file_path to candidate_file_path

parser = ArgumentParser();
parser = ArgumentParser(description=__doc__);
parser.add_argument("-i", "--jobDir",dest="jobDir",help="job name of input file",  type=str)
parser.add_argument("--muDIS", "--muDIS"	,dest="muDIS",help="produce muonDIS+MuBack files", required=False, action='store_true',default=False)
parser.add_argument("--neuDIS", "--neuDIS"	,dest="neuDIS",help="produce neuDIS+MuBack files", required=False, action='store_true',default=False)
parser.add_argument("--signal", "--signal"	,dest="signal",help="produce signal+MuBack files", required=False, action='store_true',default=False)
parser.add_argument('--embg_path', dest='embg_path' , help='path to the MuonBack files'	, required=False,default='/eos/experiment/ship/simulation/bkg/MuonBack_2024helium/8070735')
parser.add_argument('--test'    	, dest='testing_code' 	, help='Run Test'   , required=False, action='store_true',default=False)
options = parser.parse_args()

embg_path=options.embg_path

if not(options.muDIS or options.neuDIS or options.signal):
	print("Select a candidate type\nExit Normally")
	exit(0)


if options.muDIS:
	tag='muDIS'
	candidatefile_path='/eos/experiment/ship/simulation/bkg/MuonDIS_2024helium/8070735'
	if options.testing_code:
		options.jobDir='SBT/job_0'
	candidatefile_path=f"/eos/experiment/ship/simulation/bkg/MuonDIS_2024helium/8070735/{options.jobDir}"
	options.jobDir=options.jobDir.replace('/','_')
if options.neuDIS:
	tag='neuDIS'
	if options.testing_code:
		options.jobDir='job_2'
	candidatefile_path = f"/eos/experiment/ship/user/Iaroslava/train_sample_N2024_big/{options.jobDir}"
	
if options.signal:
	tag='signal'
	options.jobDir='job_0'
	candidatefile_path='/eos/experiment/ship/user/anupamar/signal/mumunu'

print(f"\n\nGenerating {tag}+ MuBack samples now\n\nMuBack filepath={embg_path}\nCandidate filepath: {candidatefile_path}\n\n")

if options.testing_code:

	seed_value = 42
else:
	import time	
	seed_value = int(time.time())

print(f"Setting Seed= {seed_value}")

random.seed(seed_value)


#-----------------------------------------------------------------------------

class EventDataProcessor:
	def __init__(self, input_path,candidate_path, output_dir,tag):
		self.tag=tag
		self.input_path = input_path
		self.candidate_path= candidate_path
		self.geo_file=None #will be updated according to the candidate_file
		self.output_dir = output_dir
		self.global_event_id = 0
		self.inputmatrix = []
		self.truth = [] 
		self.detList = None  # Placeholder for cached SBT cell map
		self.embg_file,self.MuBack_jobDir=self.choose_embg_file() 

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

	def dump(self,event,mom_threshold=0):

	    headers=['#','particle','pdgcode','mother_id','Momentum [Px,Py,Pz] (GeV/c)','StartVertex[x,y,z] (m)','Process', 'GetWeight()', ]
	    
	    event_table=[]
	    for trackNr,track in enumerate(event.MCTrack): 
	        
	        if track.GetP()/u.GeV < mom_threshold :  continue
	        
	        try: particlename=pdg.GetParticle(track.GetPdgCode()).GetName()
	        except: particlename='----'

	        event_table.append([trackNr,
	                        particlename,
	                        track.GetPdgCode(),
	                        track.GetMotherId(),
	                        f"[{track.GetPx()/u.GeV:7.3f},{track.GetPy()/u.GeV:7.3f},{track.GetPz()/u.GeV:7.3f}]",
	                        f"[{track.GetStartX()/u.m:7.3f},{track.GetStartY()/u.m:7.3f},{track.GetStartZ()/u.m:7.3f}]",
	                        track.GetProcName().Data(),
	                        track.GetWeight()
	                        ])
	    
	    print(tabulate(event_table,headers=headers,floatfmt=".3f",tablefmt='simple_outline'))

	def ImpactParameter(self,point,tPos,tMom):
	  t = 0
	  if hasattr(tMom,'P'): P = tMom.P()
	  else:                 P = tMom.Mag()
	  for i in range(3):   t += tMom(i)/P*(point(i)-tPos(i)) 
	  dist = 0
	  for i in range(3):   dist += (point(i)-tPos(i)-t*tMom(i)/P)**2
	  dist = ROOT.TMath.Sqrt(dist)
	  return dist #in cm

	def define_weight_MuBack(self,event_weight,SHiP_running=15):
	    
	    #event_weight=number of muon events per spill

	    nPOTinteraction     =(2.e+20)*(SHiP_running/5)
	    nPOTinteraction_perspill =5.e+13
	    n_Spill  = nPOTinteraction/nPOTinteraction_perspill #number of spill in 15 years
	    return event_weight*n_Spill

	def define_weight_muDIS(self,muDIS_event,SHiP_running=15):

	    w_mu=muDIS_event.MCTrack[0].GetWeight()  #weight of the incoming muon*DIS multiplicity normalised to a full spill   sum(w_mu) = nMuons_perspill = number of muons in a spill. w_mu is not the same as N_muperspill/N_gen, where N_gen = nEvents*DISmultiplicity ( events enhanced in Pythia to increase statistics) .

	    cross=muDIS_event.CrossSection

	    rho_l=muDIS_event.MCTrack[2].GetWeight()
	    
	    N_a=6.022e+23 

	    sigma_DIS=cross*1e-27*N_a #cross section cm^2 per mole
	    
	    nPOTinteraction     =(2.e+20)*(SHiP_running/5) #in years
	    nPOTinteraction_perspill =5.e+13
	    
	    n_Spill  = nPOTinteraction/nPOTinteraction_perspill  #Number of Spills in SHiP running( default=15) years
	    
	    weight_i = rho_l*sigma_DIS*w_mu*n_Spill 
	    
	    return weight_i    

	def define_weight_neuDIS(self,neuDIS_event,SHiP_running=15,N_gen=100000*98): #Each file has 100k events each change N_gen according to files(1) used for analysis, and 98 successful jobs
	    
	    w_DIS    =  neuDIS_event.MCTrack[0].GetWeight()
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

	def choose_embg_file(self):

		retry=True
		f=None

		while retry: 

		    inputFolders = [inp for inp in os.listdir(self.input_path) if os.path.isdir(os.path.join(self.input_path, inp))]
		    inputFolder = random.choice(inputFolders)

		    inputFolderPath=os.path.join(self.input_path, inputFolder)        

		    if 'geofile_full.conical.MuonBack-TGeant4.root' not in os.listdir(inputFolderPath): 
		    	print("No geofile present in {inputFolderPath}, retrying")
		    	continue
		    
		    embg_file=os.path.join(inputFolderPath, "ship.conical.MuonBack-TGeant4.root")
		    
		    try:

		    	with ROOT.TFile.Open(embg_file,"read") as f:
			        
			        embg_tree = f.cbmsim
			        print(f"Random Sampling events from MuonBack file:\n{embg_file}, nEntries: {embg_tree.GetEntries()}\n")
			        retry=False

		    except Exception as e:
		    	print(e)
		
		return embg_file,inputFolder
		        
	def digitizeSBT(self,embg_vetoPoints,candidate_vetoPoints,candidate_t0,candidate_event):

		ElossPerDetId    = {}
		tOfFlight        = {}
		#listOfVetoPoints = {}
		digiSBT={}
		key=-1

		for vetopoint_type,vetoPoints in enumerate([embg_vetoPoints,candidate_vetoPoints]):

			for aMCPoint in vetoPoints:
				key+=1
				detID=aMCPoint.GetDetectorID()
				Eloss=aMCPoint.GetEnergyLoss()
				if detID not in ElossPerDetId:
				    ElossPerDetId[detID]=0
				    #listOfVetoPoints[detID]=[]
				    tOfFlight[detID]=[]
				ElossPerDetId[detID] += Eloss
				#listOfVetoPoints[detID].append(key)

				if self.tag=='neuDIS'and (vetopoint_type==1): #only correct the neuDIS event
					hittime = candidate_event.MCTrack[0].GetStartT()/1e4+(aMCPoint.GetTime()-candidate_event.MCTrack[0].GetStartT()) #resolve time bug in production. to be removed for new productions post 2024
				else:
					hittime = aMCPoint.GetTime()
				tOfFlight[detID].append(hittime)


		index=0
		for detID in ElossPerDetId:

		    aHit = ROOT.vetoHit(detID,ElossPerDetId[detID])
		    aHit.SetTDC(min( tOfFlight[detID] )+ candidate_t0 )
		    if ElossPerDetId[detID]<0.045:    aHit.setInvalid()
		    digiSBT[index] = aHit
		    index=index+1
		return digiSBT


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

			if self.tag=='neuDIS':
				t_straw    = sTree.MCTrack[0].GetStartT()/1e4+(hit.GetTime()-sTree.MCTrack[0].GetStartT()) #resolving bug. to be changed for new productions post 2024
			else:
				t_straw    = hit.GetTime()

			d_strawhit  = [hit.GetX(),hit.GetY(),hit.GetZ()]

			dist     = np.sqrt( (candidatePos.X()-hit.GetX() )**2+( candidatePos.Y() -hit.GetY())**2+ ( candidatePos.Z()-hit.GetZ() )**2) #distance to the vertex #in cm

			Mom          = sTree.MCTrack[hit.GetTrackID()].GetP()/u.GeV
			mass         = sTree.MCTrack[hit.GetTrackID()].GetMass()
			v            = u.c_light*Mom/np.sqrt(Mom**2+(mass)**2)

			t_vertex   = t_straw-(dist/v)

			time_vtx_from_strawhits.append(t_vertex)

		t_vtx=np.average(time_vtx_from_strawhits)+t0

		return t_vtx



	def process_event(self, candidate_event,embg_event,Digi_SBTHits):

		detList = self.SBTcell_map()
		energy_array = np.zeros(854)
		time_array = np.full(854, -9999) #default value is -9999

		for track in embg_event.MCTrack: 
		        if track.GetPdgCode() in [-13,13]:
		                embg_weight=self.define_weight_MuBack(track.GetWeight(),SHiP_running=15)#<---weight over 15 years   (track.GetWeight()<---per spill)
		                break

		if self.tag=='muDIS':

			weight_i= embg_weight * self.define_weight_muDIS(candidate_event,SHiP_running=15)

		if self.tag=='neuDIS':
			weight_i= embg_weight * self.define_weight_neuDIS(candidate_event,SHiP_running=15)

		if self.tag=='signal':
			weight_i= embg_weight * 1 #do the signal candidates have a weight?


		for aDigi in Digi_SBTHits.values():

			detID = str(aDigi.GetDetectorID())
			ID_index = [lastname for lastname, firstname in detList.items() if firstname == detID][0]
			if aDigi.GetTDC()<10**7:
				energy_array[ID_index] = aDigi.GetEloss()
				time_array[ID_index] = float(aDigi.GetTDC())



		#nHits=len(embg_event.UpstreamTaggerPoint)

		for signal in candidate_event.Particles:    
		            
		    signalPos = ROOT.TLorentzVector()
		    signal.ProductionVertex(signalPos)
		    
		    inv_mass = signal.GetMass()
		    signalMom = ROOT.TLorentzVector()
		    signal.Momentum(signalMom)
		    Target_Point = ROOT.TVector3(0, 0, self.ShipGeo.target.z0)
		    Impact_par = self.ImpactParameter(Target_Point, signalPos, signalMom)

		    track_1, track_2 = signal.GetDaughter(0), signal.GetDaughter(1)

		    fitStatus_1 = candidate_event.FitTracks[track_1].getFitStatus()
		    D1_chi2ndf = fitStatus_1.getChi2() / fitStatus_1.getNdf()
		    D1_mom = candidate_event.FitTracks[track_1].getFittedState().getMom().Mag()

		    fitStatus_2 = candidate_event.FitTracks[track_2].getFitStatus()
		    D2_chi2ndf = fitStatus_2.getChi2() / fitStatus_2.getNdf()
		    D2_mom = candidate_event.FitTracks[track_2].getFittedState().getMom().Mag()

		    candidate_details = np.array([
		        len(candidate_event.Particles),
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
		    
		    t_vtx=self.define_t_vtx(candidate_event,signal)

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
		    
		    self.truth.append(0)

	def make_outputfile(self, filenumber):

	    inputmatrix = np.array(self.inputmatrix)
	    truth       = np.array(self.truth)
	    
	    rootfilename    = f"{self.output_dir}datafile_{filenumber}.root"

	    file = uproot.recreate(rootfilename)
	    file["tree"] = {
	                "inputmatrix": inputmatrix,
	                "truth": truth
	                }

	    print(f"\n\nFiles formatted and saved in {rootfilename}")
	    
	    h5filename    = f"{self.output_dir}datafile_{filenumber}.h5"
	    with h5py.File(h5filename, 'w') as h5file:
	        for i in range(inputmatrix.shape[0]):
	            event_name = f"event_{i}"
	            event_group = h5file.create_group(event_name)
	            event_group.create_dataset('data', data=inputmatrix[i])
	            event_group.create_dataset('truth', data=truth[i])
	    print(f"\n\nFiles formatted and saved in {h5filename}\n")

	    self.inputmatrix = []
	    self.truth = []  
	    return rootfilename

	def process_file(self):

		file_name = glob.glob(f"{self.candidate_path}/ship.conical*_rec.root")[0]
		f_candidate = ROOT.TFile.Open(file_name,"read")
		print("Opened file:", file_name)
		#f_candidate = ROOT.TFile.Open(f"{self.candidate_path}/ship.conical.Pythia8-TGeant4_rec.root","read")

		try:
			candidate_tree = f_candidate.cbmsim
			candidate_entries= candidate_tree.GetEntries()
			if self.geo_file==None:
				self.geo_file=glob.glob(f"{self.candidate_path}/geofile_full*.root")[0]
				#self.geo_file=os.path.join(f"{self.candidate_path}/geofile_full.conical.Pythia8-TGeant4.root")
				self.load_geofile()
				#print(f"File read successfully.\n")
		    
		except Exception as e:
		        print(e)


		print_file=False    
		
		f_embg=ROOT.TFile.Open(self.embg_file,"read")			    
		
		embg_tree=f_embg.cbmsim
		
		embg_entries=embg_tree.GetEntries()

		nEvents_looped=0
		
		while nEvents_looped < candidate_entries:
			
			if options.testing_code and nEvents_looped>500: 
				break
			
			candidate_tree.GetEntry(nEvents_looped)
			
			candidate_t0=candidate_tree.ShipEventHeader.GetEventTime()

			if not len(candidate_tree.Particles): 
				nEvents_looped+=1
				continue
			
			print(f"\nEvent {nEvents_looped}, {len(candidate_tree.Particles)} candidate(s) in event, {len(candidate_tree.Digi_SBTHits)} Digihits in the candidate event ")
			
			retry=True
			
			while retry:
				
				embg_index = random.randint(0,embg_entries - 1)

				embg_tree.GetEntry(embg_index)

				combined_Digi_SBTHits=self.digitizeSBT(embg_tree.vetoPoint,candidate_tree.vetoPoint,candidate_t0,candidate_tree)

				if combined_Digi_SBTHits: 
					print(f" Combining MuBack Event:{embg_index}...\n \t\t\t{len(combined_Digi_SBTHits)} Digihits in the combined event now")
					retry=False #only consider events with combined SBT activity
				
			self.process_event(candidate_tree,embg_tree,combined_Digi_SBTHits)

			nEvents_looped+=1

		
		filenumber=filenumber=f"{self.tag}{options.jobDir}_MuBack{self.MuBack_jobDir.split('_')[-1]}"
		self.rootfilename=self.make_outputfile(filenumber)
			
	def inspect_outputfile(self):

	    inputmatrixlist,truthlist=[],[]
	        
	    #for datafile in os.listdir(self.output_dir):

	        #if not datafile.startswith("datafile_signal"): continue
	        #if not datafile.endswith(".root"): continue
	                    
	    tree = uproot.open(self.output_dir+self.rootfilename)["tree"]
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

#-----------------------------------------------------------------------------
processor = EventDataProcessor(input_path=embg_path , candidate_path= candidatefile_path, output_dir="./",tag=tag)

processor.process_file()
processor.inspect_outputfile()