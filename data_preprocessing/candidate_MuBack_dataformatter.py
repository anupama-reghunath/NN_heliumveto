#!/usr/bin/env python
"""
Script to format candidate+Muinduced Background data file for NN studies.
SBT response for each candidate event is combined with EMBG events on an MC level (and then digitised) .
The number of EMBG events to combine is determined by the weight of the EMBG event.
"""
from argparse import ArgumentParser
import numpy as np
import os,ROOT
import uproot
from rootpyPickler import Unpickler
from tabulate import tabulate
import h5py
#import rootUtils as ut
import shipunit as u
import csv
import time
import glob


parser = ArgumentParser(description=__doc__);
parser.add_argument("-i", "--jobDir",dest="jobDir",help="job name of input file",  type=str)
parser.add_argument("-s", "--startEvent",dest="startEvent",help="start Event of the candidate file", type=int, default=0)
parser.add_argument("-n", "--nEvents",dest="nEvents",help="nEvents per root file", type=int, default=100)
parser.add_argument("--muDIS", "--muDIS"	,dest="muDIS",help="produce muonDIS+MuBack files", required=False, action='store_true',default=False)
parser.add_argument("--neuDIS", "--neuDIS"	,dest="neuDIS",help="produce neuDIS+MuBack files", required=False, action='store_true',default=False)
parser.add_argument("--signal", "--signal"	,dest="signal",help="produce signal+MuBack files", required=False, action='store_true',default=False)
parser.add_argument('--embg_path', dest='embg_path' , help='Path to MuonBack files'	, required=False,default='/eos/experiment/ship/simulation/bkg/MuonBack_2024helium/8070735',type=str)
parser.add_argument("--test"          , dest="testing_code" , help="Run Test"              , required=False, action="store_true",default=False)
options = parser.parse_args()

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

#h={}

#ut.bookHist(h,  "candidate_t0","; candidate t0 (ns); nEvents", 1000,0,10^9)
#ut.bookHist(h,  "bg_t0","; MuonBack t0 (ns); nEvents", 1000,0,10^9)
#ut.bookHist(h,  "nSBThits","; nSBThits per checked event (no threshold); nEvents checked", 1000,0,1000)

class Candidate_MuBack_dataformatter:
	def __init__(self,candidate_path,bg_path,tag):

		print(f"\n\nGenerating {tag}+ MuBack samples now\n\nMuBack filepath={bg_path}\nCandidate filepath: {candidate_path}\n\n")
		
		self.tag=tag
		self.candidate_path=candidate_path
		self.bg_path=bg_path
		
		self.random=ROOT.TRandom()
		
		if options.testing_code:
			seed_value = int(123456)
		else:
			seed_value = int(time.time())
		
		print(f"Setting Seed: {seed_value}")
		self.random.SetSeed(seed_value)
		
		#ut.bookHist(h,  f"Edep_0MeV",f"Threshold= 0 MeV; E deposition per cell (GeV); nEvents checked", 100,0,1) #"nEvents checked" is after weighting and then sampling
		
		self.detList = None  # Placeholder for cached SBT cell map
		self.geo_file=None

		self.inputmatrix = []
		self.truth = [] 
		
		self.build_embgchain()
		
		self.print_file=False
		
		# At class level
		self.embg_meta = []  # List of dicts with embgNr, weight, etc.


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

	def generate_event_time_in_spill(self,eventweight,starttime=0,endtime=10**9):
		return np.array([self.random.Uniform(starttime, endtime) for _ in range(int(eventweight))])  

	
	def assign_event_time_candidate(self,nevents):

		print("Assigning t0 time for candidate events now...")

		self.candidate_eventtimes=self.generate_event_time_in_spill(eventweight=nevents,starttime=self.timeframe[0]+75,endtime=self.timeframe[1]-75)

		print(f"Signal times assigned,{len(self.candidate_eventtimes)} entries available")
		
	def build_embgchain(self):

		self.embg_chain = ROOT.TChain("cbmsim")

		for jobNr,job_folder in enumerate(os.listdir(self.bg_path)):
		
			if not job_folder.startswith('job'): continue # to ignore the README

			if "geofile_full.conical.MuonBack-TGeant4.root" not in os.listdir(f"{self.bg_path}/{job_folder}"): continue
			
			if options.testing_code and jobNr>50: break
			
			try:

				file_path=f"{self.bg_path}/{job_folder}/ship.conical.MuonBack-TGeant4_rec.root"
				self.embg_chain.Add(file_path)
				#print(f"{file_path} added to TChain")
				if self.geo_file is None:
					self.geo_file=os.path.join(f"{self.bg_path}/{job_folder}/geofile_full.conical.MuonBack-TGeant4.root")
					self.load_geofile()
				
			except Exception as e:
				print(f"build_embgchain error:{e}")

		print(f"Number of events in the MuBack sample {self.embg_chain.GetEntries()}")
		# Disable all branches then enable only those used in the analysis
		self.embg_chain.SetBranchStatus("*", 0)
		self.embg_chain.SetBranchStatus("Digi_SBTHits*", 1)
		self.embg_chain.SetBranchStatus("vetoPoint*", 1)
		self.embg_chain.SetBranchStatus("MCTrack*", 1)
		self.embg_chain.SetBranchStatus("digiSBT2MC*", 1)
		self.embg_chain.SetCacheSize(10000000)  # 10 MB cache, adjust as needed


	def assign_event_time_bg(self):
		
		start = time.time()

		print(f"time frame used:{self.timeframe}")

		print("Assigning t0 time for MuBack events now...")
		
		self.bg_eventtimes=[]
		
		for embgNr,self.embg_event in enumerate(self.embg_chain):

			for track in self.embg_event.MCTrack: 
				if track.GetPdgCode() in [-13,13]:
					self.weight_i=track.GetWeight() #(track.GetWeight()<---per spill)      (self.define_weight(track.GetWeight())<---weight over 5 years)
					break

			#if options.testing_code:
			#	eventtimes=self.generate_event_time_in_spill(self.weight_i,starttime=self.timeframe[0],endtime=self.timeframe[1]) 
			#else:
			eventtimes=self.generate_event_time_in_spill(self.weight_i) 
			
			valid_mask = (eventtimes > self.timeframe[0]) & (eventtimes < self.timeframe[1])
			valid_times = eventtimes[valid_mask]
			
			if valid_times.size == 0:
			    continue # if no event time falls within the timeframe
			
			#print(f"\tMuBack EventNr {embgNr} falls within the timeframe")			  
			
			# Fill histogram and record the valid times
			for t in valid_times:
			    #h['bg_t0'].Fill(t)
			    self.bg_eventtimes.append({"entry": embgNr, "t0": t})

		print(f"BG times assigned,{len(self.bg_eventtimes)} (weighted) events available within timeframe [{self.timeframe[0]},{self.timeframe[1]}]")

		end = time.time()

		print(f"[TIMER] assign_event_time_bg loop took: {end - start:.3f} seconds")

		if not len(self.bg_eventtimes):
			exit(1)


	def append_vetoPoints(self,vetoPoints,candidate_event=None):

		for aMCPoint in vetoPoints:

			detID=aMCPoint.GetDetectorID()
			Eloss=aMCPoint.GetEnergyLoss()

			if detID not in self.ElossPerDetId:
			    self.ElossPerDetId[detID]=0
			    self.tOfFlight[detID]=[]

			self.ElossPerDetId[detID] += Eloss

			if self.tag=='neuDIS'and candidate_event:
				hittime = candidate_event.MCTrack[0].GetStartT()/1e4+(aMCPoint.GetTime()-candidate_event.MCTrack[0].GetStartT()) #resolve time bug in production. to be removed for new productions post 2024
			else:
				hittime = aMCPoint.GetTime()

			self.tOfFlight[detID].append(hittime)


	def digitizecombinedSBT(self,candidate_t0):
	    
	    index=0 
	    digiSBT={}
	    
	    for detID in self.ElossPerDetId:
	        aHit = ROOT.vetoHit(detID,self.ElossPerDetId[detID])
	        aHit.SetTDC(min( self.tOfFlight[detID] )+ candidate_t0 )    
	        if self.ElossPerDetId[detID]<0.045:    aHit.setInvalid()  
	        digiSBT[index] = aHit
	        index=index+1
	    return digiSBT		

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

	def ImpactParameter(self,point,tPos,tMom):
	  t = 0
	  if hasattr(tMom,'P'): P = tMom.P()
	  else:                 P = tMom.Mag()
	  for i in range(3):   t += tMom(i)/P*(point(i)-tPos(i)) 
	  dist = 0
	  for i in range(3):   dist += (point(i)-tPos(i)-t*tMom(i)/P)**2
	  dist = ROOT.TMath.Sqrt(dist)
	  return dist #in cm  

	def make_outputfile(self, filenumber):

	    inputmatrix = np.array(self.inputmatrix)
	    truth       = np.array(self.truth)
	    
	    rootfilename    = f"datafile_{filenumber}.root"

	    file = uproot.recreate(rootfilename)
	    file["tree"] = {
	                "inputmatrix": inputmatrix,
	                "truth": truth
	                }
	    
	    h5filename    = f"datafile_{filenumber}.h5"
	    with h5py.File(h5filename, 'w') as h5file:
	        for i in range(inputmatrix.shape[0]):
	            event_name = f"event_{i}"
	            event_group = h5file.create_group(event_name)
	            event_group.create_dataset('data', data=inputmatrix[i])
	            event_group.create_dataset('truth', data=truth[i])
	    
	    print(f"\n\nData succesfully formatted and saved in {h5filename},{rootfilename} \n")

	    self.inputmatrix = []
	    self.truth = []  

	    return rootfilename


	def process_event(self, candidate_event,Digi_SBTHits):

		detList = self.SBTcell_map()
		energy_array = np.zeros(854)
		time_array = np.full(854, -9999) #default value is -9999

		if self.tag=='muDIS':
			weight_i= self.define_weight_muDIS(candidate_event,SHiP_running=15)

		if self.tag=='neuDIS':
			weight_i= self.define_weight_neuDIS(candidate_event,SHiP_running=15)

		if self.tag=='signal':
			weight_i= 1 #do the signal candidates have a weight?

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


	def combine_events(self,startEvent):

		start = time.time()

		# Once before the loop
		bg_event_t0s = np.array([e["t0"] for e in self.bg_eventtimes])
		bg_event_entries = np.array([e["entry"] for e in self.bg_eventtimes])

		print("Digihits with odd times are omitted!")
		
		tabulate_header = ['Candidate Event Index','Candidate t0 (ns)','Candidate_nVetoPoints','MuBack Event Index','MuBack t0 (ns)', 'MuBack nVetoPoints','Total_cumulate_nVetoPoints ','Total_Digi_SBTHits']
		
		for i,candidate_t0 in enumerate(self.candidate_eventtimes):
			
			candidateNr= startEvent+i 
			
			self.candidate_tree.GetEntry(candidateNr)
			
			candidate_event = self.candidate_tree
			
			if not len(candidate_event.Particles): continue
			if not len(candidate_event.vetoPoint): continue
			
				
			#h['candidate_t0'].Fill(candidate_t0) 
			table_data=[]

			self.ElossPerDetId    = {}
			self.tOfFlight        = {}
				
			self.append_vetoPoints(candidate_event.vetoPoint,candidate_event)
				
			mask = np.abs(bg_event_t0s - candidate_t0) <= 75
			matching_entries = bg_event_entries[mask]
			matching_t0s = bg_event_t0s[mask]
			
			for bg_t0, entry in zip(matching_t0s, matching_entries):

				self.embg_chain.GetEntry(entry)
				
				self.embg_event = self.embg_chain
								
				self.append_vetoPoints(self.embg_event.vetoPoint)

				table_data.append([
							candidateNr,  # Signal Event Index
							round(candidate_t0, 3),  # Signal Time (ns)
							len(candidate_event.vetoPoint), #number of vetoPoints in the candidate event
							entry,  # Background Event Number
							round(bg_t0, 3),  # BG Time (ns)
							len(self.embg_event.vetoPoint), #number of vetoPoints in the EMBG event
							sum(len(vetopoints) for vetopoints in self.tOfFlight.values()),
							len(self.ElossPerDetId),#number of digihits
							])
			
			combined_Digi_SBTHits=self.digitizecombinedSBT(candidate_event.ShipEventHeader.GetEventTime())
					
			self.process_event(candidate_event,combined_Digi_SBTHits)

			if combined_Digi_SBTHits:
				
				print(f"\nDigitisation Table:\n Candidate EventNr{candidateNr}, Number of recon candidates:{len(candidate_event.Particles)}, nMuBack events combined = {len(table_data)}, nDigihits: {len(combined_Digi_SBTHits)}")
				print(tabulate(table_data, headers=tabulate_header, tablefmt="pretty"))
				self.print_file=True


		end = time.time()

		print(f"[TIMER] combine_events loop took: {end - start:.3f} seconds")

	def run_analysis(self):
		
		file_name = glob.glob(f"{self.candidate_path}/ship.conical*_rec.root")[0]
		f_candidate = ROOT.TFile.Open(file_name,"read")
		print("Opened file:", file_name)

		self.candidate_tree = f_candidate.cbmsim
			
		print("\n\n------------------------------------------------------------------------------------------------------------------------\n\n")
		
		timeblockreference=self.random.Uniform(0+500, (10**9)-(500)) #reference time in the spill to set the timeframe
		
		self.timeframe=[timeblockreference-500,timeblockreference+500] #timeframe of 1/millionth of a spill
		
		self.assign_event_time_bg()
		
		nevents_to_check = min(options.nEvents,self.candidate_tree.GetEntries()-options.startEvent)
		
		print(f"nevents_to_check: {nevents_to_check}")
		
		self.assign_event_time_candidate(nevents_to_check)
		
		self.combine_events(options.startEvent)


		if self.print_file:
			subtag = f"batch_{options.startEvent//options.nEvents}"
			filenumber=f"{self.tag}_{options.jobDir}_{subtag}_MuBack"
			self.rootfilename=self.make_outputfile(filenumber)
			self.inspect_outputfile()
		print("------------------------------------------------------------------------------------------------------------------------")

	def inspect_outputfile(self):

	    inputmatrixlist,truthlist=[],[]
	                    
	    tree = uproot.open(self.rootfilename)["tree"]
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


# Initialize and run the analysis
abc = Candidate_MuBack_dataformatter(candidatefile_path,options.embg_path,tag)
abc.run_analysis()

#ut.writeHists(h,"plots_datafile.root")