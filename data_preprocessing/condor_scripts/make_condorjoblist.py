import os
import csv
import re
from argparse import ArgumentParser
import ROOT

parser = ArgumentParser()

parser.add_argument("-ana",       	dest="analysistype",	help="muDIS/neuDIS/EMBG",		required=False, default=False)
options = parser.parse_args()


path_to_output="./"

if options.analysistype=="muDIS":
	options.filename="joblists_muDIS_ECN3_2024.csv" 
	inputDir_list=['/eos/experiment/ship/simulation/bkg/MuonDIS_2024helium/8070735/SBT','/eos/experiment/ship/simulation/bkg/MuonDIS_2024helium/8070735/Tr'] #

if options.analysistype=="neuDIS":
	options.filename="joblists_neuDIS_ECN3_2024.csv"

	inputDir_list=['/eos/experiment/ship/user/Iaroslava/train_sample_N2024_big/']

if options.analysistype=="EMBG":
	options.filename="joblists_EMBG_ECN3_2024.csv" 
	inputDir_list=['/eos/experiment/ship/simulation/bkg/MuonBack_2024helium/8070735']	
		
with open(path_to_output+options.filename, 'w') as filekey: 
	csvwriter = csv.writer(filekey)
	for inputDir in inputDir_list:
		for inputFile in os.listdir(inputDir):
			
			if options.filename=="joblists_neuDIS_ECN3_2024.csv" and inputFile.startswith('job_'): 
				csvwriter.writerow([inputFile])

			if options.filename.startswith("joblists_muDIS_ECN3"):	
				tag=inputDir.split('/')[-1]
				try:
					with ROOT.TFile.Open(f"{inputDir}/{inputFile}/ship.conical.muonDIS-TGeant4_rec.root","read") as rootfile:	
						tree = rootfile.cbmsim
						nEvents= tree.GetEntries()
						startEvent=0
						while startEvent<nEvents:
							csvwriter.writerow([f"{tag}/{inputFile}",startEvent])
							startEvent+=100
				except Exception as e:
					print(e)
			
			if options.filename=="joblists_EMBG_ECN3_2024.csv":
				csvwriter.writerow([inputFile])	


print(options.filename," file created; type: ",options.analysistype)
			
				




