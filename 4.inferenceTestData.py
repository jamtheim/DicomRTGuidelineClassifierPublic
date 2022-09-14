# *********************************************************************************
# Author: Christian Jamtheim Gustafsson, PhD, Medical Physcist Expert
# Skåne University Hospital, Lund, Sweden and Lund University, Lund, Sweden
# Description: Perform inference of DICOM or Nifti data data with majority voting 
# *********************************************************************************

import os
import numpy as np
import glob
import pandas as pd
import operator
import torch
import timeit
import shutil
from datetime import datetime
from joblib import Parallel, delayed
import multiprocessing
from createDatasetClassification import preProcessSubject # Same preprocess as for training data
# Load modules
from commonConfig import commonConfigClass
from convertDataMethods import convertDataMethodsClass
from ioDataMethods import ioDataMethodsClass
from inferenceDataMethods import inferenceDataMethodsClass, niiTestDataset
from trainClassificationModel import LitModel
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from pytorch_lightning.plugins import DDPPlugin
from monai.networks.nets import SEResNet152, SEResNext101, EfficientNetBN, SENet154
# Init needed class instances
convertData = convertDataMethodsClass() # Functions for converting DICOM to Nifty data
ioData = ioDataMethodsClass()           # Functions for reading an processing data 
infData = inferenceDataMethodsClass()   # Inference class instance
conf = commonConfigClass()              # Init config class

# Set this in config file 
# data.TaskNumber = 'inf_Task104_PerStructure'
# Define if data is DICOM or Nifti in the input
#dataType = 'DICOM' # Else 'Nifti'
dataType = 'Nifti'
# Set which model iterations to use for inference, can be multiple ones. 
#versionIterCollection =[12] #Prostate + AI
versionIterCollection =[30] #Bowel

# Entry point for the script
if __name__ == "__main__":
    # Get current directory
    codePath = os.path.dirname(os.path.realpath(__file__))
    # Change directory to code path
    os.chdir(codePath)
    # Set task number 
    TaskNumber = conf.data.TaskNumber
    print('This is inference for task ' + TaskNumber)
    # Get all subject folders 
    subjectFolders = os.listdir(conf.data.dataInputPathInf)
    # Make sure it is a folder, otherwise remove non-folder items
    subjectFolders = [folder for folder in subjectFolders if os.path.isdir(os.path.join(conf.data.dataInputPathInf, folder))]
    # Make sure the folder does not contain the name Nifti (created later) 
    subjectFolders = [folder for folder in subjectFolders if 'Nifti' not in folder]
    # Set input path depending on data type
    if dataType == 'DICOM':
        # Convert DICOM to Nifti data for each subject 
        Parallel(n_jobs=conf.base.nrCPU, verbose=10)(delayed(convertData.DicomToNifti)(i_subject, subject, conf.data.dataInputPathInf, os.path.join(conf.data.dataInputPathInf, 'Nifti')) for i_subject, subject in enumerate(subjectFolders))
        dataInputPathInf = os.path.join(conf.data.dataInputPathInf, 'Nifti')
    if dataType == 'Nifti':
        dataInputPathInf = conf.data.dataInputPathInf
    # For each subject in the dataset initiate preprocessing in parallell manner. Subject names are read from the patient folder and data from Nifti data (if different)  
    Parallel(n_jobs=conf.base.nrCPU, verbose=10)(delayed(preProcessSubject)(i_subject, subject, dataInputPathInf, conf.data.dataOutputPathInf, TaskNumber) for i_subject, subject in enumerate(subjectFolders))

# Auto load the rest of the configuration needed for the model
num_classes = len(conf.data.classStructures) + 1 # +1 for other class
channels = conf.model.channels
height = conf.model.desiredImageMatrixSize[0]
width = conf.model.desiredImageMatrixSize[1]
depth = conf.model.desiredImageMatrixSize[2]
modelName = eval(conf.model.baseArchitecture)
# Not needed for model inference but needs to be defined for the model
learning_rate = [] 
workDir = [] 
classWeights = [] 
targetNames = [] 
crossValCounter = []  

# For each model iteration in versionIterCollection do the following
for versionIter in versionIterCollection:
    # Start measuring time
    start = datetime.now()
    # Get current directory 
    codePath = os.path.dirname(os.path.realpath(__file__))
    # Model dir path
    modelDir = os.path.join(codePath, conf.model.modelDir, conf.base.ProjectName, str(versionIter))
    # Get number of models available in the model directory 
    nrModels = len(os.listdir(modelDir)) 
    # Set lightning trainer options for inference 
    trainer = Trainer(accelerator="gpu",
                #devices = 1,
                gpus=[0], #Use only one GPU, there were problems with two! 
                precision=16,
                )
    # Define where the test data is located
    testDataDir = os.path.join(conf.data.dataOutputPathInf, TaskNumber, conf.base.dataOutputStructureDir)
    # Get the paths for all files of interest 
    testPathsFull = glob.glob(os.path.join(testDataDir, '*.npz')) 
    # Create a dataset instance
    testData = niiTestDataset(testPathsFull)
    # Define data loader for test data
    def test_dataloader():
            return DataLoader(testData, batch_size=1, num_workers=10, shuffle=False, persistent_workers=True, pin_memory=True)

    # Init dictionary for saving results from each model 
    resultDict = {} # Reset for every model iteration
    # Loop over all models for inference 
    for i_model in range(nrModels):
    #for i_model in range(0,0):
        # Print model
        print('Model name ' + str(i_model) + ' (total ' + str(nrModels) + ')')
        # Get model dir path
        i_modelDir = os.path.join(modelDir, conf.model.crossValidationDir + str(i_model))
        # Make sure folder only contains one model file, outputed as the best from training 
        assert len(os.listdir(i_modelDir)) == 1, 'There should be only one model file in the folder'
        # Get model file from the model dir
        i_modelFile = os.listdir(i_modelDir)[0]
        # Load the model file 
        model = LitModel.load_from_checkpoint(os.path.join(i_modelDir, i_modelFile),
            channels=channels, width=width, height=height , depth=depth, num_classes=num_classes, learning_rate=learning_rate, workDir=workDir, versionIter=versionIter, modelName=modelName, classWeights=classWeights, targetNames=targetNames, crossValCounter=crossValCounter)
        # Run model in test mode and save results to the dictionary 
        resultDict[i_model] = trainer.test(model, dataloaders=test_dataloader())
    # resultDict has the form: resultDict[i_model][0] # These are the fileNames and predictions for each model 
    assert len(resultDict) == nrModels, "Number of models in result dictionary should be equal to the number of models"
    # Merge all predictions from all models into one dictionary so format will be key/file:[all model predictions]
    resultDictAllModles = infData.mergeDicts([resultDict[i][0] for i in range(nrModels)])
    # Create a new dictionary with the majority vote, frequency and agreeingModels for each file/key 
    resultDictAllMajVote = infData.createMajorityVoteDict(resultDictAllModles, nrModels)
    # Majority vote can be accessed by resultDictAllMajVote['key']['majVote']
    print('Inference has finished!')
    # Sort the dictionary by key names (i.e subject) 
    resultDictAllMajVoteSorted = sorted(resultDictAllMajVote.items(), key=operator.itemgetter(0))
    resultDictAllMajVoteSorted = dict(resultDictAllMajVoteSorted)
    
    # Copy classified data to a new folder with subfolders for each class and CT image for each subject
    infData.copyClassifiedData(resultDictAllMajVoteSorted, dataInputPathInf, TaskNumber, versionIter)
        
    # If evaluate option enabled 
    if conf.model.evalInference: 
        # Get GT values with the same sorting as the resultDictAllMajVoteSorted
        testDataTargets = torch.tensor(infData.getGTdata(resultDictAllMajVoteSorted.keys(), conf.data.dataInputPathInfGT))
        # This row can be used to create a draft of GT which must be manually checked and corrected. Save in correct place and run line above. 
        # Run preferably with model.useInferenceBlacklist = True to get better suggestions for ground truth when reviewing. 
        #testDataTargets = torch.tensor(ioData.getTargetVector(resultDictAllMajVoteSorted.keys(), conf.data.classStructures)) 
    else: 
        # Create array with just NaN values so result file still can be produced 
        testDataTargets = torch.tensor(np.nan*np.ones(len(resultDictAllMajVoteSorted)))

    # Assert same number of objects as ground truth
    assert len(resultDictAllMajVoteSorted) == len(testDataTargets), "Number of objects in result dictionary should be equal to the number of objects in ground truth"

    # Create a pandas data frame with majority vote data 
    df = pd.DataFrame(data=resultDictAllMajVoteSorted).T
    # Insert the defined ground truth as the first column in the data frame 
    df.insert(0, 'GT', testDataTargets)
    # Write the whole pandas data frame to an Excel file
    df.to_excel(os.path.join(conf.data.dataOutputPathInf, TaskNumber, 'infResults_' + str(TaskNumber) + '_versionIter' + str(versionIter) + '_majorityVote.xlsx'))
    print('Data has been written to Excel!')
    # Stop measuring time
    stop = datetime.now()
    # Calculate time difference
    computeTime = stop - start
    print(computeTime)

    # If evaluate option enabled (requires ground truth)
    if conf.model.evalInference: 
        # Calculate the accuracy metrics 
        # Get majority predictions from sorted order dictionary  
        testDataPredictions = torch.tensor([resultDictAllMajVoteSorted[i]['majVote'] for i in resultDictAllMajVoteSorted.keys()])
        # Calculate and print accuracy metrics with pytorch lightning methods
        print('Accuacy metrics: ')
        print(' ')
        # Define results files 
        resultLightningMetricFilePath = os.path.join(conf.data.dataOutputPathInf, TaskNumber, 'infMetricsLightning_' + str(TaskNumber) + '_versionIter' + str(versionIter) + '_majorityVote_useInfBlacklist_' + str(conf.model.useInferenceBlacklist) + '.csv')
        resultConfMatrixMetricFilePath = os.path.join(conf.data.dataOutputPathInf, TaskNumber, 'infMetricsConfMatrix_' + str(TaskNumber) + '_versionIter' + str(versionIter) + '_majorityVote_useInfBlacklist_' + str(conf.model.useInferenceBlacklist) + '.csv')
        # Calculate and save data 
        infData.calcAndPrintAccuracyMetrics(testDataPredictions, testDataTargets, num_classes, resultLightningMetricFilePath, computeTime)
        # Calculate and save confusion matrix and results CSV file 
        infData.calcAndPrintConfusionMatrix(testDataPredictions, testDataTargets, conf.base.dataSetInf, resultConfMatrixMetricFilePath)




