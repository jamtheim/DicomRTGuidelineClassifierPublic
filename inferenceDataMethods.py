# *********************************************************************************
# Author: Christian Jamtheim Gustafsson, PhD, Medical Physcist Expert
# Skåne University Hospital, Lund, Sweden and Lund University, Lund, Sweden
# Description: Class for model inference and evaluation from DICOM or Nifti data 
# *********************************************************************************

import os
import cv2
import numpy as np
import os.path
import pydicom
import scipy
import torch
import operator
from dcmrtstruct2nii import dcmrtstruct2nii, list_rt_structs
import shutil
import pandas as pd
import xlrd
import pydicom
from datetime import datetime, timedelta
from joblib import Parallel, delayed
from torch.utils.data import DataLoader, Dataset
from collections import Counter 
from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy, f1_score, confusion_matrix, precision, recall, specificity, cohen_kappa, fbeta_score
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Modules 
from commonConfig import commonConfigClass
from ioDataMethods import ioDataMethodsClass
conf = commonConfigClass()      # Init config class
ioData = ioDataMethodsClass()   # Functions for reading an processing data 


class niiTestDataset(Dataset):
    """
    Define class for the test data dataset 
    Adapted from training, but removed label dependence for inference purpose 
    """
    def __init__(self, data_paths):
        self.data_paths = data_paths
    def __getitem__(self, index):
        # Set path for index
        dataFilePath = self.data_paths[index]
        # Read image data from file    
        img = ioData.readDataToDataloader(dataFilePath)
        # Return data     
        return img, dataFilePath
    def __len__(self):
        return len(self.data_paths)


class inferenceDataMethodsClass:
    """
    Class describing functions needed for inference of data
    """

    def __init__ (self):
        """
        Init function
        """
        pass


    def mergeDicts(self, dict_list):
        """
        Merge multiple dictionaries of form {'key': value} into one dictionary of form {'key': [value1, value2, ...]}

        Arg:
            dict_list (list): list of dictionaries to merge

        Returns:
            new_dict: Merged dictionary
        """
        assert isinstance(dict_list, list)
        new_dict = {}
        # For every dictinary in the list
        for d in dict_list:
            # For every key in the dictionary
            for d_key in d:
                # If the key is not in the new dictionary add it
                if d_key not in new_dict:
                    new_dict[d_key] = []
                # Add the value to the key
                new_dict[d_key].append(d[d_key])
        assert isinstance(new_dict, dict), "Merged dictionary is not a dictionary"
        return new_dict


    def majorityVote(self, voteList): 
        """
        Get the majority vote, i.e the most frequent value in the input list. 

        Arg:
            voteList (list): List of values to vote on

        Returns:
            mostCommon: Majority vote
            freq: Frequency of the majority vote
            modelsAgreeing: The models agreeing on the majority vote

        output: value, frequency and models (positions) in ListIn that agree
        The positions will correspond to the models assessed
        """
        assert isinstance(voteList, list), "Input is not a list"
        occurenceCount = Counter(voteList) 
        mostCommon = occurenceCount.most_common(1)[0][0] 
        freq = occurenceCount.most_common(1)[0][1] 
        modelsAgreeing = [i for i,x in enumerate(voteList) if x==mostCommon]
        # If the "most common" frequency was equal to only 1 time
        if occurenceCount.most_common(1)[0][1] == 1:
            mostCommonComment = "Maximum occurence frequency was only 1 time for each label, none label selected for majority."
            print(mostCommonComment)
            # Exit loop
            freq = 1
            mostCommon = float("NaN")
            modelsAgreeing = float("NaN")
        # Return data 
        return mostCommon, freq, modelsAgreeing
        
    
    def createMajorityVoteDict(self, dictIn, nrModels): 
        """
        Calculate majority vote for each key in the input dictionary.
        Also add vote frequency and models agreeing to the dictionary.

        Arg:
            dictIn (dict): Dictionary to calculate majority vote statistics for
            nrModels (int): Number of models to calculate majority vote for

        Returns:
            dictOut: Dictionary with majority vote statistics
        """
        assert isinstance(dictIn, dict), "Input is not a dictionary"
        # Init a new output dictionary
        dictOut = {}
        # For each key in the input dictionary determine majority vote
        for key in dictIn.keys():
            assert len(dictIn[key]) == nrModels, "Input data for each key does not have the same length as the number of models" 
            # Get the majority vote for the data from each key 
            majVote, freq, modelsAgreeing = self.majorityVote(dictIn[key])
            # Add statistics to a new dictionary 
            keyData = {key: {"majVote": int(majVote), "freq": int(freq), "modelsAgreeing": modelsAgreeing}}
            dictOut.update(keyData)
        assert len(dictIn) == len(dictOut), "Input and output dictionary have different length"
        # Return dictionary
        return dictOut

        
    def write2log(self, logFilePath, logMessage):
        """
        # Write to log file, insert new line and close file
        Arg: 
            logFilePath (str): Path to log file
            logMessage (str): Message to write to log file

        Returns: 
            None, output to file 
        """
        outFileObjectErrors = open(logFilePath, "a")
        outFileObjectErrors.write(logMessage)
        outFileObjectErrors.write("\n")
        outFileObjectErrors.close()


    def calcAndPrintAccuracyMetrics(self, predictions, targets, num_classes, resultMetricFilePath, computeTime): 
        """
        Calculate accuracy metrics for a given set of predictions and targets. 
        Uses torch metrics were input must be tensor. 
        Logs values CSV output file

        Arg:
            predictions (list): List of predictions
            targets (list): List of targets

        Returns:
            Prints accuracy metrics
        """
        assert isinstance(predictions, torch.Tensor), "Predictions is not a torch tensor"
        assert isinstance(targets, torch.Tensor), "Targets is not a torch tensor"
        assert len(predictions) == len(targets), "Predictions and Targets lists have different length"
        # Calculate metrics and convert to numpy array on CPU
        acc_micro = accuracy(predictions, targets, average='micro', num_classes=num_classes).cpu().numpy()
        acc_macro = accuracy(predictions, targets, average='macro', num_classes=num_classes).cpu().numpy()
        acc_perClass = accuracy(predictions, targets, average=None, num_classes=num_classes).cpu().numpy()
        f1_micro = f1_score(predictions, targets, average= 'micro').cpu().numpy()
        f1_macro = f1_score(predictions, targets, average='macro', num_classes=num_classes).cpu().numpy()
        f1_perClass = f1_score(predictions, targets, average=None, num_classes=num_classes).cpu().numpy()
        precision_micro = precision(predictions, targets, average='micro', num_classes=num_classes).cpu().numpy()
        precision_macro = precision(predictions, targets, average='macro', num_classes=num_classes).cpu().numpy()
        precision_perClass = precision(predictions, targets, average=None, num_classes=num_classes).cpu().numpy()
        recall_micro = recall(predictions, targets, average='micro', num_classes=num_classes).cpu().numpy()
        recall_macro = recall(predictions, targets, average='macro', num_classes=num_classes).cpu().numpy()
        recall_perClass = recall(predictions, targets, average=None, num_classes=num_classes).cpu().numpy()
        specificity_micro = specificity(predictions, targets, average='micro', num_classes=num_classes).cpu().numpy()
        specificity_macro = specificity(predictions, targets, average='macro', num_classes=num_classes).cpu().numpy()
        specificity_perClass = specificity(predictions, targets, average=None, num_classes=num_classes).cpu().numpy()
        kappa = cohen_kappa(predictions, targets, weights=None, num_classes=num_classes).cpu().numpy()
        # Print metrics
        print("Accuracy (micro): ", acc_micro)
        print("Accuracy (macro): ", acc_macro)
        print("Accuracy (per class): ", acc_perClass)
        print("f1 (micro): ", f1_micro)
        print("f1 (macro): ", f1_macro)
        print("f1 (per class): ", f1_perClass)
        print("Precision (micro): ", precision_micro)
        print("Precision (macro): ", precision_macro)
        print("Precision (per class): ", precision_perClass)
        print("Recall (micro): ", recall_micro)
        print("Recall (macro): ", recall_macro)
        print("Recall (per class): ", recall_perClass)
        print("Specificity (micro): ", specificity_micro)
        print("Specificity (macro): ", specificity_macro)
        print("Specificity (per class): ", specificity_perClass)
        print("Kappa: ", kappa)
        # Write to log file 
        self.write2log(resultMetricFilePath, "Accuracy (micro): " + "\t" + str(acc_micro))
        self.write2log(resultMetricFilePath, "Accuracy (macro): " + "\t" + str(acc_macro))
        self.write2log(resultMetricFilePath, "Accuracy (per class): " + "\t" + str(acc_perClass))
        self.write2log(resultMetricFilePath, "f1 (micro): " + "\t" + str(f1_micro))
        self.write2log(resultMetricFilePath, "f1 (macro): " + "\t" + str(f1_macro))
        self.write2log(resultMetricFilePath, "f1 (per class): " + "\t" + str(f1_perClass))
        self.write2log(resultMetricFilePath, "Precision (micro): " + "\t" + str(precision_micro))
        self.write2log(resultMetricFilePath, "Precision (macro): " + "\t" + str(precision_macro))
        self.write2log(resultMetricFilePath, "Precision (per class): " + "\t" + str(precision_perClass))
        self.write2log(resultMetricFilePath, "Recall (micro): " + "\t" + str(recall_micro))
        self.write2log(resultMetricFilePath, "Recall (macro): " + "\t" + str(recall_macro))
        self.write2log(resultMetricFilePath, "Recall (per class): " + "\t" + str(recall_perClass))
        self.write2log(resultMetricFilePath, "Specificity (micro): " + "\t" + str(specificity_micro))
        self.write2log(resultMetricFilePath, "Specificity (macro): " + "\t" + str(specificity_macro))
        self.write2log(resultMetricFilePath, "Specificity (per class): " + "\t" + str(specificity_perClass))
        self.write2log(resultMetricFilePath, "Kappa: " + "\t" + str(kappa))
        # Write compute time to log file 
        self.write2log(resultMetricFilePath, "Compute time: " + "\t" + str(computeTime))


    def copyClassifiedData(self, resultDictAllMajVoteSorted, dataInputPathInf, TaskNumber, versionIter):
        """
        Copy classified data to a new folder with subfolders containing the label names and the data. 

        Arg:
            resultDictAllMajVoteSorted (dict): Dictionary containing the majority vote sorted data
            dataInputPathInf (str): Path to the input data
            TaskNumber (str): Task number
            versionIter (str): Version number of model

        Returns:
            None
            
        """
        # Copy the classified Nifti structure files to new folders with naming according to the determined majority vote.
        # Also copy the CT volume file to the new folder and rename it to the patient name. 
        # Create target folder for data copy
        sortedDataTargetFolder = os.path.join(conf.data.dataOutputPathInf, TaskNumber, conf.base.dataOutputStructureDirSorted, 'versionIter' + str(versionIter))
        # Loop through resultDictAllMajVote for every key and do the following:
        print('Copying classified data to new folder...')
        for key in resultDictAllMajVoteSorted:
            # Get the majority vote for the key 
            key_majVote = resultDictAllMajVoteSorted[key]['majVote']
            # If majority vote is for 'other' class, .i.e. the last class, skip it because it is not of interest. 
            if key_majVote == len(conf.data.classStructures): 
                continue
            # Get the patient name and the structure name for the key.
            patientName = key.split('_')[0]
            structureName = key.split(patientName)[1].replace('.npz', '') #Remove file suffix .npz
            # Create the Nifti file name for the original structure file
            niiStructureFileName = 'mask' + structureName + conf.data.fileSuffix
            # Create the source path for the Nifti structure file 
            niiStructureFilePath = os.path.join(dataInputPathInf, patientName, niiStructureFileName)
            # Create the soirce path for the CT image volume 
            ctVolumeFilePath = os.path.join(dataInputPathInf, patientName, conf.data.CTImageFileName)
            # Get label name from label index
            labelName = conf.data.classStructures[key_majVote]
            # Create the target copy path for the Nifti structure file
            niiStructureFilePath_copy = os.path.join(sortedDataTargetFolder, labelName, patientName + '_' + niiStructureFileName)
            # Create the target copy path for the corresponding CT volume file
            ctVolumeFilePath_copy = os.path.join(sortedDataTargetFolder, 'CTData', patientName + '_' + conf.data.CTImageFileName)
            # Prints for debugging 
            #print(' ')
            #print(niiStructureFilePath)
            #print(niiStructureFilePath_copy)
            #print(labelName) 
            #print(ctVolumeFilePath)
            #print(ctVolumeFilePath_copy)

            # Copy files to selected folders, make sure folders exist before copying 
            
            if not os.path.exists(sortedDataTargetFolder):
                os.makedirs(sortedDataTargetFolder, exist_ok = True)
            # Copy structure file if not already present
            if not os.path.exists(niiStructureFilePath_copy):
                    os.makedirs(os.path.dirname(niiStructureFilePath_copy), exist_ok = True)
                    shutil.copy(niiStructureFilePath, niiStructureFilePath_copy)
            # Copy CT volume file if not already present
            if not os.path.exists(ctVolumeFilePath_copy):
                    os.makedirs(os.path.dirname(ctVolumeFilePath_copy), exist_ok = True)
                    shutil.copy(ctVolumeFilePath, ctVolumeFilePath_copy)
        
        print('Copying done!')


    def getGTdata(self, keysOfInterest, excelFilePath):
        """
        Read grount truth data from xlsx sheet.
        Then extracts the values of interest and return GT vector
    
        Arg:
            keysOfInterest (list): List of keys of interest
            excelFilePath (str): Path to the excel file containing the ground truth data            

        Returns:
            GTvector (list): List of ground truth labels for the keys of interest
            
        """
        # Check existance of excel file
        if not os.path.exists(excelFilePath):
            print('Excel GT file not found!')
        # Read excel file   
        workbook = xlrd.open_workbook(excelFilePath)
        # Get first sheet
        sheet = workbook.sheet_by_index(0)
        # Get file name and GT columns
        col_a = sheet.col_values(0, 1)
        col_b = sheet.col_values(1, 1)
        # Create dictionary 
        GT_dict = {a : int(b) for a, b in zip(col_a, col_b)}
        # Extract only the values for the interesting file names (keys)
        GTvector = [GT_dict[key] for key in keysOfInterest]
        # Return GT vector
        return GTvector

                
    def calcAndPrintConfusionMatrix(self, y_pred, y_true, dataSetInfo, saveMetricsFilePath):
        """
        Calculate and print/save confusion matrix.
    
        Arg:
            y_pred (list): List of predicted labels
            y_true (list): List of true labels
            dataSetInfo (dict): Data set identifier, set from conf.base.dataSetInf

        Returns:
            None
        """
        # Define print settings, specific for each dataset
        # if dataSetInfo == 'infDataDemoProstateDataBigRawSorted':
        fontSizeDetermined = 16 
        bottomAdjust = 0.20
        topAdjust = 0.99
        # Print the report and get the data in a dictionary
        # Observe settings: if divided by zero and number of digits. 
        print(            classification_report(y_true, y_pred, zero_division=0, digits=4, output_dict=False))
        dictPerformance = classification_report(y_true, y_pred, zero_division=0, digits=4, output_dict=True)
        # Get pandas data frame from pandas and export to CSV
        df = pd.DataFrame(dictPerformance).transpose()
        df.to_csv(saveMetricsFilePath, float_format="%.4f")
        # From dictionary we can get the order of the generated numeric target labels. Some ites must be excluded though. 
        excludeThese = ['accuracy', 'macro avg', 'weighted avg']
        targetClasses = [key for key in dictPerformance.keys() if key not in excludeThese]
        # Add the 'other' class name to target name list if there are 1 more class than number of classStructures defined 
        # This is becuase the 'other' class is not defined in the config where classes are defined :/  
        if len(targetClasses) == len(conf.data.classStructures) + 1:
            allClassStructures = conf.data.classStructures
            # Add 'Other class'
            allClassStructures.append('Other')
        # Set target names 
        targetNames = [allClassStructures[int(key)] for key in targetClasses]
        # Set changed target names for bowel experiment
        if dataSetInfo == 'Dataset_pelvic_segmentation_Nifti_20211203_testData':
            targetNames = ['Devisetty', 'RTOG', 'Other'] 
        # Get confusion matrix (can be normalized, normalize='true')
        confMatrix = confusion_matrix(y_true, y_pred, normalize='true')
        # Define figure
        fig, ax = plt.subplots()
        plt.rcParams.update({'font.size': fontSizeDetermined})
        # Plot it 
        disp = ConfusionMatrixDisplay(confusion_matrix=confMatrix,
                                    display_labels=targetNames)
        dispFigure = disp.plot(include_values=True,
                        cmap='Oranges', 
                        ax=ax, xticks_rotation='vertical',
                        values_format=".2g") # 2 significant digits
        # Set axis label to bold font
        plt.xlabel('Predicted label', fontweight='bold')
        plt.ylabel('True label', fontweight='bold')
        # Remove current colorbar (to far away to the right )
        disp.im_.colorbar.remove()
        # Insert new colorbar
        fig.colorbar(dispFigure.im_, pad=0.005)
        # Set higher resolution for manual save later
        dispFigure.figure_.dpi = 150
        # Adjust image
        dispFigure.figure_.subplots_adjust(bottom=bottomAdjust) 
        dispFigure.figure_.subplots_adjust(top=topAdjust) 
        # Show image
        plt.show(dispFigure)
        
        # Do manual save by: 
        # Maximize window
        # Save in png format and change name manually 
        # Cut image manually in windows 10 image editor and auto enhance. 
        # Did not get the following to work
        # plt.savefig(dataset + '_' + inclusionOption + 'test.png', dpi=300, format='png')