# *********************************************************************************
# Author: Christian Jamtheim Gustafsson, PhD, Medical Physcist Expert
# Skåne University Hospital, Lund, Sweden and Lund University, Lund, Sweden
# Description: Create datset for classification. For each patient in the dataset
# a 4D array is generated where the 3D CT information and 3D structure information 
# exists. Data is resampled to desired resolution. QA output is also provided for 
# each structure.
# *********************************************************************************

# Load modules
from commonConfig import commonConfigClass
from ioDataMethods import ioDataMethodsClass

# Init needed class instances
conf = commonConfigClass()          # Init config class
ioData = ioDataMethodsClass()       # Functions for reading an processing data 

# Load external Python packages
import os
import numpy as np
import scipy
import csv
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
import shutil
from skimage.transform import resize


# Define the case processing loop to be called in the parallel processing
def preProcessSubject(i_subject, subject, dataInBasePath, dataOutBasePath, TaskNumber):
    """
    For each subject in the dataset read, preproces and write data.

    Inputs:
        i_subject (int): The current subject number
        subject (str): The current subject name
        dataInBasePath (str): The base path to the dataset
        TaskNumber (int): Defined task number to be processed
    """

    # Subject folder path 
    folderPath = os.path.join(dataInBasePath, subject)
    # Get all Nifti files in the subject folder path 
    niftiFiles = os.listdir(folderPath)
    # Make sure it is a nifti file, otherwise remove item
    niftiFiles = [file for file in niftiFiles if file.endswith(conf.data.fileSuffix)]
    # Remove CT image file 
    structFiles = [file for file in niftiFiles if conf.data.CTImageFileName not in file]
    # Identify BODY structure file name
    bodyStructFileUse = ioData.getBodyStructure(subject, folderPath, structFiles)
    # Read the body structure Nifti file to numpy
    np_body_orig, sitk_body_orig, pixelSpacing_body_orig = ioData.readNiftiFile(os.path.join(folderPath, bodyStructFileUse))
    # Read CT image
    # Load corresponding CT image file
    np_CT_orig, sitk_CT_orig, pixelSpacing_CT_orig = ioData.readNiftiFile(os.path.join(folderPath, conf.data.CTImageFileName))
    # Assert shapes of original CT and body structure 
    assert np_CT_orig.shape == np_body_orig.shape, 'Shape of original CT and body structure do not match'
    # Normalize CT data
    np_CT_orig = ioData.zScoreNorm(np_CT_orig, ignoreAir=True)
    # For resampling of data calculate spacing ratio
    pixelRatio = np.array(pixelSpacing_CT_orig) / np.array(conf.model.desiredPixelSpacing)
    # Calculate new shape needed 
    resampledShape = tuple(np.floor(np_CT_orig.shape * pixelRatio).astype(int))
    # Resample CT data and body structure 
    np_CT = ioData.resizeImageData(np_CT_orig, resampledShape, 'img')
    np_body = ioData.resizeImageData(np_body_orig, resampledShape, 'seg')
    # Assert shapes of resampled CT and body
    assert np_CT.shape == np_body.shape, 'Shape of resampled CT and body structure do not match'
    # Get bounding box mask for resampled body structure 
    bbMaskBody = ioData.getBoundingBoxFilled(np_body, 1, conf.model.margin) 
    # Remove table top from resampled CT data 
    np_CT = np_CT * np_body
    # Crop CT data to body bounding box
    np_CT = ioData.cropImageFromMask(np_CT, bbMaskBody)

    # Print cropped CT size if larger than
    #if np_CT.shape[0] > conf.model.desiredImageMatrixSize[0]:
    #    print('Cropped CT size: ', np_CT.shape)
    #if np_CT.shape[1] > conf.model.desiredImageMatrixSize[1]:
    #    print('Cropped CT size: ', np_CT.shape)
    #return

    # For each Nifti structure file process it and write the needed data 
    for file in structFiles:
        # Read the structure Nifti file to numpy
        np_struct_orig, sitk_struct_orig, pixelSpacing_struct_orig = ioData.readNiftiFile(os.path.join(folderPath, file))
        # If structure is empty ignore it 
        if np_struct_orig.sum() == 0: 
            # print('Emtpy structure file ' + file + ' found for subject ' + subject)
            continue # Leave iteration in loop
        # Resample the structure 
        np_struct = ioData.resizeImageData(np_struct_orig, resampledShape, 'seg')
        # Remove anything outside of the body contour for the structure (such as the table top)
        np_struct = np_struct * np_body
        # Create AddMap between body and structure
        AddMap = np_body/2 + np_struct/2
        # Assert max value of data
        assert AddMap.max() <= 1, 'Max value of AddMap is greater than 1'
        
        # Crop the structure and AddMap to the body bounding box
        np_struct = ioData.cropImageFromMask(np_struct, bbMaskBody)
        AddMap = ioData.cropImageFromMask(AddMap, bbMaskBody)
        # Assert shapes of the resampled, cropped data against the resampled cropped CT 
        assert np_struct.shape == np_CT.shape, 'Shape of resampled and cropped structure and resampled cropped CT do not match'
        assert AddMap.shape == np_CT.shape, 'Shape of resampled cropped AddMap and resampled croppedCT do not match'  
        # Check that the structure is not empty, 
        # can happen if only limited number of voxels existed before resampling or if it is a table structure outside of the body)
        if np_struct.sum() == 0: 
            # print('Emtpy structure file after resampling and cropping ' + file + ' found for subject ' + subject)
            continue # Leave iteration in loop
        
        # Get number of slices containing signal
        np_struct_nrSlicesUsed = ioData.getNumberOfUsedSlices(np_struct)
        # If structure has more than a certain number of slices ignore it (i.e. limited support for extra large structures)
        if np_struct_nrSlicesUsed > conf.model.desiredImageMatrixSize[2]:
            print('Structure file ' + file + ' has more than ' + str(conf.model.desiredImageMatrixSize[2]) + ' slices for subject ' + subject + ' and will be ignored')
            continue # Leave iteration in loop
        # Extract relevant information for training data and expand volume to desired number of slices.  
        # Structure is truncated to existing size with signal centered and then expanded on both slice sides.
        # CT is truncated to fill the desired volume with CT slices. The same goes for for AddMap. 
        # In this way no information in the CT and AddMap is wasted. Zero filling is then performed in all spatial directions. 
        np_struct_ztrunk, np_CT_ztrunk, AddMap_ztrunk = ioData.truncVolSliceToDesiredSize(np_struct, np_CT, AddMap, conf.model.desiredImageMatrixSize)
        # All volumes limited to structure extent
        # np_struct_ztrunk_obsolete, np_CT_ztrunk_obsolete, AddMap_ztrunk_obsolete = ioData.truncVolSliceToStruct(np_struct, np_CT, AddMap)
        # Assert shapes of the resampled, cropped, limited data against the resampled CT  
        assert np_struct_ztrunk.shape == np_CT_ztrunk.shape, 'Shape of resampled and cropped structure and resampled cropped CT do not match' 
        assert AddMap_ztrunk.shape == np_CT_ztrunk.shape, 'Shape of resampled cropped AddMap and resampled croppedCT do not match'

        # If matrix is larger than desiredImageMatrixSize it must be scaled down.  
        # This is because data must be a certain size when inputted to the neural network, which is pretrained during inference.
        # This pre-processing is also applied during inference.
        # From previous assertions we know that all matrixes have the same size so there is only need to check one, np_CT_ztrunk. 
        # If size is larger in any dimension 
        if np_CT_ztrunk.shape[0] > conf.model.desiredImageMatrixSize[0] or np_CT_ztrunk.shape[1] > conf.model.desiredImageMatrixSize[1]: # Slices are truncated above, no need to check
            assert np_CT_ztrunk.shape[2] == conf.model.desiredImageMatrixSize[2] # Double check number of slices should be trancated above. 
            # Calculate ratio of how much larger the matrix size is compared to desiredImageMatrixSize
            ratio_row = np_CT_ztrunk.shape[0] / conf.model.desiredImageMatrixSize[0]
            ratio_col = np_CT_ztrunk.shape[1] / conf.model.desiredImageMatrixSize[1]
            # Get maximum ratio 
            scale_ratio_max = max([ratio_row, ratio_col])
            # Add 1 % to the ratio so we make sure the final matrix fit within desiredImageMatrixSize
            scale_ratio_max = np.array(scale_ratio_max * 1.01)
            # Check limits of ratio so we dont down scale to much 
            assert scale_ratio_max > 1, 'Downscaling factor is smaller than 1' 
            # assert scale_ratio_max <= 1.15, 'Downscaling is larger than 15%' 
            # Rescale data with the same factor in all dimension s
            # Calculate the new matrix shape needed 
            resampledShapeToFit = tuple(np.floor(np_CT_ztrunk.shape / scale_ratio_max).astype(int))
            print('Structure file ' + file + ' for subject ' + subject + ' was subjected to downscaling of factor ' + str(scale_ratio_max))
            print(np_CT_ztrunk.shape)
            # Resample CT data and body structure 
            np_CT_ztrunk = ioData.resizeImageData(np_CT_ztrunk, resampledShapeToFit, 'img')
            np_struct_ztrunk = ioData.resizeImageData(np_struct_ztrunk, resampledShapeToFit, 'seg')
            AddMap_ztrunk = ioData.resizeImageData(AddMap_ztrunk, resampledShapeToFit, 'seg') # Nearest neighbour for AddMap
            print(np_CT_ztrunk.shape)
            # Assert that shapes are intact relative each other 
            assert np_struct_ztrunk.shape == np_CT_ztrunk.shape, 'Shape of resampled and cropped structure and resampled cropped CT do not match' 
            assert AddMap_ztrunk.shape == np_CT_ztrunk.shape, 'Shape of resampled cropped AddMap and resampled croppedCT do not match'
            # Assert that shapes now are smaller or equal to desiredImageMatrixSize
            assert np_CT_ztrunk.shape[0] <= conf.model.desiredImageMatrixSize[0] 
            assert np_CT_ztrunk.shape[1] <= conf.model.desiredImageMatrixSize[1]
            assert np_CT_ztrunk.shape[2] <= conf.model.desiredImageMatrixSize[2]

        # Assert shapes of the resampled, cropped, limited data against the resampled CT  
        assert np_struct_ztrunk.shape == np_CT_ztrunk.shape, 'Shape of resampled and cropped structure and resampled cropped CT do not match' 
        assert AddMap_ztrunk.shape == np_CT_ztrunk.shape, 'Shape of resampled cropped AddMap and resampled croppedCT do not match'
        # Pad data to desired image size.
        np_CT_ztrunk = ioData.padAroundImageCenter(np_CT_ztrunk, conf.model.desiredImageMatrixSize)
        np_struct_ztrunk = ioData.padAroundImageCenter(np_struct_ztrunk, conf.model.desiredImageMatrixSize)
        AddMap_ztrunk = ioData.padAroundImageCenter(AddMap_ztrunk, conf.model.desiredImageMatrixSize)
        # Check image shapes before writing data 
        assert np_CT_ztrunk.ndim == 3, 'The resampled, cropped, limited, padded CT has more than 3 dimensions'
        assert np_CT_ztrunk.shape == conf.model.desiredImageMatrixSize, 'The resampled, cropped, limited, padded, limited CT has not the desired shape'
        assert np_struct_ztrunk.shape == conf.model.desiredImageMatrixSize, 'The resampled, cropped, limited, padded, limited struct has not the desired shape'
        assert AddMap_ztrunk.shape == np_CT_ztrunk.shape, 'Shape of resampled, cropped, limited, padded AddMap and resampled, cropped, limited padded CT do not match'
        # Write out numpy training data 
        ioData.writeClassificationTrainingDataPerStructure(subject, file, np_CT_ztrunk, AddMap_ztrunk, dataOutBasePath, TaskNumber)
        # Write out one slice of PNG data for QA purposes
        if conf.model.QAflag: 
            ioData.writeClassificationQAimagesPerStructure(subject, file, np_CT_ztrunk, AddMap_ztrunk, conf.data.dataOutputQAPath, TaskNumber)
       
 
# Entry point for the script
if __name__ == "__main__":
    # Get current directory
    codePath = os.path.dirname(os.path.realpath(__file__))
    # Change directory to code path
    os.chdir(codePath)
    TaskNumber = conf.data.TaskNumber
    print('This is processing for ' + TaskNumber)
    # Get all subject folders 
    subjectFolders = os.listdir(conf.data.dataInputPath)
    # Make sure it is a folder, otherwise remove non-folder items
    subjectFolders = [folder for folder in subjectFolders if os.path.isdir(os.path.join(conf.data.dataInputPath, folder))]
    # Limit the number of patients to process
    subjectFolders = ioData.limitNrPatients(subjectFolders, conf.data.nrPatients)
    # For each subject in the dataset initiate preprocessing in parallel manner     
    Parallel(n_jobs=conf.base.nrCPU, verbose=10)(delayed(preProcessSubject)(i_subject, subject, conf.data.dataInputPath, conf.data.dataOutputPath, TaskNumber) for i_subject, subject in enumerate(subjectFolders))

