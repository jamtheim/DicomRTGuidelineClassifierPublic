# *********************************************************************************
# Author: Christian Jamtheim Gustafsson, PhD, Medical Physcist Expert
# Skåne University Hospital, Lund, Sweden and Lund University, Lund, Sweden
# Description: Class for converting data from DICOM to Nifti format
# *********************************************************************************

import os
import cv2
import numpy as np
import os.path
import pydicom
import scipy
from dcmrtstruct2nii import dcmrtstruct2nii, list_rt_structs
import shutil
import pydicom
from datetime import datetime, timedelta
from joblib import Parallel, delayed


from commonConfig import commonConfigClass
conf = commonConfigClass() 


class convertDataMethodsClass:
    """
    Class describing functions needed for converting DICOM data to Nifty data
    """

    def __init__ (self):
        """
        Init function
        """
        pass

    def someFunction(self):
        """
        Description of function 

        Args:
            
   
        Return:
            
        """

    def DicomToNifti(self, i_subject, subject, dataInBasePath, dataOutBasePath):
        """
        Convert subject DICOM CT and struct data to Nifty format

        Args:
            i_subject (int): The current subject number
            subject (str): The current subject name
            dataInBasePath (str): The base path to the DICOM dataset
            dataOutBasePath (str): The base path to the Nifti dataset
            
        Returns:
            Outputs data to directory 
            
        """
        # Assess input
        assert isinstance(i_subject, int), 'Input i_subject must be an integer'
        assert isinstance(subject, str), 'Input subject must be a string'
        assert isinstance(dataInBasePath, str), 'Input dataInBasePath must be a string'
        assert isinstance(dataOutBasePath, str), 'Input dataInBasePath must be a string'
        # Assert existing directories
        assert os.path.isdir(dataInBasePath), 'Input dataInBasePath must be a directory'
        os.makedirs(dataOutBasePath, exist_ok=True)
        assert os.path.isdir(dataOutBasePath), 'Input dataOutBasePath must be a directory'
        # Get the RT struct file and path 
        subjectFolderPath = os.path.join(dataInBasePath, subject)
        subjectStructFile = self.getRTStructFile(subjectFolderPath)
        subjectStructFilePath = os.path.join(subjectFolderPath, subjectStructFile)
        # Define subject output folder
        subjectOutFolderPath = os.path.join(dataOutBasePath, subject)
        os.makedirs(subjectOutFolderPath, exist_ok=True)
        # Get list of all structures present in the DICOM structure file 
        subjectStructList = list_rt_structs(subjectStructFilePath)
        # Count number of structures
        nrStructsinList = len(subjectStructList)
        # Convert the RT structs to Nifty format 
        # This is performed by targeting each individual structure at a time in a loop. 
        # This is slower but safer. 
        # In this way we can isolate exceptions to individual structures and not break 
        # the process of dcmrtstruc2nii which happens otherwise. This avoids modification of the 
        # dcmrtstruc2nii source code and allows us in retrospect to see if missing data was important or not.
        # Failed objects are due to the fact that the structures are not completed or simply empty in Eclipse. 
        for structNr, currStruct in enumerate(subjectStructList):
            try:
                # Extract the structure and convert to Nifty
                # We do not want convert_original_dicom=True for all structures as this will add a lot of compute time. 
                # Do this only for BODY as this structure is always present. It has nothing to do with the structure itself for enabling convert_original_dicom=True. 
                if currStruct in conf.base.bodyStructureName:
                    print(subject)
                    dcmrtstruct2nii(subjectStructFilePath, subjectFolderPath, subjectOutFolderPath, structures=currStruct, gzip=True, mask_background_value=0, mask_foreground_value=1, convert_original_dicom=True)
                else:
                    dcmrtstruct2nii(subjectStructFilePath, subjectFolderPath, subjectOutFolderPath, structures=currStruct, gzip=True, mask_background_value=0, mask_foreground_value=1, convert_original_dicom=False)
                
            except:
                print("Exception when extracting " + currStruct + ' for ' + subject )
        
        # Get total number of files outputted
        nrFiles = len(os.listdir(subjectOutFolderPath))
        # If number of output files and in the list differ
        # -1 becuase of the image file that is created by dcmrtstruct2nii
        if nrFiles -1 != nrStructsinList:
            # Throw message   
            print('Number of output files and in the list differ for patient ' + subject )
            print(str(int(nrStructsinList-(nrFiles -1))) + ' structures were not extracted')
            print(subjectStructList)
            print(os.listdir(subjectOutFolderPath))

        
    def getRTStructFile(self, path):
        """
        Search a given path for a RT structure DICOM file
        Inputs:
            path (str): Path to the DICOM file directory
        Returns:
            The RT file name
        """
        # Assert input
        assert isinstance(path, str), 'Input path must be a string'
        # Assert directory
        assert os.path.isdir(path), 'Input path must be a directory'
        # List files 
        files = os.listdir(path)
        # Get only the RS struct dicom file 
        structFile = [f for f in files if ".dcm" in f]
        structFile = [f for f in files if "RS" in f]
        # Check that there is only one 
        if len(structFile) == 0:
            raise Exception('No RT structure file could be located. Make sure the file is located in the specified folder...')
        assert len(structFile) == 1
        # Return data 
        return structFile[0]
        


    