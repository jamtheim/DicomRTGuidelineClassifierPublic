# *********************************************************************************
# Author: Christian Jamtheim Gustafsson, PhD, Medical Physcist Expert
# Skåne University Hospital, Lund, Sweden and Lund University, Lund, Sweden
# Description: Configuration file for the data pipeline. Can be customized for each task.
# *********************************************************************************

# Modules needed for this class
import os
import numpy as np
import multiprocessing

class commonConfigClass():
    """
    Class describing the common configuration used in the project.
    Defines what configuration settings are used.
    Folder names and output names of files are defined here. 
    base.workDir needs to be set according to setup.
    """

    def __init__ (self):
        """
        Init function
        """
        pass
    
    class baseConfig:
        """
        Empty class to define base configuration
        """
        pass

    class dataConfig:
        """
        Empty class to define data configuration
        """
        pass

    class organConfig:
        """
        Empty class to define prostate configuration
        """
        pass

    class modelConfig:
        """
        Empty class to define model configuration
        """
        pass

   
    # Init base configuration class
    base = baseConfig()
    #  Init data configuration class
    data = dataConfig()
    #  Init organ configuration class
    organ = organConfig()
    #  Init model configuration class
    model = modelConfig()

    # Choose task identification number
    # Use prefix for inference, same settings but used to name inference folders more clearly
    #data.TaskNumber = 'Task104_PerStructure' # Bowel dataset
    data.TaskNumber = 'inf_Task104_PerStructure' # Bowel dataset
    #data.TaskNumber = 'Task105_ProstateDataBigRawSorted' # ProstateDataBigRawSorted dataset
    #data.TaskNumber = 'inf_Task105_ProstateDataBigRawSorted' # ProstateDataBigRawSorted dataset
    

    # Bowel dataset per structure 
    if data.TaskNumber == 'Task104_PerStructure' or data.TaskNumber == 'inf_Task104_PerStructure':
        base.workDir = '/mnt/mdstore2/Christian/GuidelineClassifier'
        base.dataSet = 'Dataset_pelvic_segmentation_Nifti_20211203_trainingData'
        base.dataSetInf = 'Dataset_pelvic_segmentation_Nifti_20211203_testData' # 25 patient test data 
         # Grount truth xlsx file for test dataset
        base.dataSetInfGTfile = base.dataSetInf + '.xlsx'
        base.ProjectName = 'clfModel'
        # Define the possible BODY structure names in the CT image
        base.bodyStructureName =['BODY','External']
        # Data dir name 
        base.dataDir = 'data'
        # Define data input path
        data.dataInputPath = os.path.join(base.workDir, base.dataDir, base.dataSet)
        data.dataInputPathInf = os.path.join(base.workDir, base.dataDir, base.dataSetInf)
        data.dataInputPathInfGT = os.path.join(base.workDir, base.dataDir, base.dataSetInfGTfile)
        # Define data output path 
        base.dataOutputDir = 'classificationTrainingData'
        base.dataOutputStructureDir = 'StructureData'
        base.dataOutputStructureDirSorted = 'StructureDataSorted'
        base.dataOutputDirInf = 'classificationInferenceData'
        data.dataOutputPath = os.path.join(base.workDir, base.dataOutputDir)
        data.dataOutputPathInf = os.path.join(base.workDir, base.dataOutputDirInf)
        # QA directory
        data.dataOutputQADir = 'QA_classificationTrainingData'
        data.dataOutputQAPath = os.path.join(base.workDir, data.dataOutputQADir)
        data.classStructures = ['Z_BowelCavityMN', 'Z_BB_RTOGMN']
        model.QAflag = 1
        # File settings
        data.filePrefix = 'mask_'
        data.CTImageFileName = 'image.nii.gz'
        data.fileSuffix = '.nii.gz'
         # Set number of subjects to process
        data.nrPatients = 'all' # base.nrPatients = 50
        # Margin in pixels to add to the bounding box (x,y,z)
        model.margin =[0,0,0]  
        # Desired image resolution in mm for output training data 
        model.desiredPixelSpacing = (2,2,3)
        # Desired matrix after zero padding
        model.desiredImageMatrixSize = (184,280,96)
        # Set number of channels in the model
        model.channels = 2
        # Set base model architecture
        #model.baseArchitecture = 'SEResNext101'
        #model.baseArchitecture = 'SEResNet152'
        model.baseArchitecture = 'SENet154'
        # Batch size
        model.batch_size = 6
        # model.baseArchitecture = 'EfficientNetBN'
        # Model data dir
        model.modelDir = 'models'
        # Model log dir
        model.logDir = 'logs'
        # Model cross validation dir
        model.crossValidationDir = 'cv'
        # Flag for using structure name filtering after inference
        model.useInferenceBlacklist = True
        # Extra filtration of structure names after inference required to be counted as one of the selected classes 
        model.inferenceFileNameMustNotContain = ['tuning', 'HELP', 'X_', 'Y_', 'opt_', 'Dose', 'Match', 'Artefakter', 'Artifakter', 'artefakt', 'GTV', 'CTV', 'PTV', 'ring', 'bowelbag', 'abdomen', 'tarm', 'buk', 'peritoneum'] 
        # Define image augmentation
        model.useAugmentation = True
        # Define image augmentation probability 
        model.useAugmentationProb = 0.5
        # Model early stop patience 
        model.earlyStopPatience = 60
        # Define augmentation settings
        model.augmentationSettings = 'FlipRotateTranslate'
        # Select number of patients to use in the model training (can use 'all').
        # Cross validation splits these patients. 
        model.nrPatients = 'all'
        # Option to evaluate inference results
        model.evalInference = True
        # Count number of CPUs
        base.nrCPU = 48 # multiprocessing.cpu_count()-2 # Save two threads for main
        # Set GPUs to use
        base.GPUs = [0]
        # Set model iteration
        model.versionIter = 30

        
    # Large prostate dataset 
    if data.TaskNumber == 'Task105_ProstateDataBigRawSorted' or data.TaskNumber == 'inf_Task105_ProstateDataBigRawSorted':
        base.workDir = '/mnt/mdstore2/Christian/GuidelineClassifier'
        base.dataSet = 'Dataset_ProstateDataBigRawSorted_train_Nifti'
        # base.dataSetInf = 'infDataDemoProstateDataBigRawSorted' # Demo data for debugging
        base.dataSetInf = 'Dataset_ProstateDataBigRawSorted_test_Nifti' # Internal pelvis test set
        # base.dataSetInf = 'Umea_Nifti' # External pelvis test set
        # Grount truth xlsx file for test dataset
        base.dataSetInfGTfile = base.dataSetInf + '.xlsx'
        base.ProjectName = 'clfModelProstateDataBigRawSorted'
        # Define the possible BODY structure names in the CT image
        base.bodyStructureName =['BODY','External']
        # Data dir name 
        base.dataDir = 'data'
        # Define data input path
        data.dataInputPath = os.path.join(base.workDir, base.dataDir, base.dataSet)
        data.dataInputPathInf = os.path.join(base.workDir, base.dataDir, base.dataSetInf)
        data.dataInputPathInfGT = os.path.join(base.workDir, base.dataDir, base.dataSetInfGTfile)
        # Define data output path 
        base.dataOutputDir = 'classificationTrainingData'
        base.dataOutputStructureDir = 'StructureData'
        base.dataOutputStructureDirSorted = 'StructureDataSorted'
        base.dataOutputDirInf = 'classificationInferenceData'
        data.dataOutputPath = os.path.join(base.workDir, base.dataOutputDir)
        data.dataOutputPathInf = os.path.join(base.workDir, base.dataOutputDirInf)
        # QA directory
        data.dataOutputQADir = 'QA_classificationTrainingData'
        data.dataOutputQAPath = os.path.join(base.workDir, data.dataOutputQADir)
        data.classStructures = ['Bladder', 'Bladder_AI1', 'FemoralHead_L', 'Femur_Head_L_AI1', 'FemoralHead_R', 'Femur_Head_R_AI1', 'Rectum', 'Anorectum_AI1']
        model.QAflag = 1
        # File settings
        data.filePrefix = 'mask_'
        data.CTImageFileName = 'image.nii.gz'
        data.fileSuffix = '.nii.gz'
         # Set number of subjects to process
        data.nrPatients = 'all' # base.nrPatients = 50
        # Margin in pixels to add to the bounding box (x,y,z)
        model.margin =[0,0,0]  
        # Desired image resolution in mm for output training data 
        model.desiredPixelSpacing = (2,2,3)
        # Desired matrix after zero padding
        model.desiredImageMatrixSize = (200,328,96)
        # Set number of channels in the model
        model.channels = 2
        # Set base model architecture
        model.baseArchitecture = 'SENet154'
        #model.baseArchitecture = 'SEResNet152'
        # Batch size
        model.batch_size = 5
        # Model data dir
        model.modelDir = 'models'
        # Model log dir
        model.logDir = 'logs'
        # Model cross validation dir
        model.crossValidationDir = 'cv'
        # Flag for using structure name filtering after inference
        model.useInferenceBlacklist = True 
        # Extra filtration of structure names after inference required to be counted as one of the selected classes
        model.inferenceFileNameMustNotContain = ['tuning', 'HELP', 'X_', 'Y_', 'Z_', 'opt', 'Dose', 'Match', 'Artefakter', 'Artifakter', 'artefakt', 'GTV', 'CTV', 'PTV', 'ring', 'AnalCanal'] 
        # Define image augmentation
        model.useAugmentation = True
        # Define image augmentation probability 
        model.useAugmentationProb = 0.4
        # Model early stop patience 
        model.earlyStopPatience = 20
        # Define augmentation settings
        model.augmentationSettings = 'RotateTranslate' # Do not flip for femorals. 
        # Select number of patients to use in the model training  (can use 'all')
        # Cross validation splits these patients. 
        model.nrPatients = 'all'
        # Option to evaluate inference results
        model.evalInference = True
        # Count number of CPUs
        base.nrCPU = 60 # multiprocessing.cpu_count()-2 # Save two threads for main
        # Set GPUs to use
        base.GPUs = [0,1]
        # Set model iteration
        model.versionIter = 12
        
  