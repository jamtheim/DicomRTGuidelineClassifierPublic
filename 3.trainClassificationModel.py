# *********************************************************************************
# Author: Christian Jamtheim Gustafsson, PhD, Medical Physcist Expert
# Skåne University Hospital, Lund, Sweden and Lund University, Lund, Sweden
# Description: Train classification model
# *********************************************************************************

from cmath import pi
import os
import random
from sys import modules
import numpy as np
import glob
import time
from shutil import copyfile
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import torch
import itertools
from torchsummary import summary
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, callbacks
from torch import nn
from torch.utils.data import DataLoader, random_split, Dataset
from torchmetrics.functional import accuracy, f1_score, confusion_matrix, precision, recall, specificity, cohen_kappa, fbeta_score
import torchvision.transforms.functional as TF
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
# from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.strategies import DDPStrategy
# MONAI models and transforms 
from monai.networks.nets import SEResNet152, SEResNext101, EfficientNetBN, SENet154, SEResNext50
from monai.transforms import Compose, RandFlip, RandRotate, RandAffine
# Load modules
from ioDataMethods import ioDataMethodsClass
from commonConfig import commonConfigClass

# Init needed class instances
ioData = ioDataMethodsClass()           # Read data from Nifti files
conf = commonConfigClass()              # Init config class

# Functions
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class niiDataModule(LightningDataModule):
    def __init__(self, trainDataPaths, trainDataTargets, valDataPaths, valDataTargets, batch_size, seed, useAugmentation):
        #Define required parameters here
        super().__init__()
        self.trainDataPaths = trainDataPaths
        self.trainDataTargets = trainDataTargets
        self.valDataPaths = valDataPaths
        self.valDataTargets = valDataTargets
        self.batch_size = batch_size
        self.seed = seed
        self.useAugmentation = useAugmentation
        
    def prepare_data(self):
        # Define steps that should be done to prepare the data
        print('Preparing data...')       

    def setup(self, stage=None):
        # Define steps that should be done on 
        # all GPUs, like splitting data, applying
        # transform etc. Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            # Define transforms which handles all channels the same way
            if self.useAugmentation:

                # FlipRotateTranslate combo
                if conf.model.augmentationSettings == 'FlipRotateTranslate':
                    self.trainDataTransform = Compose([
                        # LR flip
                        RandFlip(prob=conf.model.useAugmentationProb, spatial_axis=1), 
                        # Rotate both channels with nearest neighbor (nearest should be OK for CT)
                        RandRotate((0,0),(0,0),(-5*pi/180,5*pi/180), 
                                    prob=conf.model.useAugmentationProb, keep_size=True, mode = 'nearest', dtype=np.float32), 
                        # Random affine transform for translation only
                        RandAffine(translate_range = ((-10,10),(-10,10),(0,0)),
                                    prob=conf.model.useAugmentationProb, mode = 'nearest', padding_mode="zeros")
                        ])
                
                # RotateTranslate combo
                if conf.model.augmentationSettings == 'RotateTranslate':
                    self.trainDataTransform = Compose([
                    # Rotate both channels with nearest neighbor (nearest should be OK for CT)
                    RandRotate((0,0),(0,0),(-5*pi/180,5*pi/180), 
                                prob=conf.model.useAugmentationProb, keep_size=True, mode = 'nearest', dtype=np.float32), 
                    # Random affine transform for translation only
                    RandAffine(translate_range = ((-10,10),(-10,10),(0,0)),
                                prob=conf.model.useAugmentationProb, mode = 'nearest', padding_mode="zeros")
                    ])   

                # Set deterministic transforms
                self.trainDataTransform.set_random_state(self.seed)
            else: 
                self.trainDataTransform = None
            # Set validation augmentation to None
            self.valDataTransform = None
            # Training data
            self.trainData = niiDataset(self.trainDataPaths, self.trainDataTargets, self.trainDataTransform)
            # Validation data
            self.valData = niiDataset(self.valDataPaths, self.valDataTargets, self.valDataTransform)
           
    def train_dataloader(self):
        return DataLoader(self.trainData, batch_size=self.batch_size, num_workers=10, shuffle=False, persistent_workers=True, pin_memory=True)
    def val_dataloader(self):
        return DataLoader(self.valData, batch_size=self.batch_size, num_workers=10, shuffle=False, persistent_workers=True, pin_memory=True)
 
  
class niiDataset(Dataset):
    def __init__(self, data_paths, data_targets, transform):
        self.data_paths = data_paths
        self.transform = transform
        self.targets = data_targets
        # QA check of data and target 
        assert len(self.data_paths) == len(self.targets)

        
    def __getitem__(self, index):
        # Set path and target for index
        dataFilePath = self.data_paths[index]
        target = self.targets[index]
        
        # Read image data from file    
        img = ioData.readDataToDataloader(dataFilePath)

        # Apply trasfmations (perform on CPU)
        if self.transform is not None:
            img_aug = self.transform(img)
            # Assert same size after transformation
            assert img_aug.shape == img.shape
            # Plot it with image and segmentation, nice to assess image augmentation
            #f, axarr = plt.subplots(2,2)
            #axarr[0,0].imshow(img[0,:,:,40])
            #axarr[0,1].imshow(img_aug[0,:,:,40])
            #axarr[1,0].imshow(img[1,:,:,40])
            #axarr[1,1].imshow(img_aug[1,:,:,40])
            # Override img variable and remove it from memory
            img = img_aug
            del img_aug

        # Return data and target    
        return img, target, dataFilePath

    def __len__(self):
        return len(self.data_paths)



class LitModel(LightningModule):
    def __init__(self, channels, width, height, depth, num_classes, learning_rate, workDir, versionIter, modelName, classWeights, targetNames, crossValCounter):
        # Tip: For things to work out good in multi GPU environment use self. for the variables defined here
        super().__init__()
        # We take ininput dimensions as parameters and use those to dynamically build model.
        self.channels = channels
        self.width = width
        self.height = height
        self.depth = depth
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.workDir = workDir
        self.versionIter = versionIter
        self.classWeights = torch.Tensor(classWeights)
        self.targetNames = targetNames
        self.crossValCounter = crossValCounter
        
        # Load neural net model
        if modelName==EfficientNetBN:
            self.neuralNet = EfficientNetBN('efficientnet-b4', spatial_dims=3, in_channels=channels, pretrained=False, num_classes=self.num_classes)
            self.neuralNet.set_swish(memory_efficient=True) # Memory tweak, important for GPU RAM footprint
            print('Using EfficientNet-B4')
        else:
            # Define neural net 
            self.neuralNet = modelName(spatial_dims=3, in_channels=channels, pretrained=False, num_classes=self.num_classes)
            print('Using {}'.format(modelName))
             

    def forward(self, x):
        output = self.neuralNet(x)
        return output 

        
    def training_step(self, batch, batch_idx):
        x, y, dataPaths = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, self.classWeights.type_as(x)) # Type sets to same device as x
        pred_train = torch.argmax(logits, dim=1)
        train_acc_weighted = accuracy(pred_train, y, average='weighted', num_classes=self.num_classes)
        train_acc_micro = accuracy(pred_train, y, average='micro')
        train_acc_macro = accuracy(pred_train, y, average='macro', num_classes=self.num_classes)
        train_acc_perClass = accuracy(pred_train, y, average=None, num_classes=self.num_classes)
        train_f1_micro = f1_score(pred_train, y, average='micro')
        train_f1_macro = f1_score(pred_train, y, average='macro', num_classes=self.num_classes)
        train_f1_perClass = f1_score(pred_train, y, average=None, num_classes=self.num_classes)
        #self.log("train_loss_weighted", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
        self.log("train_loss_weighted", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
        #self.log("train_acc_weighted", acc_train_weighted, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
        #self.log("train_acc_micro", acc_train_micro, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
        #self.log("train_f1_micro", f1_train_micro, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
        self.log("train_acc_macro", train_acc_macro, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
        self.log("train_f1_macro", train_f1_macro, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
        # Accuracy for each class
        #for i in range(self.num_classes):
        #    self.log("train_acc_class " + str(i), acc_train_perClass[i], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
        #    self.log("train_f1_class " + str(i), f1_train_perClass[i], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
        #print(batch_idx)
        #print(x[:].sum())
        return loss

 
    def validation_step(self, batch, batch_idx):
        x, y, dataPaths = batch
        logits = self(x)
        val_loss = F.cross_entropy(logits, y)  #val_loss = F.cross_entropy(logits, y, self.classWeights.type_as(x)) 
        pred_val = torch.argmax(logits, dim=1)
        val_acc_weighted = accuracy(pred_val, y, average='weighted', num_classes=self.num_classes)
        val_acc_micro = accuracy(pred_val, y, average='micro')
        val_acc_macro = accuracy(pred_val, y, average='macro', num_classes=self.num_classes)
        val_acc_perClass = accuracy(pred_val, y, average=None, num_classes=self.num_classes)
        val_f1_micro = f1_score(pred_val, y, average= 'micro')
        val_f1_macro = f1_score(pred_val, y, average= 'macro', num_classes=self.num_classes)
        val_f1_perClass = f1_score(pred_val, y, average=None, num_classes=self.num_classes)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
        #self.log("val_acc_weighted", val_acc_weighted, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
        self.log("val_acc_micro", val_acc_micro, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
        self.log("val_acc_macro", val_acc_macro, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
        #self.log("val_f1_micro", val_f1_micro, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
        #self.log("val_f1_macro", val_f1_macro, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
        # Accuracy for each class
        #for i in range(self.num_classes):
        #    self.log("val_acc_class " + str(i), val_acc_perClass[i], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
        #    self.log("val_f1_class " + str(i), val_f1_perClass[i], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
        # For calculation over the whole validation dataset return prediction and target values (validation_epoch_end)
        return logits, pred_val, y, dataPaths

    
    def validation_epoch_end(self, validation_step_outputs):
        # Executes at the end of each validation epoch
        # Get logits, prediction and target values
        # Extract values but first create empty lists 
        logits_val_all = []
        pred_val_all = [] 
        y_all = [] 
        dataPaths_all = []
        # Loop through each batch in the collected validation data
        for batch in range(len(validation_step_outputs)):
            # Extract the logits, predictions and target values for each batch
            logits_val_all.append(validation_step_outputs[batch][0])
            pred_val_all.append(validation_step_outputs[batch][1]) # Predictions, see order of return in validation_step
            y_all.append(validation_step_outputs[batch][2]) # Targets, see order of return in validation_step
            dataPaths_all.append(validation_step_outputs[batch][3]) # Data paths, see order of return in validation_step
        # Concatenate logits, predictions and targets into one tensor
        logits_val_all = torch.cat(logits_val_all, dim=0)
        pred_val_all = torch.cat(pred_val_all, dim=0)
        y_all = torch.cat(y_all, dim=0)
        dataPaths_all = list(itertools.chain.from_iterable(dataPaths_all))
        # Assert same size and print size
        assert logits_val_all.size()[0] == y_all.size()[0]
        assert pred_val_all.size() == y_all.size()
        assert(len(dataPaths_all) == logits_val_all.size()[0])
        # Check which objects failed
        failedObjects, failedObjectsPred, failedObjectsPredName = ioData.checkFailedObjects(torch.Tensor.cpu(pred_val_all).tolist(), torch.Tensor.cpu(y_all).tolist(), dataPaths_all, self.targetNames)
        # Log number if failed objects
        self.log("nrFailedObjects", len(failedObjects), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
        # Write out failed objects to file        
        ioData.writeFailedObjects(self.current_epoch, failedObjects, failedObjectsPredName, 
            os.path.join(self.workDir, conf.model.logDir, conf.base.ProjectName, str(self.versionIter), conf.model.crossValidationDir + str(self.crossValCounter), 'failedObjects.csv'))
        
        # Log the validation size 
        self.log("valAll_size", len(pred_val_all), prog_bar=True, logger=True, sync_dist=False)
        # Calculate metrics for aggregated validation set 
        # Loss
        valAll_loss = F.cross_entropy(logits_val_all, y_all)
        # Other metrics 
        valAll_acc_micro = accuracy(pred_val_all, y_all, average='micro', num_classes=self.num_classes)
        valAll_acc_macro = accuracy(pred_val_all, y_all, average='macro', num_classes=self.num_classes)
        valAll_acc_perClass = accuracy(pred_val_all, y_all, average=None, num_classes=self.num_classes)
        
        valAll_f1_micro = f1_score(pred_val_all, y_all, average= 'micro')
        valAll_f1_macro = f1_score(pred_val_all, y_all, average='macro', num_classes=self.num_classes)
        valAll_f1_perClass = f1_score(pred_val_all, y_all, average=None, num_classes=self.num_classes)
        
        valAll_precision_micro = precision(pred_val_all, y_all, average='micro', num_classes=self.num_classes)
        valAll_precision_macro = precision(pred_val_all, y_all, average='macro', num_classes=self.num_classes)
        valAll_precision_perClass = precision(pred_val_all, y_all, average=None, num_classes=self.num_classes)

        valAll_recall_micro = recall(pred_val_all, y_all, average='micro', num_classes=self.num_classes)
        valAll_recall_macro = recall(pred_val_all, y_all, average='macro', num_classes=self.num_classes)
        valAll_recall_perClass = recall(pred_val_all, y_all, average=None, num_classes=self.num_classes)

        valAll_specificity_micro = specificity(pred_val_all, y_all, average='micro', num_classes=self.num_classes)
        valAll_specificity_macro = specificity(pred_val_all, y_all, average='macro', num_classes=self.num_classes)
        valAll_specificity_perClass = specificity(pred_val_all, y_all, average=None, num_classes=self.num_classes)
        
        kappa = cohen_kappa(pred_val_all, y_all, weights=None, num_classes=self.num_classes)

        # Synthetic metrics
        if conf.data.TaskNumber == 'Task105_ProstateDataBigRawSorted' or conf.data.TaskNumber == 'Task106_ProstateDataBigSmallSorted':
            valAll_precision_custom = valAll_precision_perClass[0] + valAll_precision_perClass[1] + valAll_precision_perClass[2] + valAll_precision_perClass[3] + valAll_precision_perClass[4] + valAll_precision_perClass[5] + valAll_precision_perClass[6] + valAll_precision_perClass[7] 

        if conf.data.TaskNumber == 'Task104_PerStructure':
            valAll_precision_custom = valAll_precision_perClass[0] + valAll_precision_perClass[1]
        
        # Log metrics
        self.log("valAll_loss", valAll_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
        self.log("valAll_acc_micro", valAll_acc_micro, prog_bar=True, logger=True, sync_dist=False)
        self.log("valAll_acc_macro", valAll_acc_macro, prog_bar=True, logger=True, sync_dist=False)
        self.log("valAll_f1_macro", valAll_f1_macro, prog_bar=True, logger=True, sync_dist=False)
        self.log("valAll_precision_macro", valAll_precision_macro, prog_bar=True, logger=True, sync_dist=False)
        self.log("valAll_recall_macro", valAll_recall_macro, prog_bar=True, logger=True, sync_dist=False)
        self.log("valAll_specificity_macro", valAll_specificity_macro, prog_bar=True, logger=True, sync_dist=False)
        self.log("valAll_kappa", kappa, prog_bar=True, logger=True, sync_dist=False)
        self.log("valAll_precision_custom", valAll_precision_custom, prog_bar=True, logger=True, sync_dist=False)
        # Print metrics for each class
        for i in range(self.num_classes):
            self.log("valAll_acc_class " + str(i), valAll_acc_perClass[i], prog_bar=True, logger=True, sync_dist=False)
            self.log("valAll_f1_class " + str(i), valAll_f1_perClass[i], prog_bar=True, logger=True, sync_dist=False)
            self.log("valAll_precision_class " + str(i), valAll_precision_perClass[i], prog_bar=True, logger=True, sync_dist=False)
            self.log("valAll_recall_class " + str(i), valAll_recall_perClass[i], prog_bar=True, logger=True, sync_dist=False)
            self.log("valAll_specificity_class " + str(i), valAll_specificity_perClass[i], prog_bar=True, logger=True, sync_dist=False)
        # Print line break
        #print("\n")
        

    def test_step(self, batch, batch_idx):
        """
        Determines behavior when in test mode.
        Extracts information on each batch of the test set and return it. 
        Also applies secondary control from the inference result. 
        """
        x, dataPaths = batch
        logits = self(x)
        pred_val = torch.argmax(logits, dim=1)
        # Move to CPU 
        pred_val = pred_val.cpu().numpy()
        # Get only file name without full path 
        fileNames = [os.path.basename(x) for x in dataPaths]
        # Do secondary control on the batch depending on the contents of the structure fileName 
        if conf.model.useInferenceBlacklist == True: 
            # For every file in the batch 
            for fileIndex, fileName in enumerate(fileNames):
                # If any of the words defined in inferenceFileNameMustNotContain are present in the fileName
                # set the class to the trash class
                if any(keyWord.lower() in fileName.lower() for keyWord in conf.model.inferenceFileNameMustNotContain):
                    pred_val[fileIndex] = self.num_classes-1 # Class definitions starts at 0, so set to the last class ("other")
                    # print information on change or not
                    print("File " + fileName + " was by secondary filter predicted to class " + str(self.num_classes-1))
                else:
                    print("File " + fileName + " was by neural net predicted to class " + str(pred_val[fileIndex]))
        # Return the batch results 
        return fileNames, pred_val
        

    
    def test_epoch_end(self, test_step_outputs): 
        """
        Collects all outputs from each test_step.
        This will contain information on the whole test set 
        """
        # Extract values but first create empty lists 
        fileNames_all = []
        pred_val_all = [] 
        # Loop through each batch in the collected validation data
        for batch in range(len(test_step_outputs)):
            # Extract the file names and predictions for each batch
            fileNames_all.append(test_step_outputs[batch][0]) # File names, see order of return in test_step
            pred_val_all.append(test_step_outputs[batch][1]) # Predictions, see order of return in test_step
        # Concatenate fileNames and predictions into one array
        fileNames_all = list(itertools.chain.from_iterable(fileNames_all))
        pred_val_all = np.concatenate(pred_val_all).tolist()
        # Assert list type
        assert type(fileNames_all) == list
        assert type(pred_val_all) == list
        # Assert same size and list
        assert len(pred_val_all) == len(fileNames_all)
        # For each fileName print
        for i in range(len(fileNames_all)):
            print(fileNames_all[i] + ': ' + str(pred_val_all[i]))
        # Create a dictionary with the file names and predictions and return it by log method 
        # This can be gathered as output of trainer.test 
        dictionary = {fileNames_all[i]: pred_val_all[i] for i in range(len(fileNames_all))}
        self.log_dict(dictionary)
        
     
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
            optimizer = optimizer, 
            mode='min', 
            factor=0.5,                                                  
            patience=15, 
            verbose=True,
            min_lr=1e-7,
            )
        lr_scheduler_config = {
        # REQUIRED: The scheduler instance
        "scheduler": lr_scheduler,
        # The unit of the scheduler's step size, could also be 'step'.
        # 'epoch' updates the scheduler on epoch end whereas 'step'
        # updates it after a optimizer update.
        "interval": "epoch",
        # How many epochs/steps should pass between calls to
        # `scheduler.step()`. 1 corresponds to updating the learning
        # rate after every epoch/step.
        "frequency": 1,
        # Metric to to monitor for schedulers like `ReduceLROnPlateau`
        "monitor": "val_loss",
        # If set to `True`, will enforce that the value specified 'monitor'
        # is available when the scheduler is updated, thus stopping
        # training if not found. If set to `False`, it will only produce a warning
        "strict": True,
        # If using the `LearningRateMonitor` callback to monitor the
        # learning rate progress, this keyword can be used to specify
        # a custom logged name
        "name": None,
            }
        
        return [optimizer],[lr_scheduler_config] 


# -----Entry point for the script -----
if __name__ == '__main__':
    # Get working dir
    workDir = os.getcwd()
    # Set expeiment iteration
    versionIter = conf.model.versionIter
    # Set dimensions of input data
    channels = conf.model.channels
    height = conf.model.desiredImageMatrixSize[0]
    width = conf.model.desiredImageMatrixSize[1]
    depth = conf.model.desiredImageMatrixSize[2]
    # Model settings
    modelName = eval(conf.model.baseArchitecture) 
    learning_rate = 1e-4
    batch_size = conf.model.batch_size 
    # Cross validation
    num_folds = 5
    # Seed for random number generator
    seed = 42
    seed_everything(seed)
    # Data handling
    useFullDataBool = True # Filter data on structures or not (True = all, False = filtered)
    # Data augmentation 
    useAugmentation = conf.model.useAugmentation
    # Number epochs
    num_epochs = 10000
    # Set flag for final model training where all data is used 
    # Can be combined with num_folds for repeating full model training. 
    # However, unclear how to use this with lr decay. Stick with splits.
    useFinalTrainingAllDataBool = False 
    # Defina data directory and get full data paths 
    # data_dir = os.path.join(conf.data.dataOutputPath, 'Task104_PerStructure')
    data_dir = os.path.join(conf.data.dataOutputPath, conf.data.TaskNumber)
    dataPathsFull = glob.glob(os.path.join(data_dir, conf.base.dataOutputStructureDir, '*.npz'))
    # Structures of interest 
    classStructures = conf.data.classStructures
    # Create filtered data paths from class structure names 
    dataPathsFiltered = ioData.filterDataPaths(dataPathsFull, classStructures)
    # Set the what dataset to use in the training (full or filtered) and get number of classes 
    dataPathsUse, num_classes = ioData.setTrainingData(useFullDataBool, dataPathsFull, dataPathsFiltered, classStructures)
    # Create target naming from class structure names  
    targetNames = ioData.getTargetNaming(classStructures, useFullDataBool)
    # Get unique patients from the selected data paths 
    allPatients = ioData.getUniquePatients(dataPathsUse)
    # Select n number of patients for training and validation. conf.model.nrPatients can take value 'all'.
    allPatients = ioData.limitNrPatients(allPatients, conf.model.nrPatients)
    # Define folds
    kf = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
    # Init crossValCounter
    crossValCounter = 0
    # Copy this run file and config to log directory  
    runFileName = os.path.join(workDir, conf.model.logDir, conf.base.ProjectName, str(versionIter), 'runCode_trainClassification_verIter' + str(versionIter) + '.py')
    configFileName = os.path.join(workDir, conf.model.logDir, conf.base.ProjectName, str(versionIter), 'commonConfig_verIter' + str(versionIter) + '.py')
    os.makedirs(os.path.dirname(runFileName), exist_ok=True)
    copyfile(__file__, runFileName)
    copyfile(os.path.join(os.path.dirname(__file__), 'commonConfig.py'), configFileName)
        
    # Cross validation loop
    for train_index, val_index in kf.split(allPatients):
        os.makedirs(os.path.join(workDir, conf.model.logDir), exist_ok=True)
        # Define Tensor board logger with customization 
        tb_logger = TensorBoardLogger(save_dir=conf.model.logDir + "/", name=conf.base.ProjectName, version=str(versionIter), sub_dir=conf.model.crossValidationDir + str(crossValCounter))
        # CSV logger is limited in its abilities to handle logs for cross validation 
        csv_logger = CSVLogger(save_dir=conf.model.logDir + '/' +  conf.base.ProjectName + '/' + str(versionIter) + '/' + conf.model.crossValidationDir + str(crossValCounter), name='csv_logger', flush_logs_every_n_steps=100)
        # Assign training and validation patients
        trainPatients = [allPatients[i] for i in train_index]
        valPatients = [allPatients[i] for i in val_index]
        # If option to use whole dataset as model training (i.e useFinalTrainingAllDataBool) is selected do this: 
        if useFinalTrainingAllDataBool:
            trainPatients = trainPatients + valPatients
            valPatients = trainPatients # Set validation patients to training patients 
            # Get data paths for training and validation patients
        trainDataPaths, valDataPaths = ioData.getCrossValPathData(dataPathsUse, trainPatients, valPatients, useFinalTrainingAllDataBool)
        # Shuffle data order (this makes all classes apear in most training and validation batches)
        random.shuffle(trainDataPaths, random.seed(seed))
        random.shuffle(valDataPaths, random.seed(seed))
        # Print the cross validation number
        print('Cross validation number: ', crossValCounter)
        # Assign targets to the training and validation data 
        trainDataTargets = ioData.getTargetVector(trainDataPaths, classStructures)
        valDataTargets = ioData.getTargetVector(valDataPaths, classStructures)
        # Calculate class weights for the whole data set (does strictly not need to be in the loop in that case)
        classWeights = ioData.getClassWeights(trainDataTargets+valDataTargets, 'balanced')
        # Print information about the training data and class weights
        print('This is training for task {} '.format(conf.data.TaskNumber))
        print('Number of patients selected for training: ' + str(conf.model.nrPatients) )
        print('There are {} training data items'.format(len(trainDataPaths)))
        print('There are {} training targets'.format(len(trainDataTargets)))
        print('There are {} validation data items'.format(len(valDataPaths)))
        print('There are {} validation targets'.format(len(valDataTargets)))
        print('There are {} training targets for the other class '.format(trainDataTargets.count(len(classStructures)))) # Remember class counting starts at 0
        print('There are {} validation targets for the other class '.format(valDataTargets.count(len(classStructures)))) # Remember class counting starts at 0
        print('There are {} classes to train on '.format(num_classes))
        print('The class weights are: {}'.format(classWeights))
        print('Number of GPUs: {}'.format(len(conf.base.GPUs)))
        print('Image augmentation: {}'.format(useAugmentation))
        # Init DataModule
        dm = niiDataModule(trainDataPaths, trainDataTargets, valDataPaths, valDataTargets, batch_size, seed, useAugmentation)
        # Init model from datamodule's attributes
        model = LitModel(channels, width, height, depth, num_classes, learning_rate, workDir, versionIter, modelName, classWeights, targetNames, crossValCounter)
        # Init callbacks
        # progress_bar_callback = callbacks.RichProgressBar(refresh_rate_per_second=1, leave=True) # Migrated version
        progress_bar_callback = callbacks.RichProgressBar(refresh_rate=1, leave=True)
        early_stop_callback = callbacks.EarlyStopping(monitor="valAll_precision_custom", min_delta=0, patience=conf.model.earlyStopPatience, verbose=True, mode="max", strict=True, check_finite=False)
        lr_monitor_callback = callbacks.LearningRateMonitor(logging_interval='epoch')
        os.makedirs(os.path.join(workDir, conf.model.modelDir), exist_ok=True)
        model_save = callbacks.ModelCheckpoint(dirpath=conf.model.modelDir + '/' + conf.base.ProjectName + '/' + str(versionIter) + '/' + conf.model.crossValidationDir + str(crossValCounter), filename='{epoch}-{valAll_precision_custom:.2f}-{valAll_loss:.2f}-{valAll_acc_macro:.2f}', monitor='valAll_precision_custom', verbose=True, save_top_k=5, every_n_epochs=1, save_weights_only=True, mode='max') 
        
        # Init trainer
        trainer = Trainer(
            accelerator="gpu",
            gpus=conf.base.GPUs, #num_gpus,
            benchmark=True, 
            max_epochs=num_epochs,
            check_val_every_n_epoch=1,
            log_every_n_steps=20,
            callbacks=[progress_bar_callback, early_stop_callback, lr_monitor_callback, model_save],
            num_nodes=1, 
            strategy=DDPStrategy(find_unused_parameters=False), # This seems to cause hickups for cross validation, ignore for stability 
            precision=16, 
            auto_lr_find=True,
            auto_scale_batch_size=False,
            fast_dev_run=False,
            logger=[tb_logger,csv_logger],
            deterministic=False # Does not work with some 3D max pool operations (too bad)
            )
        
        trainer.fit(model, dm)
        # Step crossValCounter            
        crossValCounter += 1
        # Pause for a while (avoid GPU problems in cross validation it seems)
        time.sleep(120)
        
        ### 
        # Added for the experiments when patient material is varyed in size 
        # Send email when training is finished
        # Execute system command for sending email. Relays on correct Linux system SMTP configuration
        emailCmd = r'echo "Training has completed versionIter {} fold {} " | mail -s "Training has completed versionIter {} fold {} " XXX@gmail.com'.format(conf.model.versionIter, crossValCounter, conf.model.versionIter, crossValCounter)
        os.system(emailCmd) 
     
        # Break if cross validation continues past the first fold 
        # Used for validation different amounts of patient data 
        #if crossValCounter > 0:
        #    break


        ## END cross validation loop ##
 