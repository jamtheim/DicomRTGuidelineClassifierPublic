# *********************************************************************************
# Author: Christian Jamtheim Gustafsson, PhD, Medical Physcist Expert
# Skåne University Hospital, Lund, Sweden and Lund University, Lund, Sweden
# Description: Class for loading data from Nifti files
# *********************************************************************************

# Modules needed for this class
import numpy as np
import os
import SimpleITK as sitk 
import matplotlib.pyplot as plt
import json
import logging
import sys
import pickle
from numpy.core.fromnumeric import size
from scipy import ndimage
from skimage import morphology
from skimage.measure import label
from skimage.transform import resize

from commonConfig import commonConfigClass
conf = commonConfigClass() 


class ioDataMethodsClass:
    """
    Class describing functions needed for reading Nifti data
    """

    def __init__ (self):
        """
        Init function
        """
        pass


    def createDatasetTextFile(self, folderPath, fileName, organList):
        """
        Write a text file which defines the dataset for the training
        
        Args:
            filePath (str): Path to the folder where the file will be saved
            fileName (str): name of the file

        Return:
            None
        """
        assert isinstance(folderPath, str), "Input must be string"
        assert isinstance(fileName, str), "Input must be string"
        assert isinstance(organList, list), "Input must be list"
        # Create file path
        filePath = os.path.join(folderPath, fileName)
        # If the data set file already exists, do nothing
        if os.path.isfile(filePath):
            return

        # Make sure folder exist 
        os.makedirs(os.path.dirname(filePath), exist_ok=True)
        # Get number of organs 
        totnrOrgans = len(organList)
        # Write file
        with open(filePath, 'w') as f:
            f.write("{  \n")
            f.write("\t \"task\": \"{}\", \n".format(conf.data.TaskNumber))
            f.write("\t \"name\": \"{}\", \n".format(conf.data.TaskNumber))
            f.write("\t \"dim\": 3, \n")
            if totnrOrgans == 1:
                f.write("\t \"target_class\": 0, \n")
            else:
                f.write("\t \"target_class\": null, \n")
            f.write("\t \"test_labels\": false, \n")
            f.write("\t \"labels\": { \n")
            # Write the organs and their class labels in the text file
            # Make sure the last line is not ending with a comma
            for i_organ, currOrgan in enumerate(organList):
                # Get the index position of the organ in the defined organ list  
                organIndex = conf.organ.organNames.index(currOrgan)
                # Define organ class 
                # If there is only one organ the class must be set to zero
                if totnrOrgans == 1:
                    organClass = 0
                else: 
                    # Get organ class organ class vector index position.
                    # organClassIndex is defined in config from conf.organ.organNames
                    organClass = conf.organ.organClasses[organIndex]
                # Fill text file
                if i_organ != totnrOrgans-1: 
                    f.write("\t \t \"{}\": \"{}\", \n".format(organClass, currOrgan))
                if i_organ == totnrOrgans-1:
                    f.write("\t \t \"{}\": \"{}\" \n".format(organClass, currOrgan))
            f.write("\t }, \n")
            f.write("\t \"modalities\": { \n")
            f.write("\t \t \"0\": \"CT\" \n " )
            f.write("\t } \n")
            f.write("} \n")

        # Close file
        f.close()
        print("Created dataset description file: {}".format(filePath))

 
    def createLabelInstanceTextFile(self, filePath, fileName, organsList, instanceValuesList):
        """
        Write a text file which defines the class label for each instance segmentation label
        that is availbale in the mask 

        Args:
            filePath (str): Path to the folder where the file will be saved
            fileName (str): Name of the file
            organs (list): List of organs to write information about
 
        Return:
            None
        """
        assert isinstance(filePath, str), "Input must be string"
        assert isinstance(fileName, str), "Input must be string"
        assert isinstance(organsList, list), "Input must be list"
        assert isinstance(instanceValuesList, list), "Input must be list"
        assert len(organsList) == len(instanceValuesList), "List must be the same size"
        # Create file path
        filePath = os.path.join(filePath, fileName)
        # Make sure folder exist 
        os.makedirs(os.path.dirname(filePath), exist_ok=True)
        # Write file
        with open(filePath, 'w') as f:
            f.write("{ \n")
            f.write("\t \"instances\": {  \n")
            # Get number of organs 
            totNrOrgans = len(organsList)
            # Write the instanceValues and class label in the text file for each organ in the list
            for i_organ, currOrgan in enumerate(organsList):
                # Get the index position of the organ in the defined organ list
                organIndex = conf.organ.organNames.index(currOrgan)

                # Define organ class 
                # If there is only one organ the class must be set to zero
                if totNrOrgans == 1:
                    organClass = 0
                else: 
                    # Get organ class organ class vector index position.
                    # organClassIndex is defined in config from conf.organ.organNames
                    organClass = conf.organ.organClasses[organIndex]

                # Get instanceValue 
                instanceValue = instanceValuesList[i_organ]
                # Fill text file
                if i_organ != totNrOrgans-1: 
                    f.write("\t \t \"{}\": {}, \n".format(instanceValue, organClass))
                if i_organ == totNrOrgans-1:  # Last iteration
                    f.write("\t \t \"{}\": {} \n".format(instanceValue, organClass))
            f.write("\t } \n")
            f.write("} \n")
        # Close file
        f.close()
        # print("Created label instance description file: {}".format(filePath))


    def removeSmallSignalRegions(self, mask, threshold, patFolder, organ):
        """
        Remove small signal regions from the mask 
        Return mask withput the small regions 

        Args:
            mask (array): Original mask 
            threshold (int): Size of region to remove
            patFolder (str): Path to patient folder
            organ (str): Name of organ

        Return:
            Mask where small regions have been removed       
        """
        assert len(mask.shape) == 3, "dim should be 3"
        # Label the mask and count the labels
        labelMask, nrLabels = label(mask, return_num=True)    
        if nrLabels > 1: 
            # Remove small objects 
            mask = morphology.remove_small_objects(label(mask), threshold) 
            # Make resulting mask uint8
            mask = np.array(mask, dtype='uint8')
            print("Removed small signal regions from mask in {} {}".format(patFolder, organ))
        # Return the mask
        return mask


    def limitNrPatients(self, AllPatList, nrPatients):
        """
        Limit the number of patients to proces to nrPatients

        Args:
            AllPatList (list): List of all patients
            nrPatients (int or str): Number of patients to limit to, can be 'all' also

        Return:
            List of patients to limit processing to
        """
        if isinstance (nrPatients, int): 
            assert isinstance(AllPatList, list), "Input must be list"
            # Limit the number of patients
            patListOut = AllPatList[:nrPatients]
            # Return the list
            return patListOut

        if  nrPatients=='all':
            assert isinstance(AllPatList, list), "Input must be list"
            patListOut = AllPatList 
            # Return the list
            return patListOut
   

    def saveNiftiFile(self, np_imageData, sitk_imageData, outPutFilePath):
        """
        Saves 3D Nifty file

        Args:
            np_imageData (array): 3D numpy array
            sitk_imageData (sitk image): 3D sitk image
            outPutFilePath (str): Path to the file to be saved
 
        Return:
            None
        """
        # Assert numpy array
        assert isinstance(np_imageData, np.ndarray), "Input must be numpy array"
        # Reorder back so slices in the 3D stack will be in the first dimension
        # This is the numpy format from SITK when exported as numpy array
        # Input assertion to make sure 3D image
        assert len(np_imageData.shape) == 3, "dim should be 3"
        np_imageData = np.transpose(np_imageData, (2,0,1))
        # Converting back to SimpleITK 
        # (assumes we didn't move the image in space as we copy the information from the original)
        outImage = sitk.GetImageFromArray(np_imageData)
        outImage.CopyInformation(sitk_imageData)
        # Make sure folder exist before saving 
        os.makedirs(os.path.dirname(outPutFilePath), exist_ok=True)
        # Write the image
        sitk.WriteImage(outImage, outPutFilePath)


    def zScoreNorm(self, imageVolume, ignoreAir=False):
        """
        Z-score normalize the image volume

        Args:
            imageVolume (array): 3D numpy array 

        Return:
            imageVolumeNormalized (array): 3D numpy array 
        """
        assert len(imageVolume.shape) == 3, "dim should be 3"
        # Set HU threshold to ignore air or not. Air = -1000 HU 
        if ignoreAir==True:
            threshold = -1000
        else:
            threshold = -1001
        # Z-score normalization
        mean = np.mean(imageVolume[imageVolume>threshold])
        std = np.std(imageVolume[imageVolume>threshold])
        imageVolumeNormalized = (imageVolume - mean) / std
        # Return the normalized image
        return imageVolumeNormalized


    def cropImageFromMask(self, image, mask):
        """
        Crop image from mask 

         Args:
            image (array): 3D image
            mask (array): 3D mask 

        Return:
            croppedImage (array): Cropped image
        
        """
        assert len(image.shape) == 3, "dim should be 3"
        assert len(mask.shape) == 3, "dim should be 3"
        # Coordinates of non-zero elements in the mask
        coords = np.argwhere(mask)
        # Bounding box coordinates of the box mask
        x0, y0, z0 = coords.min(axis=0)
        x1, y1, z1 = coords.max(axis=0) + 1   # slices are exclusive at the top
        # Get the extracted contents of the box
        croppedImage = image[x0:x1, y0:y1, z0:z1]
        # Return the cropped image
        return croppedImage


    def readNiftiFile(self, filePath): 
        """
        Read 3D Nifti files to numpy array. 
        Get image resolution in image and the SITK object.
        input: file path for Nifti file
        output: image data in Numpy format, SITK object, image resolution in tuple format
                
        Args:
            filePath (str): Path to the file to be read
   
        Return:
            None
        """
        assert isinstance(filePath, str), "Input must be string"
        # Read the .nii image containing the volume using SimpleITK.
        # With SimpleITK the image orientation in the numpy array is correct
        sitk_imageData = sitk.ReadImage(filePath)
        # Access the numpy array from the SITK object
        np_imageData = sitk.GetArrayFromImage(sitk_imageData)
        # Get pixelSpacing in image from the SITK object
        pixelSpacing = sitk_imageData.GetSpacing()
        # Input assertion to make sure 3D image
        assert len(np_imageData.shape) == 3, "dim should be 3"
        # Reorder so slices in the 3D stack will be in the last dimension
        np_imageData = np.transpose(np_imageData, (1,2,0))
        # Return np_imagedata, sitk_imageData and pixel spacing
        return np_imageData, sitk_imageData, pixelSpacing


    def calcImageVolume(self, array, pixelSpacing):
        """
        Calculate the volume of array originating from reading a Nifti structure volume.

        Args:
            array (array): 3D numpy array
            pixelSpacing (tuple): Pixel spacing in the image

        Return:
            volume (float): Volume of the structure in the image
        """
        # Cast as int8
        array = array.astype(np.int8)
        # Assert binary mask
        assert isinstance(array, np.ndarray), "Input must be numpy array"
        assert len(array.shape) == 3, "Mask must be 3D"
        # Assert tuple
        assert isinstance(pixelSpacing, tuple), "Pixel spacing must be tuple"
        assert array.max() == 1, "Mask must be binary"
        assert array.min() == 0, "Mask must be binary"
        # Calculate sum of voxels
        nrVoxels = np.sum(array)
        # Calculate volume
        volume = nrVoxels * pixelSpacing[0] * pixelSpacing[1] * pixelSpacing[2]
        # Return volume in mm3
        return volume


    def assertBinaryMask(self, mask):
        """
        Assert that the input is a binary mask

        Args:
            mask (array): 3D numpy array
   
        Return:
            None
        """
        assert isinstance(mask, np.ndarray), "Input must be numpy array"
        assert len(mask.shape) == 3, "Mask must be 3D"
        if mask.sum() ==  0: 
            assert mask.max() == 0, "Mask must be binary"
            assert mask.min() == 0, "Mask must be binary"
        if mask.sum() >  0: 
            assert mask.max() == 1, "Mask must be binary"
            assert mask.min() == 0, "Mask must be binary"


    def getBoundingBoxFilled(self, mask, value, margin):
        """
        Get a filled bounding box as a binary mask

        Args:
            mask (array): 3D numpy array
            value (int): Value to fill the bounding box with
            margin (tuple): Margin to add to the bounding box, in voxels

   
        Return:
            None
        """
        # Get bounding box of the mask
        x, y, z = np.where(mask)
        # Load margin
        xMargin, yMargin, zMargin = margin
        # Get the bounding box
        boundingBox = [x.min()-xMargin, x.max()+xMargin, y.min()-yMargin, y.max()+yMargin, z.min()-zMargin, z.max()+zMargin]
        # If negative values are created in the bounding box after adding margins, set them to value 0
        boundingBox = [0 if x < 0 else x for x in boundingBox]
         # Create a mask of the bounding box
        boundingBoxmask = np.zeros(mask.shape)
        boundingBoxmask[boundingBox[0]:boundingBox[1],boundingBox[2]:boundingBox[3],boundingBox[4]:boundingBox[5]] = value
        return boundingBoxmask


    def getCenterSlice(self, mask): 
        """
        Return the center slice of a mask calculated from where signal is present
                
        Args:
            mask (array): 3D numpy array

        Return:
            None
        """
        # Check mask 
        self.assertBinaryMask(mask)
        # Get slices with signal 
        sliceWithSignal = np.argwhere(mask == True)
        # Get minimum and maximum, value in third dimension
        minZ = sliceWithSignal[:,2].min()
        maxZ = sliceWithSignal[:,2].max()
        # Calculate center slice
        centerZ = int(np.round(minZ + (maxZ-minZ)/2))
        # View slice
        #plt.imshow(mask[:,:,centerZ])
        #plt.show()
        return centerZ

    
    def logAndPrint(self, message):
        """
        For enhanced logging to both file and console
        This function does not work in multi processing

        Args:
            message (str): Message to be logged

        Return:
            None
        """
        # Define logger object
        a_logger = logging.getLogger()
        # Set loggin level
        a_logger.setLevel(logging.DEBUG)
        # Create file handler from defined file path
        output_file_handler = logging.FileHandler(os.path.join(conf.data.dataOutputPath, conf.data.logFileName))
        stdout_handler = logging.StreamHandler(sys.stdout)
        # Output both to file and console
        a_logger.addHandler(output_file_handler)
        a_logger.addHandler(stdout_handler)
        # return logging object
        return a_logger
    

    def removeItemsFromList(self, originalList, itemsToRemove): 
        """
        Remove items from input list which are defined in itemsToRemove
        
        Args:
            originalList (list): List to be filtered
            itemsToRemove (list): List of items to be removed

        Return:
            editedList (list): List with items removed  
            existFlag (bool): True if itemsToRemove exist in originalList
        """              
        # Init exist flag
        existFlag = 0
        # First make sure all is lower case for itemsToRemove as this is standard
        itemsToRemove = [each_string.lower() for each_string in itemsToRemove]
        # Copy the original list to a new variable
        # Do not use X = Y, this creates only a reference for lists.
        editedList = originalList.copy()
        # Loop through all items in input list to see if they starts with any of the objects defined in the itemsToRemove. 
        # If so, remove it from new list. 
        for i, item in enumerate(originalList): 
            if item.lower().startswith(tuple(itemsToRemove)): 
                editedList.remove(item)
                # Set exist flag
                existFlag = 1
        # Return data    
        return editedList, existFlag


    def getLargestFile(self, folderOfInterest): 
        """
        Get the largest mask file in a directory 

        Args: 
            folderOfInterest (str): Path to the folder containing the masks
        
        Return: 
            The file name
        """
        # Get files
        folderFiles = os.listdir(folderOfInterest)
        # Selct only files that has prefix 'mask_'
        folderFiles = [file for file in folderFiles if file.startswith('mask_')]
        # Get file sizes
        fileSizes = [os.path.getsize(folderOfInterest + '/' + file) for file in folderFiles]
        # Get maximum file size
        maxFileSize = np.amax(fileSizes)
        # Get index of that entry in the list
        maxFileSizeIndex = np.argmax(fileSizes)
        # Get the file from original list
        largestFile = folderFiles[maxFileSizeIndex]
        assert len([largestFile]) == 1, "There should only be one file with the largest size"
        # Return file
        return largestFile, maxFileSize


    def checkImageResolution(self, imageArray, pixelSpacing, desiredResolution, tolerance, folder, organ):
        """
        Check that the image resolution is correct and within tolerence

        Args:
            imageArray (array): 3D numpy array
            pixelSpacing (tuple): pixel spacing tuple
            tolerance (float): Tolerance for the pixel spacing

        Return:
            None
        """
        # Check matrix size
        try:
            assert imageArray.shape[0] == 512, "Matrix has the wrong size (rows)" 
            assert imageArray.shape[1] == 512, "Matrix has the wrong size (columns)" 
            # Check if pixel spacing is within tolerance
            assert abs(1-pixelSpacing[0]/desiredResolution[0]) < tolerance, "Row pixel spacing is not within tolerance" 
            assert abs(1-pixelSpacing[1]/desiredResolution[1]) < tolerance, "Column pixel spacing is not within tolerance" 
            assert abs(1-pixelSpacing[2]/desiredResolution[2]) < tolerance, "Slice pixel spacing is not within tolerance" 
        except:
            print("Image resolution is not correct for " + folder + " " + organ)
            
    
    def fuseRTstructures(self, folder, ignoreStructures):
        """
        Read in all Nifti structure files and fuse the binary masks.
        Body structure is not added, but used to exclude the table top. 
        Truncate the fused array value to 1 and return uint8 format.
        Exclude structures which are defined as optimization structures. 

        Args:
            folder (str): Path to the folder containing the Nifti files
            ignoreStructures (list): List of structures to be ignored

        Return:
            fusedMask (array): Fused array from all masks
        """ 
        # Get all structure files available 
        structFiles = os.listdir(folder)
        # Make sure the list only contains nii.gz files
        structFiles = [file for file in structFiles if '.nii.gz' in file]   
        # Remove CT image file 
        structFiles = [file for file in structFiles if conf.data.CTImageFileName not in file]   
        # Remove file endings and get available structure list
        structFiles = [file.replace(conf.data.fileSuffix,'') for file in structFiles if conf.data.fileSuffix in file]  
        # Remove mask_ prefix
        structFiles = [file.split(conf.data.filePrefix)[1] for file in structFiles if conf.data.filePrefix in file]
        # Create lower letter version of the list
        structFilesLower = [file.lower() for file in structFiles]
        # Remove structures to ignore 
        structFilesCleaned, existFlag = self.removeItemsFromList(structFiles, ignoreStructures)
        # Make sure the BODY structure is included in the list, can be defined as multiple names.  
        assert any(item in conf.data.bodyStructureName for item in structFiles), "BODY structure is missing"
        # Remove it from the list as we do not want value 1 everywhere. Several BODY names can be matched. Uses lower case. 
        structFilesCleaned, existFlag = self.removeItemsFromList(structFilesCleaned, conf.data.bodyStructureName)
        # Determine the exact existing BODY structure name(s) 
        existingBodyStructureName = [item for item in conf.data.bodyStructureName if item.lower() in structFilesLower]
        # Assert only one item for existingBodyStructureName
        # assert len(existingBodyStructureName) == 1, "Multiple BODY structure names found in the data"
       
        # Collect all data from the structures in the final list 
        # This is not parallellized because we parallellize calculation over patients instead
        # Init variable
        fused_np_imageData = []  
        for i_struct, currStructure in enumerate(structFilesCleaned):
            filePath = os.path.join(folder, conf.data.filePrefix + currStructure + conf.data.fileSuffix)
            # Read Nifti data  from file 
            np_imageData, sitk_imageData, pixelSpacing = self.readNiftiFile(filePath)
            # Make sure np_image data is uint8
            np_imageData = np.uint8(np_imageData)
            # Add to the fused numpy array 
            if not len(fused_np_imageData): # If not existing, create new uint8 array
                        fused_np_imageData = np.zeros(np_imageData.shape, dtype=np.uint8)
                        fused_np_imageData = fused_np_imageData + np_imageData
            else:
                fused_np_imageData = fused_np_imageData + np_imageData

        # Read the fist BODY structure found (possibly several can be found)
        filePathBody = os.path.join(folder, conf.data.filePrefix + str(existingBodyStructureName[0]) + conf.data.fileSuffix)
        # Read Nifti body data from file 
        np_imageDataBody, sitk_imageDataBody, pixelSpacingBody = self.readNiftiFile(filePathBody)
        # Cut away table top data by multiplying with body mask
        fused_np_imageData = fused_np_imageData * np_imageDataBody
        # Make sure data is uint8
        fused_np_imageData = np.uint8(fused_np_imageData)
        # Truncate data
        fused_np_imageData[fused_np_imageData > conf.data.structureFuseTruncValue] = conf.data.structureFuseTruncValue
        # Make sure image data only contains binary values
        self.assertBinaryMask(fused_np_imageData)
        # Return fused array
        return fused_np_imageData


    def saveNpArray(self, np_array, folderPath, fileName):
        """
        Save compressed numpy array to file
        
        Args:
            np_array (array): Numpy array to be saved
            folderPath (str): Path to the folder where the file should be saved
            fileName (str): Name of the file to be saved

        Return:
            None
        """
        # Make sure folder exists before saving
        os.makedirs(os.path.dirname(os.path.join(folderPath, fileName)), exist_ok=True)
        # Save compressed data. (fusedStructures is the name to call when loading data)
        np.savez_compressed(os.path.join(folderPath, fileName), fusedStructures=np_array)
     

    def writeClassificationQAimages(self, GLNumber, patient, label, imageData, maskData):
        """
        Write QA PNG file of image data with mask of mask overlay
        
        Args:
            imageData (array): Image data 
            maskData (array): Mask data 

        Return:
            Write out a PNG file 
        """
        # Make sure folder exists
        os.makedirs(os.path.join(conf.data.dataOutputQAPath, conf.data.TaskNumber, GLNumber), exist_ok=True)
        # Get center slice of the croppedFused structure 
        centerSlice = self.getCenterSlice(maskData)
        # Get edge of mask 
        edgeMask = self.getMaskEdge(maskData)
        # Save fusion to PNG file 
        plt.imsave(os.path.join(conf.data.dataOutputQAPath, conf.data.TaskNumber, GLNumber, patient + '_box_label' + str(int(label)) + '.png'), imageData[:,:,centerSlice] + edgeMask[:,:,centerSlice])


    def writeClassificationQAimagesPerStructure(self, subject, fileName, CTData, AddMapData, basePath, TaskNumber):
            """
            Write QA PNG file of image data for each structure with a mask overlay
            
            Args:
                subject (str): Subject ID
                fileName (str): Name of the structure
                CTData (array): Cropped CT image data
                AddMapData (array): Cropped AddMap data

            Return:
                Write out a PNG file 
            """
            assert CTData.shape == AddMapData.shape, "CT and AddMap data have different shapes"
            # Make sure folder exists
            os.makedirs(os.path.join(basePath, conf.data.TaskNumber, conf.base.dataOutputStructureDir), exist_ok=True)
            # Get center slice of the AddMap structure with structure of interest in focus
            sliceWithSignal = np.argwhere(AddMapData == 1)
            # Get minimum and maximum, value in third dimension
            minZ = sliceWithSignal[:,2].min()
            maxZ = sliceWithSignal[:,2].max()
            # Calculate center slice
            centerSlice = int(np.round(minZ + (maxZ-minZ)/2))
            # Get label from file name 
            tmp_label = fileName.replace(conf.data.fileSuffix,'')
            # Remove mask_ prefix
            label = tmp_label.split(conf.data.filePrefix)[1] 
            # Make sure save directory exists
            os.makedirs(os.path.join(basePath, TaskNumber, conf.base.dataOutputStructureDir), exist_ok=True)
            # Save fusion data to PNG file 
            plt.imsave(os.path.join(basePath, TaskNumber, conf.base.dataOutputStructureDir, str(subject) + '_' + str(label) + '.png'), CTData[:,:,centerSlice] + 2*AddMapData[:,:,centerSlice])


    def resizeImageData(self, imageData, newShape, imageType):
        """
        Resize image data to new shape
        Uses interpolation set by the number 
        0:Nearest-neighbor, 2:Bi-linear, 3:Bi-quadratic, 4:Bi-cubic, 5:Bi-quartic
        and adapt to if input is image or binary segmentation 
   
        Args:
            imageData (array): Image data 
            newShape (tuple): New shape of the image

        Return:
            Resized image data
        """
        # Assert numpy array
        assert isinstance(imageData, np.ndarray), "Image data is not a numpy array"
        # Assert tuple
        assert isinstance(newShape, tuple), "New shape is not a tuple"
        if imageType == 'img':
            order = 3
            anti_aliasing_value = True
            clip_value = True
            preserve_rangebool_value = True
        if imageType == 'seg':
            order = 0   
            anti_aliasing_value = False
            clip_value = True
            preserve_rangebool_value = True
            self.assertBinaryMask(imageData)
            
        # Resize image data to new size
        imageData = resize(imageData.astype(float), newShape, order, mode="constant", cval=0, clip=clip_value, preserve_range=preserve_rangebool_value, anti_aliasing=anti_aliasing_value)
        # Assert still binary if segmentation 
        if imageType =='seg': 
            self.assertBinaryMask(imageData)

        # Return resized image data
        return imageData
        
     
    def padAroundImageCenter(self, imageArray, paddedSize):
        """
        Pad matrix with zeros to desired shape.
        
        Args:
            imageArray (array): Image array to be padded
            paddedSize (int): Size of matrix after zero padding

        Return:
            paddedImageArray (array): Padded image array
        """
        # Assert tuple and np array
        assert isinstance(paddedSize, tuple), "Padded size is not a tuple"
        assert isinstance(imageArray, np.ndarray), "Image array is not a numpy array"
        # Assert image size is not larger than padded size
        assert imageArray.shape[0] <= paddedSize[0], "Image size is larger than requested padded size in row: " + str(imageArray.shape[0]) + " vs " + str(paddedSize[0])
        assert imageArray.shape[1] <= paddedSize[1], "Image size is larger than requested padded size in column: " + str(imageArray.shape[1]) + " vs " + str(paddedSize[1])
        assert imageArray.shape[2] <= paddedSize[2], "Image size is larger than requested padded size in slice: " + str(imageArray.shape[2]) + " vs " + str(paddedSize[2])
        # Get shape of the image array
        origShape = imageArray.shape
        # Caluclate half the difference between the desired 
        # size and the original shape and round up
        diff = np.round((np.array(paddedSize) - np.array(origShape))//2)
        # Calculate padding. Takes care of case when matrix are uneven size. 
        extraLeft = diff[0]
        extraRight = paddedSize[0] - origShape[0] - diff[0]
        extraTop = diff[1]
        extraBottom = paddedSize[1] - origShape[1] - diff[1]
        extraFront = diff[2]
        extraBack = paddedSize[2] - origShape[2] - diff[2]
        # Pad the image array with zeros
        paddedImageArray = np.pad(imageArray, ((extraLeft,extraRight), (extraTop,extraBottom), (extraFront, extraBack)), 'constant', constant_values=0)
        # Assert correct padded size, very important
        assert paddedImageArray.shape[0] == paddedSize[0], "Padded image size is incorrect in row"
        assert paddedImageArray.shape[1] == paddedSize[1], "Padded image size is incorrect in column"
        assert paddedImageArray.shape[2] == paddedSize[2], "Padded image size is incorrect in slice"
        # Return the padded image array
        return paddedImageArray


    def getMaskEdge(self, mask):
        """
        Get the edge of the binary mask. 
        This is used to display the mask without overlaying the whole mask 

        Args:
            mask (array): Binary mask array

        Return:
            edge (array): Edge of the mask
        """
        # Assert binary mask
        self.assertBinaryMask(mask)
        # Convert to double
        img_data = np.asarray(mask[:, :, :], dtype=np.double)
        # Calculate gradient of mask 
        gx, gy, gz = np.gradient(img_data)
        # Convert to positive values of gradient (ignore z direction)
        edge = gy * gy + gx * gx 
        # Assign value to gradient edge 
        edge[edge != 0] = 1
        # Cast as uint8
        edge = np.asarray(edge, dtype=np.uint8)
        # Make sure they are the same size
        assert edge.shape == mask.shape, "Edge and mask are not the same size"
        # Return gradient of mask 
        return edge


    def readDataToDataloader(self, dataFilePath): 
        """            
        Read data from a Numpy save file and return the data as a float 32 numpy array

        Args:
            dataFilePath (str): Path to the data file

        Return:
            data (array): Data array
        """
        assert isinstance(dataFilePath, str), "Data file path is not a string"
        assert os.path.isfile(dataFilePath), "Data file does not exist"
        # Load data from file
        img = np.load(dataFilePath)['fusedStructures'].astype(np.float32) #preDetermined name of the numpy array
        # Move channels first in array
        img = np.moveaxis(img, -1, 0) 
        # Assert 4D Numpy array
        assert img.ndim == 4, "Image data is not 4D"
        # Return data 
        return img
        

    def writeClassificationTrainingData(self, GLNumber, patient, label, croppedCT, croppedStructures):
        """
        Write out Numpy training data to file as a 4D array

        Args:
            cropped CT (array): Cropped part of the CT
            cropped fused structures (array): Cropped part of the fused structures 

        Return:
            Numpy file (array): 4D array with cropped CT and fused structures in different channels
        """
        # Assert binary mask
        self.assertBinaryMask(croppedStructures)
        # Stack the 3D arrays into 4D array 
        array4Dfused = np.stack((croppedCT, croppedStructures), axis = -1)
        # Assert 4D array
        assert len(array4Dfused.shape) == 4, "Array must be 4D"
        # Make sure save folder exists
        os.makedirs(os.path.join(conf.data.dataOutputClassificationTrainingDataPath, conf.data.TaskNumber, GLNumber), exist_ok=True)
        # Save fusion to Numpy file 
        self.saveNpArray(array4Dfused, os.path.join(conf.data.dataOutputClassificationTrainingDataPath, conf.data.TaskNumber, GLNumber), patient + '_4D_label' + str(int(label)))
   

    def writeClassificationTrainingDataPerStructure(self, subject, fileName, CTData, AddMapData, basePath, TaskNumber):
        """
        Write out Numpy training data for each structure to file as a 4D array

        Args:
            subject (string): Subject ID
            fileName (string): File name of the structure
            CTData (array): Cropped part of the CT
            AddMapData (array): Cropped part of the fused structures
                        
        Return:
            Numpy file (array): 4D array with cropped CT and cropped AddMap in different channels
        """
        
        assert CTData.shape == AddMapData.shape, "CT and AddMap data are not the same size"
        # Fuse CT and AddMap to a 4D matrix, AddMap as second channel 
        np_CT_AddMap_4D = np.stack((CTData, AddMapData), axis=-1)
        # Assert number of slices > 0
        assert np_CT_AddMap_4D.shape[2] > 0, 'Number of slices in CT and AddMap is 0'
        # Assert 4D array
        assert len(np_CT_AddMap_4D.shape) == 4, "Array must be 4D"
        # Get label from file name 
        tmp_label = fileName.replace(conf.data.fileSuffix,'')
        # Remove mask_ prefix
        label = tmp_label.split(conf.data.filePrefix)[1] 
         # Make sure save folder exists
        os.makedirs(os.path.join(basePath, conf.data.TaskNumber, conf.base.dataOutputStructureDir), exist_ok=True)
        # Save fusion to Numpy file 
        self.saveNpArray(np_CT_AddMap_4D, os.path.join(basePath, TaskNumber, conf.base.dataOutputStructureDir), subject + '_' + str(label))
   


    def editFusedStructuresToy(self, fusedStructures):
        """
        Edit structures for Toy dataset creation. 
                
        Args:
            fusedStructures (array): Fused structures 

        Return:
            fusedStructuresEdited (array): Fused structures edited 
            
        """
        #  Assert binary mask
        self.assertBinaryMask(fusedStructures)
        # Get size of the mask
        maskSize = fusedStructures.shape
        # Get middle row
        middleRow = int(maskSize[0] / 2)
        # Copy too another variable
        fusedStructuresEdited = fusedStructures.copy()
        # Set all data between first row and middle row to 0
        # i.e we remove part of the mask
        fusedStructuresEdited[:middleRow,:,:] = 0
        # Return data 
        return fusedStructuresEdited


    def getBodyStructure(self, subject, folder, structFiles):
        """
        Get the body structure from the folder contents.
                
        Args:
            folder (string): Folder name

        Return:
            bodyStructure (string): Body structure
            
        """
        # Assertions
        assert type(folder) == str, "Folder must be a string"
        assert type(structFiles) == list, "structFiles must be a list"
        
        # Create an empty list 
        bodyStructNameCandidates = [] 
        # Extract all list items in structFiles containing any of the template body words in the conf.base.bodyStructureName
        for templateName in conf.base.bodyStructureName:
            for structure in structFiles:
                if templateName.lower() in structure.lower(): # If word in structure name 
                    bodyStructNameCandidates.append(structure) # Add to list
      
        # Define what body structure file to use (bodyStructNameCandidates) if multiple body structures are found 
        if len(bodyStructNameCandidates) == 0: 
            raise Exception("No BODY structure names was found for the subject " + subject)   
        elif len(bodyStructNameCandidates) == 1:
            bodyStructFileUse = bodyStructNameCandidates[0] # Use the only one existing 
        else:
            print('Warning: Multiple BODY structure files found for subject ' + subject)
            bodyStructFileUse = [file for file in bodyStructNameCandidates if conf.data.filePrefix + 'body' in file.lower()] # Select the ones with mask_body in the name
            if len(bodyStructFileUse) == 0:
                bodyStructFileUse = bodyStructNameCandidates[0] # No files with 'body' found, use the first one
                print('Using the first body file one found: ' + bodyStructFileUse)
            else:
                bodyStructFileUse = bodyStructFileUse[0] # Use the one with 'body' in the name
                print('Using the one containing the name body: ' + bodyStructFileUse)

        # Old solution: Get the largest mask file. This should logically correspond to the body structure mask.
        # This did not work as other files interfered with larger size. However saving the line for future use. 
        # bodyStructFileUse, largestFileSize = self.getLargestFile(folder)
        # Get lower version of the body structure keywords
        bodyStructureNamesTemplateLower = [each_string.lower() for each_string in conf.base.bodyStructureName]
        assert len([bodyStructFileUse]) == 1, "More than one body structure file found"
        # Double Check that any of the body template keywords are in the determined file name (lower case)
        boolBodyCheck = any(bodyName in bodyStructFileUse.lower() for bodyName in bodyStructureNamesTemplateLower)
        # Make sure some body structure keywords exist within this file name 
        assert boolBodyCheck, "Body structure name not found in detected body file name"
        # Return data 
        return bodyStructFileUse


    def getNumberOfUsedSlices(self, structure):
        """
        Get the number of slices and slice distance in the structure which contain signal.
        
        Args:
            structure (array): Structure to check
        Return:
            numberOfSignalSlices (int): Number of slices with signal
        """
        x, y, z = np.where(structure)
        firstSlice = np.min(z)
        lastSlice = np.max(z)
        numberOfSlices = lastSlice - firstSlice + 1
        return numberOfSlices


    def truncVolSliceToStruct(self, np_struct, np_CT, AddMap):
        """
        Be Aware: This pre-processing might not be as good as extendring the CT (truncVolSliceToDesiredSize)
        Truncation of volumes in slice direction with respect to the extent of the structure. 
        The new volume has the same number of slices as the structure. 
        This is performed to limit the CT and AddMap information to the same extent as the structure.
        
        Args:
            np_struct (array): Structure volume 
            np_CT (array): CT volume
            AddMap (array): AddMap volume         
                    
        Return:
            np_struct_trunc (array): Truncated structure volume
            np_CT_trunc(array): Truncated CT volume
            AddMap_trunc (array): Truncated AddMap volume
        """
        # Get voxel coordinates of the structure
        x, y, z = np.where(np_struct)
        firstSlice = np.min(z)
        lastSlice = np.max(z) +1 # +1 because max is exclusive in the selection 
        # Limit the slices in the data volumes with respect to number of slices in the structure  
        np_CT_ztrunk = np_CT[:,:,firstSlice:lastSlice]
        np_struct_ztrunk = np_struct[:,:,firstSlice:lastSlice]
        AddMap_ztrunk = AddMap[:,:,firstSlice:lastSlice]
        # Make sure they are not empty after truncation
        if np_CT_ztrunk.sum() == 0 or np_struct_ztrunk.sum() == 0 or AddMap_ztrunk.sum() == 0:
            raise ValueError('Data was found to be empty empty after truncation') 
        # Assert shapes 
        assert np_struct_ztrunk.shape == np_CT_ztrunk.shape, 'Shape of cropped CT do not match' 
        assert AddMap_ztrunk.shape == np_CT_ztrunk.shape, 'Shape of the AddMap do not match'
        # Return truncated volumes
        return np_struct_ztrunk, np_CT_ztrunk, AddMap_ztrunk


    def truncVolSliceToDesiredSize(self, np_struct, np_CT, AddMap, desiredImageMatrixSize):
        """
        Truncation of volume in slice direction with respect to the center slice of the structure. 
        The new volume has the desired number of slices where the slices not containing structure 
        is symmetrically spaced around the slices containing structure information.
        With other words: the strucutre is in the center of the new truncated volume.
        This is performed to be able to contain the CT information for more slices than the structure content.
        
        Args:
            np_struct (array): Structure volume 
            np_CT (array): CT volume
            AddMap (array): AddMap volume         
            desiredImageMatrixSize (tuple): Desired number of slices in the new volume  
            
        Return:
            np_struct_trunc (array): Truncated structure volume
            np_CT_trunc(array): Truncated CT volume
            AddMap_trunc (array): Truncated AddMap volume
        """
        # We are counting distances between slices, not amount of slices, important to remember below
        # Assertions
        assert type(desiredImageMatrixSize) == tuple, "desiredImageMatrixSize must be a tuple"
        assert np_struct.shape == np_CT.shape, "CT and structure data are not the same size"
        assert np_struct.shape == AddMap.shape, "AddMap and structure data are not the same size"
        totSlicesDesired = desiredImageMatrixSize[2]
        assert (totSlicesDesired % 2) == 0, "totSlicesDesired of slices must be even as it should be defined as a multiplyer of 8"
        # Get voxel coordinates for the used voxels in the structure 
        x, y, z = np.where(np_struct)
        startSlice = np.min(z)  
        endSlice = np.max(z)  
        # Get distance in slices between first and last slice
        spanSliceDist = endSlice - startSlice
        # Calculate middle position in the structure 
        middleSliceDist = startSlice + spanSliceDist/2   
        # Calculate new starting position for volume truncation (rounded down)   
        newStartSlice = np.int(np.floor(middleSliceDist - spanSliceDist/2 - (totSlicesDesired-spanSliceDist)/2))
        # If new start slice is negative, set it to 0                 
        if newStartSlice < 0:
            newStartSlice = 0
            newEndSlice = newStartSlice + totSlicesDesired # Start from first possible slice (0) and expand from there 
        else:
            newEndSlice = newStartSlice + totSlicesDesired
        # If calculated end slice is larger than the number of slices in 
        # the original struct volume set to last slice in original volume.
        # This will produce a number of slices less than desired. 
        # However, will be zero padded later on to the desired number of slices.
        if newEndSlice > np_struct.shape[2]:
            newEndSlice = np_struct.shape[2]
        # Condition if the original structure volume is smaller or equal than the desired number of slices
        # This will produce a number of slices less than desired. 
        # However, will be zero padded later on. This will overwrite the above conditions.
        if np_struct.shape[2] <= totSlicesDesired:
            newStartSlice = 0
            newEndSlice = np_struct.shape[2]
        # Extract the slices to a new truncated volume
        np_struct_trunc = np_struct[:,:,newStartSlice:newEndSlice]
        np_CT_trunc = np_CT[:,:,newStartSlice:newEndSlice]
        AddMap_trunc = AddMap[:,:,newStartSlice:newEndSlice]
        # Assert shapes 
        assert np_struct_trunc.shape == np_CT_trunc.shape, 'Shape of cropped CT do not match' 
        assert AddMap_trunc.shape == np_CT_trunc.shape, 'Shape of the AddMap do not match'
        # Return truncated volumes
        return np_struct_trunc, np_CT_trunc, AddMap_trunc
        

    def readPickleFileBoxes2nii(self, folderPathIn, file, folderPathOut, threshold):
        """
        Read pickle file generated from the output of the model and get best scored bounding box for 
        each reported label. 
        Something similar existed in the nnDetection package in utils, function boxes2nii
        but it could not handle multiple labels (error in for loop, indentation?). 
        However, my code is heavily inspired from that function and does the same thing. 
        
        Args:
            Pickle file from the prediction output of the model
            
        Return:
            saved Nifti files (output to disk): One with bounding box for each label
            foundLabels (list(str)): List of labels found in the pickle information 
        
        """
        # Create file path for input pickle file
        filePath = os.path.join(folderPathIn, file)
        # Assert existance of pickle file
        assert os.path.isfile(filePath), "Pickle file does not exist"
        # Make sure output tmp folder exist for saving the Nifti files 
        os.makedirs(folderPathOut, exist_ok=True)
        # Init variable
        outputFilePathArray = [] 

        # Read pickle file
        with open(filePath, "rb") as input_file:
            # Get patient name from file name
            patient = file.split(conf.data.boxFileSuffix)[0]
            # Load pickle file
            res = pickle.load(input_file)
            # Get important data from pickle file, boxes, scores and labels
            boxes = res["pred_boxes"]
            scores = res["pred_scores"]
            labels = res["pred_labels"]
            # Select those above the set threshold
            _mask = scores >= threshold
            boxes = boxes[_mask]
            labels = labels[_mask]
            scores = scores[_mask]
            # Get unique labels in the data 
            uniqueLabels = np.unique(labels)
            # Get largest probability value for each unique label
            for label in uniqueLabels:
                # Create an empty mask 
                instance_mask = np.zeros(res["original_size_of_raw_data"], dtype=np.uint8)
                # Get index of label
                idx = np.where(labels == label)[0]
                # Get the best score for this label
                score = np.max(scores[idx])
                # Get the index of the best score
                idx = np.where(scores == score)[0]
                # Get the bounding box for this label
                box = boxes[idx[0]]
                # Add box to the empty mask
                mask_slicing = [slice(int(box[0]) + 1, int(box[2])),
                slice(int(box[1]) + 1, int(box[3])),
                ]
                if instance_mask.ndim == 3:
                    mask_slicing.append(slice(int(box[4]) + 1, int(box[5])))
                instance_mask[tuple(mask_slicing)] = 1 # No instance, every label get value 1, unique output files however. 
                # Create ITK data and meta information 
                instance_mask_itk = sitk.GetImageFromArray(instance_mask)
                instance_mask_itk.SetOrigin(res["itk_origin"])
                instance_mask_itk.SetDirection(res["itk_direction"])
                instance_mask_itk.SetSpacing(res["itk_spacing"])
                # Define output file name 
                outputFilePath = os.path.join(folderPathOut, patient + '_box_label' + str(int(label)) + conf.data.fileSuffix)
                # Save ITK data to Nifti file with label appendix in file name 
                sitk.WriteImage(instance_mask_itk, outputFilePath)
                # Save file path to array
                outputFilePathArray.append(outputFilePath)
        
        # Get integer labels
        returnLabels = [int(label) for label in uniqueLabels] 
        # Return found labels and file path array
        return returnLabels, outputFilePathArray


    def setTrainingData(self, useFullDataBool, dataPathsFull, dataPathsFiltered, classStructures): 
        """
        Define the final data paths to be used in the training pipeline. 
        
        Args:
            useFullDataBool (bool): If False, filter data, otherwise use full data
            dataPathsFull (list(str)): List of full data paths
            dataPathsFiltered (list(str)): List of filtered data paths
            
        Return:
            data_paths (list(str)): List of data paths to be used in the pipeline
        
        """
        # Assert input data
        assert isinstance(useFullDataBool, bool), "Input dataPathsFull is not a bool"
        assert isinstance(dataPathsFull, list), "Input dataPathsFull is not a list"
        assert isinstance(dataPathsFiltered, list), "Input dataPathsFiltered is not a list"
        assert isinstance(classStructures, list), "Input structure names is not a list"
        # If filterDataBool is False, use filtered data, otherwise use full data
        if useFullDataBool == False:
            data_paths = dataPathsFiltered
            num_classes = len(classStructures) 
        elif useFullDataBool == True:
            data_paths = dataPathsFull
            num_classes = len(classStructures) + 1 # +1 for the added 'other' class 
        # Return data paths
        return data_paths, num_classes
        

    def filterDataPaths(self, dataPaths, structureNames): 
        """
        Filter the input data paths to only include the ones that have the 
        structure names in their file name coming from structureNames list.
        Sets demand that the structure name must follow after ID_ in the file name. 
        
        Args:
            dataPaths (list(str)): List of paths to the input data
            structureNames (list(str)): List of structure names to filter the data paths
        
        Return:
            filePathsFiltered (list(str)): Filtered list of paths to the input data
        
        """
        # Assert input data
        assert isinstance(dataPaths, list), "Input data is not a list"
        assert isinstance(structureNames, list), "Input structure names is not a list"
        # Define an empty list 
        filePathsFiltered = [] 
        # Extract all list path items from training data if they contain one of the structures of interest
        for structure in structureNames: 
             # Create a more detailed struct name which helps selection 
            structExt = '_' + structure.lower() + '.npz'
            for path in dataPaths:
                if structExt in path.lower(): # If structure of interest is in item 
                    # Split the path into parts with respect to the detailed struct name 
                    pathParts = path.lower().split(structExt)
                     # Get last character of the fist split part
                    lastChar = pathParts[0][-1]
                    # Check if it is a digit and assign flag 
                    if lastChar.isdigit():
                        isHelpStructure = False
                    else:
                        isHelpStructure = True
                    # If not a help structure, assign the item to the filtered list 
                    if isHelpStructure==False:
                        filePathsFiltered.append(path) # Add to selected list
        # Return selected list
        return filePathsFiltered
        

    def getTargetVector(self, dataPaths, structureNames): 
        """
        Create a target vector for objects defined as file paths.
        Target is determined if the structure name is in the file path.
             
        Args:
            dataPaths (list(str)): List of file paths to the objects
            structureNames (list(str)): List of structure names
            
        Return:
            targetVector (list(int)): List of target vector values
        
        """
        # Init target vector
        targetVector = []
        # Loop over all file data paths, calculate the target value for each file and append to target vector
        # For every file path do: 
        for path in dataPaths:
            # Check for every structure name if the path contains the structure name
            # Set status if a target has been assigned to the file path
            # This has been evolved to be more precise, see below. 
            pathTargetSet = 0
            for struct in structureNames:
                # If the structure name is in the path
                # This was a bit blunt as for example "allbowels" is contained in "smallbowels" 
                # Therefore added demand on prefix of _ and suffix of .npz to isolate the file
                # This did not take care of Z structures, for example Z_Bladder.nii.gz, Z_Rectum.nii.gz was included when asked for _Bladder and _Rectum
                # This was solved by using isdigit on the last character before the _ + structure name 
                # This is the way the training data has been prepared. See the training data preparation script.
                # Create a more detailed struct name which helps selection 
                structExt = '_' + struct.lower() + '.npz'
                # Check first if the path contains the detailed structure name
                if structExt in path.lower(): 
                    # Split the path into parts with respect to the detailed struct name 
                    pathParts = path.lower().split(structExt)
                    # Get last character of the fist split part
                    lastChar = pathParts[0][-1]
                    # Check if it is a digit and assign flag
                    # Training data had digit as last letter. 
                    # However, test data was not constructed with real personal numbers
                    # To create draft of GT this line can be activated for such test data (hexadecimals personal numbers)
                    if lastChar.isdigit():
                    #if lastChar.isdigit() or lastChar=='a' or lastChar=='b' or lastChar=='c' or lastChar=='d' or lastChar=='e' or lastChar=='f':
                        isHelpStructure = False
                    else:
                        isHelpStructure = True
                    # If not a help structure, assign the target 
                    if isHelpStructure==False:
                        # Check pathTargetSet through exception. This checks for any conflicts in the target assignment. 
                        if pathTargetSet == 1:
                            raise Exception('File path target can not be uniquely defined in file path: ' + path)
                        # Get index of structure name and use as target value
                        idx = structureNames.index(struct)
                        target = int(idx)
                        # Append index to target vector
                        targetVector.append(target)
                        #print('File path target was assigned to class: ' + str(target) + ' for: ' + path)
                        pathTargetSet = 1

            # If target could not been defined above define the data as other class type
            # This thereby includes help structures among all other structure types 
            if pathTargetSet == 0: 
                # Assign a other class to the rest of the data 
                target = int(len(structureNames) + 1 -1) #(-1 because of indexing starting with zero)
                # Append index to target vector
                targetVector.append(target)
                #print('File path target was assigned to other class: ' + str(target) + ' for: ' + path)
                pathTargetSet = 1
            else:
                pass # Do nothing

        # Assert that all file paths have been assigned a target
        assert len(targetVector) == len(dataPaths), 'Not all file paths have been assigned a target'
        # Assert that all classes are represented in the target vector
        # np.testing.assert_array_equal(np.unique(targetVector), np.arange(0, np.max(targetVector) + 1), 'Not all classes are represented in the target vector')
        # Return target vector
        return targetVector
            
          
    def getUniquePatients(self, data_paths): 
        """
        Get a list of unique patients from the input data paths. 
        
        Args:
            data_paths (list(str)): List of data paths
            
        Return:
            uniquePatients (list(str)): List of unique patients
        
        """
        # Assert input data
        assert isinstance(data_paths, list), "Input data is not a list"
        # Get all patients from all data paths
        patients = [os.path.basename(path).split('_')[0] for path in data_paths]
        # Get unique patients
        uniquePatients = np.unique(patients).tolist()
        # Assert list
        assert isinstance(uniquePatients, list), "Output uniquePatients is not a list"
        # Return unique patients
        return uniquePatients


    def getCrossValPathData(self, data_paths, trainPatients, valPatients, useFinalTrainingAllDataBool): 
        """
        Get training and validation data paths for cross validation. 
        Selected from what patients are defined as training and validation patients.
        This elimiates the risk of information leakage if the same patient 
        is used for training and validation.
        
        Args:
            data_paths (list(str)): List of data paths
            trainPatients (list(str)): List of training patients
            valPatients (list(str)): List of validation patients
            useFinalTrainingAllDataBool (bool): If True, use all data for final training
            
        Return:
            train_data_paths (list(str)): List of training data paths
            val_data_paths (list(str)): List of validation data paths
        
        """
        # Assert input data
        assert isinstance(data_paths, list), "Input data is not a list"
        assert isinstance(trainPatients, list), "Input trainPatients is not a list"
        assert isinstance(valPatients, list), "Input valPatients is not a list"
        # Assert no patient overlap
        if useFinalTrainingAllDataBool == False:
            assert len(np.intersect1d(trainPatients, valPatients)) == 0, "Patients overlap in train and validation set"
        # Init emtpy training and validation data paths
        train_data_paths = []
        val_data_paths = []
        # Loop over all data paths
        for path in data_paths:
            # Get patient name from path
            patient = os.path.basename(path).split('_')[0]
            # If patient is in training patients, add to training data paths
            if patient in trainPatients:
                train_data_paths.append(path)
            # If patient is in validation patients, add to validation data paths
            if patient in valPatients:
                val_data_paths.append(path)    
        # Assert no overlap between training and validation data paths
        if useFinalTrainingAllDataBool == False: 
            # assert len(train_data_paths) + len(val_data_paths) == len(data_paths), 'There are missing data paths' # removed to support n number of patients
            assert len(np.intersect1d(train_data_paths, val_data_paths)) == 0, 'Overlap between training and validation data paths'
        # If used for final training
        if useFinalTrainingAllDataBool == True:
            assert len(train_data_paths) == len(val_data_paths), 'Training and validation data are different '
            assert train_data_paths == val_data_paths, 'Training and validation data are not the same  '
        # Return training and validation data paths
        return train_data_paths, val_data_paths  
        

    def getClassWeights(self, y_train, method='balanced'): 
        """
        Calulates class weights in balanced or uniform mode.
                     
        Args:
            y_train (list(str)): List of labels for training data
            
        Return:
            classWeights (dict): Dictionary with class weights
        
        """         
        from sklearn.utils import compute_class_weight
        class_weights = compute_class_weight(
                                    class_weight = method,
                                    classes = np.unique(y_train),
                                    y = y_train)                                                    
        # Return class weights
        return class_weights 



    def getTargetNaming(self, classStructures, useFullDataBool):
            """
            Create target names from structure names. 
            
            Args:
                structureNames (list(str)): List of structure names
                useFullDataBool (bool): If True add 'other' class to target names
                                                    
            Return:
                targetNames (list(str)): List of target names
                """   
            # Assert input data
            assert isinstance(useFullDataBool, bool), "Input dataPathsFull is not a bool"
            assert isinstance(classStructures, list), "Input classStructures is not a list"
            # If useFullDataBool is True add 'other' class to target names
            if useFullDataBool==True:
                targetNames = classStructures.copy() #Damn you Python for forcing copy
                targetNames.append('Other')
            elif useFullDataBool==False:
                targetNames = classStructures
            assert isinstance(targetNames, list), "Output targetNames is not a list"
            # Return targetNames
            return targetNames 


    def checkFailedObjects(self, predictions, targets, objects, targetNames):
            """
            Check which objects failed to predict correctly 
                        
            Args:
                predictions (list(int)): List of predictions
                targets (list(int)): List of targets
                objects (list(str)): List of object
                targetNames (list(str)): List of target names
                
            Return:
                failedObjects (list(str)): List of failed objects
                failedObjectsPred (list(str)): List of failed objects predictions
                failedObjectsPredName (list(str)): List of given names for failed objects predictions
              
            """  
            # Assess lists
            assert isinstance(predictions, list), "Input predictions is not a list"
            assert isinstance(targets, list), "Input targets is not a list"
            assert isinstance(objects, list), "Input objects is not a list"
            assert isinstance(targetNames, list), "Input targetNames is not a list"
            # Assess same length
            assert len(predictions) == len(targets), "Input predictions and targets are not of same length"
            assert len(predictions) == len(objects), "Input predictions and objects are not of same length"
            assert len(np.unique(targets)) <= len(targetNames)
            # Convert lists to numpy array
            predictions = np.array(predictions)
            targets = np.array(targets)
            # Get index where predictions and targets differ
            # We know its one dimensional so index 0 is ok. 
            notCorrectIndex = np.where(predictions!=targets)[0]
            # Collect all objects with wrong classification
            failedObjects = [objects[i] for i in notCorrectIndex]  
            # Remove path from the objects
            failedObjects = [os.path.basename(obj) for obj in failedObjects]
            # Collect their prediction labels
            failedObjectsPred = [predictions[i] for i in notCorrectIndex] 
            # Get prediction class name 
            failedObjectsPredName = [targetNames[predictions[i]] for i in notCorrectIndex] 
            # Return data 
            assert isinstance(failedObjects, list), "Output failedObjects is not a list"
            assert isinstance(failedObjectsPred, list), "Output failedObjectsPred is not a list"
            assert isinstance(failedObjectsPredName, list), "Output failedObjectsPredName is not a list"
            return failedObjects, failedObjectsPred, failedObjectsPredName


    def writeFailedObjects(self, epoch, failedObjects, failedObjectsPredName, filePath): 
        """ 
        Write out data line by line to a text file
        Writes the name of the failed objects and the predicted class name
        Inserts new line after each writing

        Args:
            failedObjects (list(str)): List of failed objects
            failedObjectsPredName (list(str)): List of failed objects prediction names
            filePath (str): Path to file
        
        Return:
                
        """
        # Assert input data
        assert isinstance(failedObjects, list), "Input failedObjects is not a list"
        assert isinstance(failedObjectsPredName, list), "Input failedObjectsPredName is not a list"
        assert isinstance(filePath, str), "Input filePath is not a string"
        # Display message of file written if not existing
        if os.path.exists(filePath) == 0:
            print("Created new file for failed objects")
        # Define file object (appendable writing)
        outWriteFileObject = open(filePath, "a")
        # Write epoch number
        outWriteFileObject.write('Epoch number : ' + str(epoch) + '\n')
        # For every object in the list, write out data on a new line in the file
        for index, currObject in enumerate(failedObjects): 
            # Write the name    
            outWriteFileObject.write(currObject + ' \t ' + failedObjectsPredName[index])
            # Insert new line
            outWriteFileObject.write("\n")
            # Close file write
        outWriteFileObject.close()





