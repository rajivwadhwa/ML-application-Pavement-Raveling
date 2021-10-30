import numpy as np
import pandas as pd
from tqdm import tqdm
from dataset_class import myDataset
import matplotlib.pyplot as plt
from itertools import cycle
from skimage.util import crop, view_as_blocks
from scipy.stats import scoreatpercentile as prctile


class featureExtractor:
    def __init__(self, dataset: myDataset, hSize=75, wSize=75, numOfCDFSamples=50):
        """ initialize featureExtractor, a dataset object will be the input for the extractor to use"""
        self.dataset = dataset  # Add a myDataset object as a property of the featureExtractor

        # Parameter related to patch level feature extraction
        self.hSize = hSize  # Height of image patches (default is 75)
        self.wSize = wSize  # Width of image patches (default is 75)
        # create a list of percentile values to sample on the CDF.
        self.listOfPercentiles = np.linspace(0, 100, num=int(numOfCDFSamples))


    def getFlattenedImage(self, index):

        array_image = self.dataset.getImageData(index)
        
        # flatten the image
        flaten_image = array_image.reshape(1,-1)
        return flaten_image
 

    def getImagePatches(self, index):

        image = self.dataset.getImageData(index)  # read image data using index

        hSize = self.hSize  # height of each patch
        wSize = self.wSize  # width of each patch

        # figure out how much the image should be cropped so it can evenly divided by the patch size
        cropSizeH = image.shape[0] % hSize
        cropSizeW = image.shape[1] % wSize
        topCropSize = int(np.floor(cropSizeH/2))
        bottomCropSize = int(np.ceil(cropSizeH/2))
        leftCropSize = int(np.floor(cropSizeW/2))
        rightCropSize = int(np.ceil(cropSizeW/2))

        # crop and get the patch of the image
        imageCropped = crop(
            image, ((topCropSize, bottomCropSize), (leftCropSize, rightCropSize)))
        imagePatches = view_as_blocks(imageCropped, (hSize, wSize))

        # reshape the image patches so the output will have a size of (number of patches x patch height x patch width)
        imagePatches = imagePatches.reshape((-1, imagePatches.shape[2], imagePatches.shape[3]))
        return imagePatches

    def getFlattenedPatches(self, index):

        imagePatches = self.getImagePatches(index)  # get image patches
        flattenedPatch = imagePatches.reshape((imagePatches.shape[0], -1))  # flatten the image patchese
        return flattenedPatch

    def computeTextureDescriptor(self, flattenedArray: np.ndarray, descriptor: str):

        # The code below shows an example of computing texture descriptor (mean)
        # Mean signifies the brightness of range image- Which in turn gives us information about elevation
        # Std Dev gives the uniformity of the given region 
        
        if descriptor == 'mean': 
            descriptorValues = np.mean(flattenedArray, axis=1)

        elif descriptor == 'std':  
            descriptorValues = np.std(flattenedArray, axis=1)


        return descriptorValues

    def getCDFFeatures(self, descriptorValues, listOfPercentiles):

        # 25th, 50th, 75th precentile to get CDF features
        CDF = prctile(descriptorValues, listOfPercentiles)
        return CDF


    def extractFeatures(self, listOfIndices, imageFeature=True, patchFeature=True):

        allFeatures = None  
        allFeatures_std = None
        
        for index in tqdm(listOfIndices):

            allImageFeatures = []  
            #allImageFeatures_std = [] 
            if imageFeature:  # if imageFeature flag is True, extract image level features
                flattenedImage = self.getFlattenedImage(index) 

                # extract features using mean as the texture descriptor
                imageLevelFeatures = self.computeTextureDescriptor(flattenedImage, descriptor='mean')
                # append imageLevelFeatures to allImageFeatures using np.hstack(), hstack function append values in row direction
                allImageFeatures = np.hstack((allImageFeatures, imageLevelFeatures))

                # extract features using standard deviation as the texture descriptor
                imageLevelFeatures = self.computeTextureDescriptor(flattenedImage, descriptor='std')
                # append imageLevelFeatures to allImageFeatures using np.hstack(), hstack function append values in row direction
                allImageFeatures = np.hstack((allImageFeatures, imageLevelFeatures))

            allPatchFeatures = [] 
            #allPatchFeatures_std = []
            
            if patchFeature:  # if patchFeature flag is True, extract patch level features
                flattenedPatches = self.getFlattenedPatches(index)  

                # extract texture descriptor values using mean
                descriptorValues = self.computeTextureDescriptor(flattenedPatches, descriptor='mean')
                patchLevelFeatures = self.getCDFFeatures(descriptorValues, self.listOfPercentiles)
                allPatchFeatures = np.hstack((allPatchFeatures, patchLevelFeatures))

 
                descriptorValues = self.computeTextureDescriptor(flattenedPatches, descriptor='std')
                # extract CDF features to represent the distribution of descriptor values
                patchLevelFeatures = self.getCDFFeatures(descriptorValues, self.listOfPercentiles)
                allPatchFeatures = np.hstack((allPatchFeatures, patchLevelFeatures))


            # store features extracted from the image to allFeatures
            if allFeatures is None:  
                allFeatures = np.hstack((allImageFeatures, allPatchFeatures))
            else:
                allFeatures = np.vstack((allFeatures, np.hstack((allImageFeatures, allPatchFeatures))))     
        return pd.DataFrame(allFeatures) 
        
        
    def extractLabels(self, listOfIndices):

        labelList = []
        for i in listOfIndices:
            temp = self.dataset.imageLabel[i]
            labelList.append(temp)
        return np.array(labelList)
        
 
    def getAverageFeatureCDF(self, listOfIndices, descriptor='mean'):

        # the list of precentile will be from 0 - 99
        listOfPercentiles = list(range(100))

        # Get the list of ratings and the unique ratings in the list
        listOfRating = self.extractLabels(listOfIndices)
        uniqueRatings = np.unique(listOfRating)

        averageFeatureCDF = None  
        # for every unique rating, compute the average CDF feature values
        for rating in uniqueRatings:
            listOfIndices_np = np.array(listOfIndices)
            ratingIndices = listOfIndices_np[listOfRating == rating]

            allPatchFeatures = None  

            for index in tqdm(ratingIndices):
                # get flattened patches
                flattenedPatches = self.getFlattenedPatches(index)
                # extract texture descriptor values
                descriptorValues = self.computeTextureDescriptor(flattenedPatches, descriptor=descriptor)
                # extract CDF features to represent the distribution of descriptor values
                patchLevelFeatures = self.getCDFFeatures(descriptorValues, listOfPercentiles)

                if allPatchFeatures is None:  
                    allPatchFeatures = patchLevelFeatures
                else:
                    allPatchFeatures = np.vstack((allPatchFeatures, patchLevelFeatures))

            # Compute average values of "allPatchFeatures"
            averagePatchFeatures = np.mean(allPatchFeatures, axis=0)

            if averageFeatureCDF is None: 
                averageFeatureCDF = averagePatchFeatures
            else:
                averageFeatureCDF = np.vstack((averageFeatureCDF, averagePatchFeatures))

        return averageFeatureCDF

    def plotFeatureCDF(self, listOfIndices, descriptor='mean'):

        # get uniques ratings
        listOfRating = self.extractLabels(listOfIndices)
        uniqueRatings = np.unique(listOfRating)

        # get average features values of each uniques rating as a CDF
        averageFeatureCDF = self.getAverageFeatureCDF(listOfIndices, descriptor)

        # plot the average CDF values
        lines = ["-", "--", "-.", ":"]
        linecycler = cycle(lines)
        plt.figure(figsize=(10, 6))
        for (idx, rating) in enumerate(uniqueRatings):
            plt.plot(averageFeatureCDF[idx, :], list(
                range(100)), next(linecycler), label=rating)
        plt.legend(fontsize=15)
        plt.xlabel(descriptor, fontsize=15)
        plt.ylabel('Percentile', fontsize=15)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
