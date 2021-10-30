import pandas as pd
import cv2
import matplotlib.pyplot as plt


class myDataset:
    def __init__(self) -> None:
        # Initialize the dateset by loading a label file to a dataframe
        self.allData = pd.read_csv('../ImageList_CEE4803.csv')
        self.imagePath = self.allData['Image_Path']
        self.imageLabel = self.allData['Rating']

    def getImageData(self, index):
        
        # Fetch a pavement image as grayscale from the data folder based on index.

        listOfImagePath = self.imagePath
        img = cv2.imread(listOfImagePath[index], 0)
        return img


    def showImageByIndex(self, index, predRating = None):
        
        # Display a pavement image and its raveling condition rating by specified index.

        listOfImageRating = self.imageLabel
        plt.figure(figsize=(30, 15))

        img = self.getImageData(index)

        plt.imshow(img, cmap='gray')

        if predRating is None: # if the predicted rating is not given, show only the labeled rating
            plt.title('Index: ' + str(index) +
                    ', Rating: ' + listOfImageRating[index])
        else: # if the predicted rating is given, show both ratings
            plt.title('Index: ' + str(index) +
                      ', Pred. Rating: ' + str(predRating) +
                      ', Actual Rating: ' + listOfImageRating[index])
        plt.show()


    def getUniqueRatings(self):
        
        # Return the unique ratings in the dataset.

        listOfImageRating = self.imageLabel
        uniqueRatings = listOfImageRating.unique()
        return uniqueRatings


    def plotRatingDistribution(self):

        self.imageLabel.value_counts(sort=False).plot.bar()
        plt.title('Rating Distribution')
        plt.show()

    def showImageByRating(self, rating):

        if not rating in self.getUniqueRatings():
            raise Exception("Requested rating does not exist in the dataset!")
        listOfImageRating = self.imageLabel
        listOfImagePath = self.imagePath

        index = listOfImagePath[listOfImageRating == rating].index[0]
        self.showImageByIndex(index)
 
