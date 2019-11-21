import sys,os
import string
import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageio
import dist2 as dist2
import math
from random import sample
from sklearn.cluster import KMeans
##%matplotlib inline

#inputs
perOfTotalDesc = 0.8
k = 500

#functions:
def convertFileNameToPicPath(name):
    buf = name.split("_")
    seqNum = buf[0]
    SubNum = buf[1]
    picPath = os.path.join(imageRoot, seqNum, SubNum, name.replace('_emotion.txt','.png'))
    return picPath

#load image
#img = loadIm(im)
def loadIm(imName):
     img = cv2.imread(imName,0)
     test_image = img.copy()
     return test_image
     #plt.imshow(test_image)

#crop face from image in resized format
#img = cropFace(headShot)
def cropFace(headShot):
     haar_cascade_face = cv2.CascadeClassifier('C:/Users/sanme/Documents/CVproject-master/CVproject-master/data/haarcascades/haarcascade_frontalface_default.xml')
     faces_rects = haar_cascade_face.detectMultiScale(headShot, scaleFactor = 1.2, minNeighbors = 5);

     for (x,y,w,h) in faces_rects:
          cv2.rectangle(headShot, (x, y), (x+w, y+h), (255, 0, 0), 2)
     # plt.imshow(headShot)
     # plt.show()
##     print(faces_rects)
##     [x,y,w,h] = faces_rects[0]
     face_cropped = headShot[y:y+h, x:x+w]
     return face_cropped

#get keypoints and descriptors using SIFT
#kp, des, imgKpDrawn = siftOut(img)
def siftOut(img):
     sift = cv2.xfeatures2d.SIFT_create()
     kp, des = sift.detectAndCompute(img, None)
     imgKpDrawn=cv2.drawKeypoints(img,kp,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
     return kp, des, imgKpDrawn

#do kmeans clustering and get histogram of given images
#find correlation between datasets using 

#main
if __name__ == '__main__':

    #loop through set of images to get kmeans centers and labels
    #We can use labels themselves to get histogram, but if we want to recalculate them without
    #clustering again, use centers
    emotionsList = ["anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]

#####################################################################################
#GENERATE KMEANS CLUSTER CENTERS
#####################################################################################
##    totalDesc = np.zeros((1,128))
##    for emotion in emotionsList:
##        print('reading from emotions file: ', emotion)
##        with open("SanmeshEmotionsPathFiles/"+ emotion+".txt","r+") as emotionFile:
##            for line in emotionFile:
##                # load that file
##                path = line
##                test_image = loadIm(path.strip())
##
####                if emotion == "surprise":
####                    print(path)
##                
##                #crop faces from the images using haar_cascade classifier
##                face_cropped = cropFace(test_image)
##                
##                # Resize image to 256 x 256
##                face_cropped = cv2.resize(face_cropped, (256,256))
##
##                # SIFT
##                kp, des, imgKpDrawn = siftOut(face_cropped)
##
##                #get random descriptors
##                randomDescIndex = sample(list(range(des.shape[0])), int(perOfTotalDesc*des.shape[0]))
##                randomDesc = des[randomDescIndex,:]    
##                #append
##                totalDesc = np.append(totalDesc, randomDesc,axis=0)
##    totalDesc = totalDesc[1:,:]
##    np.save("totalDesc", totalDesc)
##    print("sizeOfAllDesc = ", totalDesc.shape)
##
##    kmeans = KMeans(n_clusters=k, random_state=0).fit(totalDesc)
##    np.save("kmeansLabels", kmeans.labels_)
##    np.save("kmeansClusterCenters", kmeans.cluster_centers_)

#####################################################################################
#GENERATE HISTOGRAMS
#####################################################################################
    #load cluster centers
    clusterCent = np.load('kmeansClusterCenters.npy')



    for emotion in emotionsList:
        print('reading from emotions file: ', emotion)
        with open("SanmeshEmotionsPathFiles/"+ emotion+".txt","r+") as emotionFile:
            for line in emotionFile:
                # load that file
                path = line
                test_image = loadIm(path.strip())
                
                #crop faces from the images using haar_cascade classifier
                face_cropped = cropFace(test_image)
                
                # Resize image to 256 x 256
                face_cropped = cv2.resize(face_cropped, (256,256))

                # SIFT
                kp, des, imgKpDrawn = siftOut(face_cropped)

                pairwiseDist = dist2.dist2(des,clusterCent)
                labelsBuf= np.argmin(pairwiseDist, axis=1)
                unique1, counts1 = np.unique(labelsBuf, return_counts=True)
                hist1 = np.zeros(clusterCent.shape[0])
                for j in range(unique1.shape[0]):
                    hist1[unique1[j]] = counts1[j]
                identifier = emotion+"Histograms/"+emotion+"_"+ line[-22:-5]
                np.save(identifier, hist1)

    for emotion in emotionsList:
        print('reading from emotions file: ', emotion)
        with open("SanmeshEmotionsPathFiles/"+ emotion+"Test.txt","r+") as emotionFile:
            for line in emotionFile:
                # load that file
                path = line
                test_image = loadIm(path.strip())
                
                #crop faces from the images using haar_cascade classifier
                face_cropped = cropFace(test_image)
                
                # Resize image to 256 x 256
                face_cropped = cv2.resize(face_cropped, (256,256))

                # SIFT
                kp, des, imgKpDrawn = siftOut(face_cropped)

                pairwiseDist = dist2.dist2(des,clusterCent)
                labelsBuf= np.argmin(pairwiseDist, axis=1)
                unique1, counts1 = np.unique(labelsBuf, return_counts=True)
                hist1 = np.zeros(clusterCent.shape[0])
                for j in range(unique1.shape[0]):
                    hist1[unique1[j]] = counts1[j]
                identifier = emotion+"HistogramsTest/"+emotion+"_"+ line[-22:-5]
                np.save(identifier, hist1)
    plt.show()
