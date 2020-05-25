import sys,os
import string
import numpy as np
import cv2
import matplotlib.pyplot as plt
percOfImagesTaken = 0.3
percTest = 0.25
#emotionRoot = os.path.abspath("Emotion_labels/Emotion").replace('PythonApplication1\\','')
#imageRoot = os.path.abspath("cohn-kanade-images").replace('PythonApplication1\\','')
emotionRoot = "C:/Users/sanme/Documents/CVproject-master/CVproject-master/Emotion_labels/Emotion"
imageRoot = "C:/Users/sanme/Documents/CVproject-master/CVproject-master/cohn-kanade-images"
def convertFileNameToPicPath(name):
    buf = name.split("_")
    seqNum = buf[0]
    SubNum = buf[1]
    picPath = os.path.join(imageRoot, seqNum, SubNum, name.replace('_emotion.txt','.png'))
    return picPath
def loadIm(imName):
     img = cv2.imread(imName,0)
     test_image = img.copy()
     return test_image
     #plt.imshow(test_image)

angerList = []
contemptList = []
disgustList = []
fearList = []
happyList = []
sadnessList = []
surpriseList = []
for path, subdirs, files in os.walk(emotionRoot):
    for name in files:
##        if (len(happyList) > 20) and (len(sadnessList) > 20):
##            print(len(happyList))
##            break
##        print("name: ", name)
        filePath = os.path.join(path, name)
        f = open(filePath, "r")
        emotionLabel = f.readline()
        emotionLabel = emotionLabel.translate({ord(c): None for c in string.whitespace})
        emotionLabel = int(emotionLabel[0])

        picPath = convertFileNameToPicPath(name)
        numberOfImages = int(int((picPath.split('_')[-1])[0:-4])*percOfImagesTaken)
        for i in range(numberOfImages):
            num = int((picPath.split('_')[-1])[0:-4])-i
            num = str(num).zfill(8)
            newPath = picPath[0:-12] + num+'.png'
            angerList.append(newPath)
##            print(newPath)
            if emotionLabel == 1: #anger
##                picPath = convertFileNameToPicPath(name)
                angerList.append(newPath)
            elif emotionLabel == 2: #contempt
##                picPath = convertFileNameToPicPath(name)
                contemptList.append(newPath)
            elif emotionLabel == 3: #disgust
##                picPath = convertFileNameToPicPath(name)
                disgustList.append(newPath)
            elif emotionLabel == 4: #fear
##                picPath = convertFileNameToPicPath(name)
                fearList.append(newPath)
            elif emotionLabel == 5: #happy
##                picPath = convertFileNameToPicPath(name)
                happyList.append(newPath)
            elif emotionLabel == 6: #sadness
##                picPath = convertFileNameToPicPath(name)
                sadnessList.append(newPath)
            elif emotionLabel == 7: #surprise
##                picPath = convertFileNameToPicPath(name)
                surpriseList.append(newPath)
        f.close()
##    if (len(happyList) > 20) and (len(sadnessList) > 20):
##        break
file1 = open("anger.txt","w")
file2 = open("angerTest.txt","w")
testNum = int(len(angerList)*percTest)
print("len(angerList)", angerList)
print("testNum", testNum)
for i in range(len(angerList)):
    if i < testNum:
        file2.write(angerList[i])
    else:
        file1.write(angerList[i])
    file1.write('\n')
file1.close() #to change file access modes
file2.close() #to change file access modes
file1 = open("contempt.txt","w")
for element in contemptList:
     file1.write(element)
     file1.write('\n')
file1.close() #to change file access modes
file1 = open("disgust.txt","w")
for element in disgustList:
     file1.write(element)
     file1.write('\n')
file1.close() #to change file access modes
file1 = open("fear.txt","w")
for element in fearList:
     file1.write(element)
     file1.write('\n')
file1.close() #to change file access modes
file1 = open("happy.txt","w")
for element in happyList:
     file1.write(element)
     file1.write('\n')
file1.close() #to change file access modes
file1 = open("sadness.txt","w")
for element in sadnessList:
     file1.write(element)
     file1.write('\n')
file1.close() #to change file access modes
file1 = open("surprise.txt","w")
for element in surpriseList:
     file1.write(element)
     file1.write('\n')
file1.close() #to change file access modes


##print("happyList")
##print(happyList)
##print("sadnessList")
##print(sadnessList)
##happyIm = loadIm(happyList[0])
##plt.figure(1)
##plt.imshow(happyIm)
##sadnessIm = loadIm(sadnessList[0])
##plt.figure(2)
##plt.imshow(sadnessIm)
##plt.show()
