from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import numpy as np
import os, glob

# basedir = './SanmeshCode/k500/'
# basedir = './SanmeshCode/k500_rep/'
basedir = './SanmeshCode/k500_moretrain/'
# basedir = './SanmeshCode/k1000/'
trainingDir = ['angerHistograms', 'contemptHistograms', 'disgustHistograms', 'fearHistograms', 'happyHistograms', 'sadnessHistograms', 'surpriseHistograms']
testDir = ['angerHistogramsTest', 'contemptHistogramsTest', 'disgustHistogramsTest', 'fearHistogramsTest', 'happyHistogramsTest', 'sadnessHistogramsTest', 'surpriseHistogramsTest']
# For k1000 and k500
# trainingLabels = np.zeros((259))
# testLabels = np.zeros((67))

# For k500_moretrain (no augmentation but 90/10 split)
trainingLabels = np.zeros((274))
testLabels = np.zeros((52))

# For k500 (repetition augmentation)
# trainingLabels = np.zeros((343))
# testLabels = np.zeros((87))
trainingData = np.array([])
testData = np.array([])
dataFillCount = 0
labelCount = 0
for emotion in range(len(trainingDir)):
    subdirname = basedir + trainingDir[emotion]
    histfilenames = glob.glob(subdirname + '/' + '*.npy')

    for hist in histfilenames:
        curData = np.load(hist)
        if dataFillCount == 0:
            trainingData = curData
            dataFillCount += 1
        elif dataFillCount == 1:
            trainingData = np.stack((trainingData, curData))
            dataFillCount += 1
        else:
            trainingData = np.concatenate((trainingData, curData[np.newaxis,:]), axis=0)
        trainingLabels[labelCount] = emotion
        labelCount += 1
        
dataFillCount = 0
labelCount = 0
for emotion in range(len(testDir)):
    subdirname = basedir + testDir[emotion]
    histfilenames = glob.glob(subdirname + '/' + '*.npy')
    for hist in histfilenames:
        curData = np.load(hist)
        if dataFillCount == 0:
            testData = curData
            dataFillCount += 1
        elif dataFillCount == 1:
            testData = np.stack((testData, curData))
            dataFillCount += 1
        else:
            testData = np.concatenate((testData, curData[np.newaxis,:]), axis=0)   
        testLabels[labelCount] = emotion
        labelCount += 1
            
# print(trainingData.shape)
# print(testData.shape)
# print(trainingLabels.shape)
# print(testLabels.shape)

# Train SVM
clf = SVC(gamma='auto')
clf.fit(trainingData, trainingLabels)
prediction = clf.predict(testData)
confMat = confusion_matrix(testLabels, prediction)
print(trainingDir)
print(confMat)
accuracy = prediction - testLabels
correct = accuracy[accuracy==0]
print("Accuracy: ", correct.shape[0]/accuracy.shape[0])
