import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#Load cat image
cats = []
for filename in os.listdir('Training Data/Cat'):
    if(filename.endswith('.jpg')):
        img = cv2.imread(f'Training Data/Cat/{filename}')
        if img is not None:
            img  = cv2.resize(img,(16,16))
            cats.append(img.flatten()/255.0)

#Load dog image
dogs = []
for filename in os.listdir('Training Data/Dog'):
    if(filename.endswith('.jpg')):
        img = cv2.imread(f'Training Data/Dog/{filename}')
        if img is not None:
            img = cv2.resize(img, (16,16))
            dogs.append(img.flatten()/255.0)


#Create trainging data
x_train = np.array(cats+dogs)
y_train = np.array([0]*len(cats) + [1]*len(dogs))

#Check the shape of the training data
print(x_train.shape)
print(y_train.shape)


#SVM using scikit learn
svm = SVC(kernel='rbf', random_state=42)
svm.fit(x_train, y_train)

#Checking training accuracy
prediction = svm.predict(x_train)
accuracy = accuracy_score(y_train, prediction)
print(f"Accuracy: {accuracy}")

#prediction of test data
#Data is in testData folder
for filename in sorted(os.listdir('TestData')):
    if(filename.endswith('.jpg')):
        img = cv2.imread(f'TestData/{filename}')
        if img is not None:
            img = cv2.resize(img,(16,16))
            x_test = img.flatten()/255.0

            #SVM
            prediction = svm.predict([x_test])[0]

            #print
            if(prediction == 0 ):
                print(f"{filename}: Cat")
            else:
                print(f"{filename}: Dog")
