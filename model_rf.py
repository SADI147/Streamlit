#Importing Necessary Libraries

import cv2
import pandas as pd
import os
import pickle 
import numpy as np

#Finding Length of the directories
list_stable = os.listdir('stable')
num_st_img = len(list_stable)+1

list_unstable = os.listdir('unstable')
num_us_imgs = len(list_unstable)+1

list_check = os.listdir('check')
num_ch_img = len(list_check)+1

#Creating Dataframes to store necessay attributes
data_unstable = pd.DataFrame(columns=["Mx", "My", "Height", "Width", "Status"])
data_stable = pd.DataFrame(columns=["Mx", "My", "Height", "Width", "Status"])
data_check = pd.DataFrame(columns=["Mx", "My", "Height", "Width"])

#Generate Data for Stable Images
# Load image, convert to grayscale, and Otsu's threshold 
for i in range(1,num_st_img,1):
    image = cv2.imread('stable/'+str(i)+".png")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Find contours and extract the bounding rectangle coordintes
    # then find moments to obtain the centroid
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        # Obtain bounding box coordinates and draw rectangle
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
        print(h)
        print(w)
    
        # Find center coordinate and draw center point
        M = cv2.moments(c)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv2.circle(image, (cx, cy), 2, (36,255,12), -1)
        print('Center: ({}, {})'.format(cx,cy))
        data_stable = data_stable.append({"Mx":cx, "My":cy, "Height":h, "Width":w, "Status":1}, 
                                         ignore_index = True)

#Generate Data for Unstable Images 
for i in range(1,num_us_imgs,1):
    image = cv2.imread('unstable/'+str(i)+".png")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Find contours and extract the bounding rectangle coordintes
    # then find moments to obtain the centroid
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        # Obtain bounding box coordinates and draw rectangle
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
        print(h)
        print(w)
    
        # Find center coordinate and draw center point
        M = cv2.moments(c)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv2.circle(image, (cx, cy), 2, (36,255,12), -1)
        print('Center: ({}, {})'.format(cx,cy))
        data_unstable = data_unstable.append({"Mx":cx, "My":cy, "Height":h, "Width":w, "Status":0}, 
                                         ignore_index = True)

#Lets Merge Both DataSets
df = [data_stable, data_unstable]
final = pd.concat(df)
#Shuffle Dataset to get a better split
from sklearn.utils import shuffle
final =shuffle(final)
final.to_csv('Dataset/Dataset.csv')

#Split Dataset to X(Independent Variable) & Y(Dependent Variable)
X = final.iloc[:, 0:4]
Y = final.iloc[:, 4:5]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

#Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

y_train=y_train.astype('float64')
y_test=y_test.astype('float64')

np.savetxt('Train Labels.csv', X_train, fmt = '%.3f', delimiter=',', header="'Center_x', 'Center_Y','Height', 'Width'")
np.savetxt('Train Classes.csv', y_train, fmt = '%.3f', delimiter=',', header="Status")
np.savetxt('Test Labels.csv', X_test, fmt = '%.3f', delimiter=',', header="'Center_x', 'Center_Y','Height', 'Width'")
np.savetxt('Test Classes.csv', y_test, fmt = '%.3f', delimiter=',', header="Status")

#Fitting Random Forest Classifier to Training Set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 5, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train.values.ravel())


## Predicting the Test set results
y_pred = classifier.predict(X_test)
#

## Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#
#from sklearn import metrics
#report = metrics.classification_report(y_test, y_pred, output_dict = 'True')
#report_save = pd.DataFrame(report).transpose()
#report_save.to_csv('Classification Report.csv')
#
#from yellowbrick.classifier import ROCAUC
#visualizer = ROCAUC(classifier, classes=["Unstable", "Stable"])
#
#visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
#visualizer.score(X_test, y_test)        # Evaluate the model on the test data
#visualizer.show()                       # Finalize and show the figure

#Generate Data for different images
#for i in range(1,num_ch_img,1):
#    image = cv2.imread("check/" + str(i) + ".png")
#    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
#    
#    # Find contours and extract the bounding rectangle coordintes
#    # then find moments to obtain the centroid
#    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
#    for c in cnts:
#        # Obtain bounding box coordinates and draw rectangle
#        x,y,w,h = cv2.boundingRect(c)
#        cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
#        print(h)
#        print(w)
#    
#        # Find center coordinate and draw center point
#        M = cv2.moments(c)
#        cx = int(M['m10']/M['m00'])
#        cy = int(M['m01']/M['m00'])
#        cv2.circle(image, (cx, cy), 2, (36,255,12), -1)
#        print('Center: ({}, {})'.format(cx,cy))
#        data_check = data_check.append({"Mx":cx, "My":cy, "Height":h, "Width":w}, 
#                                             ignore_index = True)
#        data_check.to_csv('Test Dataset.csv')
#
#data_check = sc.transform(data_check)
#result = classifier.predict(data_check)


pickle.dump(classifier, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))
