import numpy as np
import os
import cv2 as cv

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']  # Make sure names match folder names exactly
DIR = r'C:\Users\siddhi\Desktop\VIT\dum\faceDetecttion\Faces\train'

haar_cascade = cv.CascadeClassifier('haar_face.xml')

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv.imread(img_path)
            if img_array is None:
                continue 
                
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)
                # cv.imshow(f"face {x}", faces_roi)  # Shows only the detected face region
                # cv.imshow(f"face {x}", img_array)  # Shows the entire image
                # print(f'{features}')
                # print(f'{labels}')
                
create_train()
print('Training done ---------------')
print(f'Length of features = {len(features)}')
print(f'Length of lables = {len(labels)}')
cv.waitKey(0)


features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the Recognizer on the features list and the labels list
face_recognizer.train(features,labels)

face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)
