import face_recognition
import imutils
import pickle
import time
import cv2
import os

import numpy as np
 

cascPathface = os.path.dirname(
 cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")

data = pickle.loads(open('face_enc', "rb").read())

print(data)

image = cv2.imread('face.jpg')

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(gray,
                                     scaleFactor=1.1,
                                     minNeighbors=4,
                                     minSize=(60, 60),
                                     )
print(faces)

encodings = face_recognition.face_encodings(rgb)
names = []
filtered_faces =[]

for encoding in encodings:

    best =10000
    name = 'none'
    

    matches = face_recognition.compare_faces(data["encodings"],
    encoding)

    print(f' matches {matches}')
    if True in matches :

        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
        print(f' matched indses {matchedIdxs}')
        counts = {}

        f = ''
        for i in matchedIdxs:

            name = data["names"][matchedIdxs[0]]

            counts[name] = counts.get(name, 0) + 1

            name = max(counts, key=counts.get)

            name = data["names"][matchedIdxs.index(i)]

        names.append(name)

        for ((x, y, w, h), name) in zip(faces, names):

                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 255, 0), 2)
        print(f' data {data["names"]}')

cv2.imshow("Frame", image)
cv2.waitKey(0)