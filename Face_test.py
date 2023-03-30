import face_recognition
import numpy as np
from PIL import Image, ImageDraw
from IPython.display import display
import cv2
import pickle

fares_image = face_recognition.load_image_file("Images/Mohamed fares.jpg")
obama_face_encoding = face_recognition.face_encodings(fares_image)[0]

Metwally_image = face_recognition.load_image_file("Images/Mustafa Metwally.jpg")
biden_face_encoding = face_recognition.face_encodings(Metwally_image)[0]

abdulla_image = face_recognition.load_image_file("Images/abdulla yehia.jpg")
biden_face_encoding = face_recognition.face_encodings(abdulla_image)[0]

Moemen_image = face_recognition.load_image_file("Images/Mo'men Gouda.jpg")
biden_face_encoding = face_recognition.face_encodings(Moemen_image)[0]


known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding
]
known_face_names = [
    "fares",
    "Metwally",
    "Abdulla",
    "Moemen"
]
print('Learned encoding for', len(known_face_encodings), 'images.')


data = pickle.loads(open('face_enc', "rb").read())
known_face_encodings = data['encodings']
known_face_names = data['names']

unknown_image = face_recognition.load_image_file("test4.jpg")

face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

pil_image = Image.fromarray(unknown_image)

draw = ImageDraw.Draw(pil_image)


for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

    name = "Unknown"

    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]

    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))


    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 55))
    draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

del draw


pil_image.show()
cv2.waitKey(0)