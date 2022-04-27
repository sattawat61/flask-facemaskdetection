# from turtle import right
from PIL import Image, ImageDraw
import face_recognition
import numpy as np
import pickle

known_face_names, known_face_encodings = pickle.load(open('faces.p','rb'))

print(known_face_names)
# print(known_face_encodings)

image = Image.open('testpeople/rose.jpg')

face_locations = face_recognition.face_locations(np.array(image))
face_encodings = face_recognition.face_encodings(np.array(image), face_locations)
# print(face_locations)
# print(face_encodings)

draw = ImageDraw.Draw(image)

for face_encoding , face_location in zip(face_encodings, face_locations):
    face_distances = face_recognition.face_distance(known_face_encodings,face_encoding)
    best_match_index = np.argmin(face_distances)
    print(face_distances)
    print(best_match_index)
    #########
    # name = known_face_names[best_match_index]
    # top, right, bottom, left = face_location
    # print(best_match_index)
    # draw.rectangle([left,top,right,bottom])
    # draw.text((left,top), name)
    #######
    if (face_distances < 1).all():
            name = known_face_names[best_match_index]
            top, right, bottom, left = face_location
            draw.rectangle([left,top,right,bottom])
            draw.text((left,top), name)
    else:
        name = "unknow"
        top, right, bottom, left = face_location
        draw.rectangle([left,top,right,bottom])
        draw.text((left,top), name)
    # print(face_distances)
    # print(name) 
image.show()