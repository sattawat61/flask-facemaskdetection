#ตัวทอดลอง face-recognition 28/4/2022!!!!!!!!!
import imp
from PIL import Image, ImageDraw
import face_recognition
import numpy as np
import pickle
import os
import glob
import cv2
import sys

# datatext = []
# def createreport():
#     # print('This is standard output', file=sys.stdout)
#     # known_face_names, known_face_encodings = pickle.load(open('faces.p','rb'))
#     path = glob.glob("D:/project/test1/Flaskmyweb/testpeople/*.jpg")
#     for file in path:
#         known_face_names, known_face_encodings = pickle.load(open('faces.p','rb'))
        
#         image = Image.open(file)
#         face_locations = face_recognition.face_locations(np.array(image))
#         face_encodings = face_recognition.face_encodings(np.array(image), face_locations)
#         draw = ImageDraw.Draw(image)
#         # print(file)
#         # แสดงรูปที่อ่านได้จากในpath
#         # img = cv2.imread(file)
#         # cv2.imshow("img",img)
#         # cv2.waitKey(0)
#         # cv2.destroyAllWindows()
#         for face_encoding , face_location in zip(face_encodings, face_locations):
#             face_distances = face_recognition.face_distance(known_face_encodings,face_encoding)
#             best_match_index = np.argmin(face_distances)
#             print(face_distances)
#             print(best_match_index)
#             if (face_distances < 0.8).any():
#                     name = known_face_names[best_match_index]
#                     top, right, bottom, left = face_location
#                     draw.rectangle([left,top,right,bottom])
#                     draw.text((left,top), name)
            
#             else:
#                 name = "unknow"
#                 top, right, bottom, left = face_location
#                 draw.rectangle([left,top,right,bottom])
#                 draw.text((left,top), name)
#             # print(face_distances)
#             # print(name)
#     print(name)
#     # image.show()
