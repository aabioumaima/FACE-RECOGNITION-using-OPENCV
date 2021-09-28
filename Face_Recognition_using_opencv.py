import cv2
import numpy as np
import face_recognition

imgAri = face_recognition.load_image_file('ImagesBasic/ariana.png')
imgAri = cv2.cvtColor(imgAri,cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('ImagesBasic/ari.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

imgDua = face_recognition.load_image_file('ImagesBasic/dua.jpg')
imgDua = cv2.cvtColor(imgDua,cv2.COLOR_BGR2RGB)


#Detecte the location of Ariana's face:

faceLoc = face_recognition.face_locations(imgAri)[0]
encodeAri = face_recognition.face_encodings(imgAri)[0]
cv2.rectangle(imgAri,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

#Detecte the location of Ariana's face second image:

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeAriTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

#Detecte the location of Dua's face:

faceLocTestDua = face_recognition.face_locations(imgDua)[0]
encodeDuaTest = face_recognition.face_encodings(imgDua)[0]
cv2.rectangle(imgDua,(faceLocTestDua[3],faceLocTestDua[0]),(faceLocTestDua[1],faceLocTestDua[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeAri], encodeAriTest)
faceDis = face_recognition.face_distance([encodeAri], encodeAriTest)

results2 = face_recognition.compare_faces([encodeAri], encodeDuaTest)
faceDis2 = face_recognition.face_distance([encodeAri], encodeDuaTest)
print(results2, faceDis2)
cv2.putText(imgDua, f'{results2} {round(faceDis2[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)


cv2.imshow('Ariana Grande', imgAri)
cv2.imshow('Ariana Test', imgTest)
cv2.imshow('Dua Lipa', imgDua)
cv2.waitKey(0)
cv2.destroyWindow()


