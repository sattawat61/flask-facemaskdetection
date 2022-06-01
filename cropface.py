import cv2
import os

face_cascade = cv2.CascadeClassifier('C:/Users/tueku/Downloads/python-opencv-detect-master/python-opencv-detect-master/haarcascade_frontalface_alt.xml')
eyePair_cascade = cv2.CascadeClassifier('C:/Users/tueku/Downloads/opencv-master/opencv-master/data/haarcascades/haarcascade_mcs_eyepair_big.xml')

def return_eye_pair(infile, outfile):
	img = cv2.imread(infile)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	if len(faces) == 0: return
	for x,y,w,h in faces:
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
		eyes = eyePair_cascade.detectMultiScale(roi_gray)	
		if len(eyes) == 0: return	
		for (ex,ey,ew,eh) in eyes:
			eyes_roi = roi_color[ey: ey+eh, ex:ex + ew]
			#cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

	cv2.imwrite(outfile, eyes_roi)
	
def main():
    return_eye_pair('Sattawat_Tusumran_002.png', 'eye_pair.jpg') 
 
if __name__ == '__main__':
    main()