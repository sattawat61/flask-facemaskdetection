import os
import tensorflow as tf
import numpy as np
import cv2
import time
import datetime
from os import times_result
import imutils


try:
    log = open('log.txt',"w")
except:
    print( "No log")
def testIntersectionIn(x, y):
    res = -450 * x + 400 * y + 157500
    if ((res >= -550) and (res < 550)):
        # print(str(res))
        return True
    return False
def testIntersectionOut(x, y):
    res = -450 * x + 400 * y + 180000
    if ((res >= -550) and (res <= 550)):
        # print(str(res))
        return True
    return False

face_mask = ['Masked', 'No mask']
size = 224

# Load face detection and face mask model
path = r'D:/project/masknew/face_mask.model/saved_model.pb'
model = tf.keras.models.load_model(os.path.join(path, 'D:/Project/masknew/face_mask.model'))
faceNet = cv2.dnn.readNet(os.path.join(path, 'D:/Project/masknew/face_mask.model', 'D:/project/masknew/face_mask.model/face_detect/deploy.prototxt.txt'),
                          os.path.join(path, 'D:/Project/masknew/face_mask.model', 'D:/Project/masknew/face_mask.model/face_detect/res10_300x300_ssd_iter_140000.caffemodel'))

pathnomask = r'D:/project/test1/Flaskmyweb/static/testpeople'

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(os.path.join(path, 'face_mask4.mp4'))
# def stream():
#     frame_width = int(cap.get(3))
#     frame_height = int(cap.get(4))
#     out = cv2.VideoWriter(os.path.join(path, 'test4.avi'),
#                         cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width, frame_height))
#     while True:
#         ret, frame = cap.read()
#         (h, w) = frame.shape[:2]
#         blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
#         faceNet.setInput(blob)
#         detections = faceNet.forward()

#         for i in range(0, detections.shape[2]):
#             confidence = detections[0, 0, i, 2]

#             if confidence < 0.5:
#                 continue

#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (startX, startY, endX, endY) = box.astype('int')
#             (startX, startY) = (max(0, startX), max(0, startY))
#             (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
#             face = frame[startY:endY, startX:endX]
#             face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
#             face = cv2.resize(face, (size, size))
#             face = np.reshape(face, (1, size, size, 3)) / 255.0
#             result = np.argmax(model.predict(face))

#             if result == 0:
#                 label = face_mask[result]
#                 color = (0, 255, 0)
#             else:
#                 label = face_mask[result]
#                 color = (0, 0, 255)

#             cv2.rectangle(frame, (startX, startY), (endX, endY), color, 3)
#             cv2.rectangle(frame, (startX, startY - 60), (endX, startY), color, -1)
#             cv2.putText(frame, label, (startX + 10, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
#             ###ถ่ายรูป
#             # cv2.imwrite(filename='saved_img.jpg', img=frame)
#             ###แปลงเพื่อไปแสดงหน้าเว็บ
#         # return cv2.imencode('.jpg', frame)[1].tobytes()
#         # if ret == True:
#         #     out.write(frame)
#         # else:
#         #     pass

#         # cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
#         # cv2.resizeWindow('Video', 800, 600)
        
#         # cv2.imshow('Video', frame)

#         # if cv2.waitKey(1) & 0xFF == ord('q'):
#         #     break

#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()
def stream():
    width = 800
    textIn = 0
    textOut = 0
    images = []
    countnomask = 0
    countimage = 0
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(os.path.join(path, 'test4.avi'),
                        cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width, frame_height))
    firstFrame = None

    while True:      
        ret, frame = cap.read()
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        faceNet.setInput(blob)
        detections = faceNet.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence < 0.5:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (size, size))
            face = np.reshape(face, (1, size, size, 3)) / 255.0
            result = np.argmax(model.predict(face))

            if result == 0:
                label = face_mask[result]
                color = (0, 255, 0)
            else:
                label = face_mask[result]
                color = (0, 0, 255)
                # countnomask += 1
                # print("countnomask {}".format(countnomask))
                
                # return countnomask
                

            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 3)
            cv2.rectangle(frame, (startX, startY - 60), (endX, startY), color, -1)
            cv2.putText(frame, label, (startX + 10, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            ###ถ่ายรูป
            # cv2.imwrite(filename='saved_img.jpg', img=frame)
            ###แปลงเพื่อไปแสดงหน้าเว็บ
        if ret == True:
            out.write(frame)
        else:
            pass
        cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Video', 800, 600)
        cv2.imshow('Video', frame)
        dim = (500,500)
        frame = cv2.resize(frame,dim,interpolation = cv2.INTER_AREA) 
            #converting images into grayscale       
        frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        images.append(frame_gray)
        # removing the images after every 50 image
        if len(images)==50:
                images.pop(0)

        image = np.array(images)
        # gettting the tracker value
        val = 50
        image = np.mean(image,axis=0)
        image = image.astype(np.uint8)
        # cv2.imshow('background',image)
        image = image.astype(np.uint8)
        # foreground will be background - curr frame
        foreground_image = cv2.absdiff(frame_gray,image)

        a = np.array([0],np.uint8)
        b = np.array([255],np.uint8)

        img = np.where(foreground_image>val,frame_gray,a)
        cv2.imshow('foreground',img)
        if (img[0:9]>0).any():
            countnomask += 1
            print(frame)
            print("countnomask {}".format(countnomask))
            # cv2.imwrite(filename='saved_img.jpg', img=frame)
        (grabbed, frame) = cap.read()
        text = "Unoccupied"
        # if the frame could not be grabbed, then we have reached the end
        # of the video
        if not grabbed:
            break
        # resize the frame, convert it to grayscale, and blur it
        frame = imutils.resize(frame, 800)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        # if the first frame is None, initialize it
        if firstFrame is None:
            firstFrame = gray
            continue
        # compute the absolute difference between the current frame and
        # first frame
        frameDelta = cv2.absdiff(firstFrame, gray)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        # loop over the contours
        for c in cnts:
            # print(c)
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < 12000:
                continue
            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)
            # น้ำเงินออก แดงเข้า
            # (x,y)
            # เส้นล่าง
            # cv2.line(frame, (0,550), (800, 550), (0, 0, 255), 1)  # red line
            # cv2.line(frame, (0,525), (800, 525), (250, 0, 1),1)  # blue line
            
            # เส้นกลาง
            cv2.line(frame, (390,0), (390, 800), (0, 0, 255), 1)  # red line
            cv2.line(frame, (395,0), (395, 800), (250, 0, 1),1)  # blue line
            rectagleCenterPont = ((x + x + w) // 2, (y + y + h) // 2)
            
            cv2.circle(frame, rectagleCenterPont, 1, (0, 0, 255), 5)

            if (testIntersectionOut((x + x + w) // 2, (y + y + h) // 2)):
                textOut += 1
                print("Out = {} ".format(textOut) , time.strftime("%d/%m/%y %H:%M:%S"))
                log.write("Out = {} ".format(textOut)+ time.strftime("%d/%m/%y %H:%M:%S") +'\n')
            if (testIntersectionIn((x + x + w) // 2, (y + y + h) // 2)):
                textIn += 1
                print("In = {} ".format(textIn) , time.strftime("%d/%m/%y, %H:%M:%S"))
                log.write("In = {} ".format(textOut)+ time.strftime("%d/%m/%y %H:%M:%S") +'\n')
            # draw the text and timestamp on the frame

            # show the frame and record if the user presses a key
            # cv2.imshow("Thresh", thresh)
            # cv2.imshow("Frame Delta", frameDelta)ๆ

        k = cv2.waitKey(1)
        # while(countimage == countnomask) :
        #     countimage += 1
        #     time.sleep(3) 
        #     img_name = "nomask{}.jpg".format(countimage)
        #     cv2.imwrite(os.path.join(pathnomask,img_name), frame)
        #     print("{} written!".format(img_name))


        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif(countimage < countnomask) :
            # SPACE pressed
            countimage += 1
            # time.sleep(3) 
            img_name = "nomask{}.jpg".format(countimage)
            cv2.imwrite(os.path.join(pathnomask,img_name), frame)
            print("{} written!".format(img_name))
           
            # return countimage
        cv2.putText(frame, "Out: {}".format(str(textOut)), (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, "In: {}".format(str(textIn)), (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.imshow("Counting People", frame)
        cv2.imencode('.jpg', frame)[1].tobytes()
    log.flush()
    log.close()
        # return cv2.imencode('.jpg', frame)[1].tobytes()
# cap.release()
# # out.release()
# cv2.destroyAllWindows()
