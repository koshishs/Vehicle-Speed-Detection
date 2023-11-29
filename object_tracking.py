import cv2
from speed_tracker import *
import numpy as np
end = 0

#Creater Tracker Object
tracker = ObjectTracker()

cap = cv2.VideoCapture("video/clip1.mp4")

fps = 60 #60 frames per second
wait_time = int(1000/(fps)) #difference between the frame rate and 1.


#Object Detection
object_detector = cv2.createBackgroundSubtractorMOG2(history=None,varThreshold=None)
#100,5

#KERNALS
kernalOp = np.ones((3,3),np.uint8)
kernalOp2 = np.ones((5,5),np.uint8)
kernalCl = np.ones((11,11),np.uint8)
fgbg=cv2.createBackgroundSubtractorMOG2(detectShadows=True)
kernal_e = np.ones((5,5),np.uint8)

while True:
    ret,frame = cap.read() # Read the next frame
    # Check if the frame was read successfully
    if not ret:
        break
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5) # Resize the frame
    height,width,_ = frame.shape

    #Extract ROI
    roi = frame[20:720, 0:1980]

    #MASKING
    fgmask = fgbg.apply(roi)
    ret, imBin = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    mask1 = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernalOp)
    mask2 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernalCl)
    e_img = cv2.erode(mask2, kernal_e)

    # Object Detection
    contours,_ = cv2.findContours(e_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)

        #THRESHOLD
        if area > 1000:
            x,y,w,h = cv2.boundingRect(cnt)
            # Draw bounding rectangle around the detected object
            cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),3)
            detections.append([x,y,w,h]) # Append the detection coordinates to the list

    #Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x,y,w,h,id = box_id

        # Check if the speed is below the limit
        if(tracker.getsp(id)<tracker.limit()):
            cv2.putText(roi,str(id)+" "+str(tracker.getsp(id)),(x,y-15), cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (255,0,0), 3)
        else:
            cv2.putText(roi,str(id)+ " "+str(tracker.getsp(id)),(x, y-15),cv2.FONT_HERSHEY_PLAIN, 1,(0,0,255),2)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0,0,255), 3)

        s = tracker.getsp(id)
        if (tracker.f[id] == 1 and s != 0):
            tracker.capture(roi, x, y, h, w, s, id)

    # DrawingLINES
    cv2.line(roi, (0, 430), (960, 430), (0, 0, 0), 1)
    cv2.line(roi, (0, 255), (960, 255), (0, 0, 0), 1)


    #DISPLAY
    cv2.imshow("Erode", e_img)
    cv2.imshow("ROI", roi)

    # Check for key press
    key = cv2.waitKey(wait_time-10)
    # If ESC key is pressed, end the tracker
    if key==27:
        tracker.end()
        end=1
        break

if(end!=1):
    tracker.end()

cap.release()
cv2.destroyAllWindows()
