

import cv2
#Reading Cascade Files
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
glass_casscade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye_tree_eyeglasses.xml')
smile_casscade = cv2.CascadeClassifier('haarcascade_smile.xml')

#Using webcam
cap = cv2.VideoCapture(0)

while True:
    _,frame = cap.read()   # _ is the boolean value

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  #converting frame to gray 
    faces = face_cascade.detectMultiScale(gray,1.3,5)   # detecting the faces from the gray frame 
    for (x,y,w,h) in faces:

        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)  # drawing the rectangle around the face

        roi_gray = gray[y:y+h,x:x+w]   #roi is Region of interest
        roi_color = frame[y:y+h,x:x+w]

        eye = glass_casscade.detectMultiScale(roi_gray)  # detecting the eyes from the roi
        smile = smile_casscade.detectMultiScale(roi_gray) # detecting the eyes from the roi
        
        for (a,b,c,d) in eye:
            cv2.rectangle(roi_color,(a,b),(a+c,b+d),(0,255,0),2) # drawing the rectangle around the eye

        for (k,l,m,n) in smile:
            cv2.rectangle(roi_color,(k,l),(k+m,l+n),(0,0,255),2) # drawing the rectangle around the smile
            

    cv2.imshow("Video",frame)  # showing the video frame which is named Video
    k= cv2.waitKey(30) & 0xff   #press escape to exit
    if k==27:
        break

cap.release()
cv2.destroyAllWindows()