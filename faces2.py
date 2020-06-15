#import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier(r"C:\Users\abrar\OneDrive - Bangladesh University of Professionals\FR new\haarcascades\haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {}
with open("labels.pickel",'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

#cap = cv2.VideoCapture(0)
img = cv2.imread(r"C:\Users\abrar\OneDrive - Bangladesh University of Professionals\FR new\av.jpg")

while(True):
    #ret,frame = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    #print(faces)
    for x,y,w,h in faces:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = img[y:y+h,x:x+w]
        
        id_,conf = recognizer.predict(roi_gray)
        if conf>=45:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255,255,255)
            stroke = 2
            cv2.putText(img,name,(x,y),font,1,color,stroke,cv2.LINE_AA)
            
        
    img_item = "my-image.png" #
    cv2.imwrite(img_item,roi_gray)
    
    color = (255,0,0)
    stroke = 3
    cv2.rectangle(img,(x,y),(x+w,y+h),color,stroke) #
    
    cv2.imshow('frame',img)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
    
#cap.release()
cv2.destroyAllWindows()