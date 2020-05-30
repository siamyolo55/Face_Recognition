import cv2

face_cascade = cv2.CascadeClassifier(r"C:\Users\abrar\Downloads\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml")

img = cv2.imread("E:\mem cpy\DCIM\Facebook\FB_IMG_1556697602134.jpg")

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray,scaleFactor = 1.05,minNeighbors = 5)

print(type(faces))
print(faces)

for x,y,w,h in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
    

cv2.imshow("lol",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
