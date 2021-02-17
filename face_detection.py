import cv2

# load pre-trained data on face frontals from opencv
trained_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# choosing image to detect faces in
img = cv2.imread('Cillian.png')

# conversion to grayscale
grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detection algorithm
face_detect = trained_data.detectMultiScale(grayscale_img)
print(face_detect)

# rectangular boxes identifying faces 
# setting coordinates
(x, y, w, h) = face_detect[0]

# cv2.rectangle(img, (150, 212), (920, 920+212), (0, 255, 0), 2)
cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 225), 2)

# displaying images + pauses the python program with the image for viewing
# clicking any key will close the app
# message will appear
cv2.imshow('Face Detector', img)
cv2.waitKey()

print('success')