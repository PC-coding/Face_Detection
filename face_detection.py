import cv2
from random import randrange

# load pre-trained data on face frontals from opencv
trained_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# choosing image to detect faces in
img = cv2.imread('Cillian.png')


# capturing video from webcam
webcam = cv2.VideoCapture(0)

# iterate over frames continously
while True:
    # reading current frame
    successful_frame_read, frame = webcam.read()
   # boolean (always true)       #img 

    # conversion to grayscale
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
"""
# waits in the program until a key is pressed
# key = cv2.waitKey(1)


# detection algorithm
face_detect = trained_data.detectMultiScale(grayscale_img)
print(face_detect)

# setting coordinates to detect the first face from top left to bottom right
 # (x, y, w, h) = face_detect[0]

# rectangular boxes identifying faces 
# coordinates set to x,y,w,h and color set to red with a border thickness of 7 
 # cv2.rectangle(img, (150, 212), (920, 920+212), (0, 255, 0), 2)
for (x, y, w, h) in face_detect:
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 7)

# displaying images + pauses the python program with the image for viewing
# clicking any key will close the app
# message will appear
cv2.imshow('Face Detector', img)
cv2.waitKey()
"""

print('success')