import cv2

# load pre-trained data on face frontals from opencv
trained_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# choosing image to detect faces in
img = cv2.imread('Cillian.png')

# displaying images + pauses the python program with the image for viewing
# clicking any key will close the app
# message will appear
cv2.imshow('Face Detector', img)
cv2.waitKey()

print('success')