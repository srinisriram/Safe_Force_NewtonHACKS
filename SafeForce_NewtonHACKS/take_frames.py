import cv2
import time


input = input("What facial area are you going to be touching?")
#person = int(input("What person is doing this?"))
person = 2

vs = cv2.VideoCapture(0)

i=0

time.sleep(5)
while i < 300:
	ret, frame = vs.read()

	img = frame
	cv2.imshow("img",img)
	cv2.waitKey(1)

	cv2.imwrite(input + "Person"+ str(person) + str(i)+'.jpg',img)
	i+=1
