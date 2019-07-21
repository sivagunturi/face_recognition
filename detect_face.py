import numpy as np
import cv2
from cv2 import CascadeClassifier
from cv2 import rectangle
from cv2 import circle
from mtcnn.mtcnn import MTCNN
import sys

classifier = None
detector = None

def main():
    global classifier, detector
    if(len(sys.argv) > 1):
        if(sys.argv[1] == 'cc'):
            classifier = CascadeClassifier('haarcascade_frontalface_default.xml')
        elif(sys.argv[1] == 'mtcnn'):
            detector = MTCNN()
        else:
            print ('Provide proper arguments to proceed ...')
            return
    else:
       print ('Arguments needed to proceed ...')
       return

    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Our operations on the frame come here
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if(sys.argv[1] == 'cc'):
            UseCascadeClassifier(frame)
        elif(sys.argv[1] == 'mtcnn'):
            face = UseMTCNN(frame)
            predictFace(face)
        else:
            break;
        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def UseCascadeClassifier(frame):
    # load the pre-trained model
    # perform face detection
    global classifier
    bboxes = classifier.detectMultiScale(frame)
    # print bounding box for each detected face
    for box in bboxes:
        x, y, width, height = box
        x2, y2 = x + width, y + height
        # draw a rectangle over the pixels
        rectangle(frame, (x, y), (x2, y2), (0,0,255), 1)

def UseMTCNN(frame):
    global detector
    faces = detector.detect_faces(frame)
    for result in faces:
        # get coordinates
        x, y, width, height = result['box']
        x2, y2 = x + width, y + height
        # create the shape
        rectangle(frame, (x, y), (x2, y2), (0,0,255), 1)
        return frame[x:x2, y:y2]

def predictFace(face):



if __name__ == '__main__':
    main()
