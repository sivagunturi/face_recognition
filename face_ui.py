from cv2 import circle
from cv2 import rectangle
from numpy import savez_compressed
from PIL import Image, ImageTk
from tkinter import Tk, Button, Label, Entry
import cv2
import face_utils
import numpy as np
import PIL
import threading
import load_dummy_data


def CreateDummyDataSet():
    trainX, trainy = load_dummy_data.load_dataset(
        '5-celebrity-faces-dataset/train/')
    # load test dataset
    testX, testy = load_dummy_data.load_dataset(
        '5-celebrity-faces-dataset/val/')
    savez_compressed('faceset/dummy.npz', trainX, trainy, testX, testy)


def runFaceThread(required_size=(160, 160)):
    if(e1.get() == ""):
        print("Enter the label to register face")
        return
    trainX, trainy = face_utils.createTrainSet(cap, e1.get())
    # print(" 2 -- > trainx shape",
    #       str(trainX.shape[0]), "train y shape", str(trainy.shape[0]))
    testX, testy = face_utils.createTestSet(cap, e1.get())
    #CreateDummyDataSet()
    testX = np.concatenate((testX, dummytestX), axis=0)
    testy = np.concatenate((testy, dummytesty), axis=0)
    savez_compressed('faceset/' + e1.get() + '.npz',
                     trainX, trainy, testX, testy)
    face_utils.create_faceEmbeddings()
    #face_utils.train_face()


def registerFace():
    #threading.Thread(target=runFaceThread).start()
    runFaceThread()

def predictFace():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    face_utils.UseMTCNN(frame)
    face_utils.predict(frame)


root = Tk()

width, height = 800, 600
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

btn = Button(root, text="Register face!", command=registerFace)
#btn.grid(row=1, column=0, sticky=W, pady=4)
btn.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)

name = Label(root,
             text="Enter Name")
name.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)

e1 = Entry(root)
#1.grid(row=0, column=1)
e1.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)

predict_btn = Button(root, text="Predict face!", command=predictFace)
#btn.grid(row=1, column=0, sticky=W, pady=4)
predict_btn.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)

root.bind('<Escape>', lambda e: root.quit())
lmain = Label(root)




def show_frame():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    face_utils.UseMTCNN(frame)
    #face_utils.predict(frame)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = PIL.Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)
    
show_frame()
lmain.pack()
root.mainloop()
cap.release()
cv2.destroyAllWindows()
