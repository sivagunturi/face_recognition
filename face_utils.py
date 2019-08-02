from cv2 import rectangle
from keras.models import load_model
from mtcnn.mtcnn import MTCNN
from numpy import asarray
from numpy import expand_dims
from numpy import load
from numpy import savez_compressed
from numpy import linalg
from PIL import Image, ImageTk
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import cv2
import numpy as np
from os import listdir

directory_path = 'faceset/'
file_types = ['npy', 'npz']

detector = MTCNN()
facenet_model = load_model('facenet_keras.h5')


# develop a classifier for the 5 Celebrity Faces Dataset
# load dataset
# data = load('face-embeddings.npz', allow_pickle=True)
# trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
# print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))
# # normalize input vectors
# in_encoder = Normalizer(norm='l2')
# #trainX = trainX.reshape(trainy.shape[0])
# trainX = in_encoder.transform(trainX)
# #testX = testX.reshape(testy.shape[0])
# testX = in_encoder.transform(testX)
# # label encode targets
# out_encoder = LabelEncoder()
# print("trainy = ", trainy)
# out_encoder.fit(trainy)
# trainy = out_encoder.transform(trainy)
# testy = out_encoder.transform(testy)
# # fit model
# model = SVC(kernel='sigmoid', probability=False)
# print("trainx shape", str(trainX.shape[0]),
#         "train y shape", str(trainy.shape[0]))
# model.fit(trainX, trainy)
# print("Fiting the model")




def createTrainSet(cap, labels, required_size=(160, 160)):
    X, Y = list(), list()
    for i in range(1):
        faces = list()
        #cap.set(cv2.CAP_PROP_CONVERT_RGB, 0);
        _, frame = cap.read()
        # print("name = ", labels)
        frame = cv2.flip(frame, 1)
        face = UseMTCNN(frame)
        # print("type of face = ", type(face))
        image = Image.fromarray(np.asarray(face))
        image = image.resize(required_size,  Image.NEAREST)
        face_array = np.asarray(image)
        # print("shape of face = ", face.shape)
        # faces.append()
        # print("run face", len(faces))
        X.append(face_array)
        Y.append(str(labels))
        
    print("Trainset ==> ", Y)
    return np.asarray(X), np.asarray(Y)


def createTestSet(cap, labels):
    return createTrainSet(cap, labels)


# get the face embedding for one face
def get_embedding(face_pixels, need_extra_dim=False):
    global facenet_model
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    #print("samples shape before", face_pixels.shape)
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    #print("samples shape after", samples.shape)
    yhat = facenet_model.predict(samples)
    #print("embedding shape", yhat[0])
    return yhat[0]


def create_faceEmbeddings():
    print("creating face embeddings")
    global facenet_model
    # load the face dataset
    #data = np.load('data_set.npz', allow_pickle = True)
    print ("loading faceset")
    data = {dir_content: np.load("faceset/"+dir_content)
            for dir_content in listdir(directory_path)
            if dir_content.split('.')[-1] in file_types}

    # sample = [item[0] for item in data]
    # print(sample)
    # trainX, trainy, testX, testy = np.empty_like(sample['arr_0']),  np.empty_like(
    #     sample['arr_1']),  np.empty_like(sample['arr_2']),  np.empty_like(sample['arr_3'])
    newTrainX = list()
    newTestX = list()

    #final_data = np.emty_like(data)
    trainy, testy = list(), list()

    for i in data:
        trainX, tay, testX, tey = data[i]['arr_0'], data[i]['arr_1'], data[i]['arr_2'], data[i]['arr_3']
        #print('Loaded: ', trainX.shape, tay.shape, testX.shape, tey.shape)
        #print('Loaded: ', tax.shape, tay.shape, tex.shape, tey.shape  )
        #trainX.append(tax)
        trainy.extend(tay)
        #testX.append(tex)
        testy.extend(tey)
        #print("trainy = ", trainy, " testy = ", testy)
        # trainX = np.concatenate((trainX, tax), axis=0)
        # trainy = np.concatenate((trainy, tay), axis=0)
        # testX = np.concatenate((testX, tex), axis=0)
        # testy = np.concatenate((testy, tey), axis=0)
        #print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape  )
        # convert each face in the train set to an embedding
        for face_pixels in trainX:
            embedding = get_embedding(face_pixels)
            #newTrainX = np.concatenate((newTrainX, embedding),axis=0)
            #print("train x type = ",type(newTrainX), "embedding type =", type(embedding))
            newTrainX.append(embedding)
        # convert each face in the test set to an embedding
        for face_pixels in testX:
            embedding = get_embedding(face_pixels)
            #newTestX = np.concatenate((newTestX, embedding), axis=0)
            newTestX.append(embedding)
        # save arrays to one file in compressed format
    newTrainX = asarray(newTrainX)
    newTestX = asarray(newTestX)
    #print('Dataset: train=%d, test=%d' % (newTrainX.shape[0], newTestX.shape[0]))
    # print("trainx shape", str(trainX.shape[0]),
    #       "train y shape", str(trainy.shape[0]))
    print ("Recreating face-embeddings.npz")
    savez_compressed('face-embeddings.npz', newTrainX, trainy, newTestX, testy)


def train_face():
    # develop a classifier for the 5 Celebrity Faces Dataset
    # load dataset
    data = load('face-embeddings.npz', allow_pickle=True)
    trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    #print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))
    # normalize input vectors
    in_encoder = Normalizer(norm='l2')
    #trainX = trainX.reshape(trainy.shape[0])
    trainX = in_encoder.transform(trainX)
    #testX = testX.reshape(testy.shape[0])
    testX = in_encoder.transform(testX)
    # label encode targets
    out_encoder = LabelEncoder()
    #print("trainy = ", trainy)
    out_encoder.fit(trainy)
    trainy = out_encoder.transform(trainy)
    testy = out_encoder.transform(testy)
    # fit model
    model = SVC(kernel='linear', probability=True)
    #print("trainx shape", str(trainX.shape[0]),
    #     "train y shape", str(trainy.shape[0]))
    model.fit(trainX, trainy)
    # predict
    yhat_train = model.predict(trainX)
    yhat_test = model.predict(testX)
    score_train = accuracy_score(trainy, yhat_train)
    score_test = accuracy_score(testy, yhat_test)
    print('Accuracy: train=%.3f, test=%.3f' %
          (score_train*100, score_test*100))


def UseMTCNN(frame):
    global detector
    faces = detector.detect_faces(frame)
    for result in faces:
        # get coordinates
        x, y, width, height = result['box']
        x2, y2 = x + width, y + height
        # create the shape
        rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 1)
        return frame[x:x2, y:y2]

def predict_embedding(target_x):
    data = load('face-embeddings.npz', allow_pickle=True)
    #print ("data shape=", data['arr_0'])
    min_distance = 100
    predict = "unkown";
    count = 0
    index = 0
    for i, j in zip(data["arr_0"], data["arr_1"]):
        dist = linalg.norm(i - target_x)
        print("dist = " , dist, "label = ", j)
        count = count + 1
        if(dist < float(min_distance)):
            predict = j;
            min_distance = dist
            print("min_distance", min_distance)
            if(count >= len(data["arr_0"])):
                print("label = ", predict)
    return predict
    # temp = 0
    # for j in data["arr_1"]:
    #     temp = temp + 1
    #     if(temp == index):
    #         print("label = ", j)
    # for i in data:
    #     sourcex, sourcey= data[i]['arr_0'], data[i]['arr_1']
    #     #print ("sourcex =", sourcex[0], "sourcey = ", sourcey[0]);
    #     dist = numpy.linalg.norm(sourcex-target_x)
    #     print("distance = ", dist)

def predict(frame):
    # print("type of face = ", type(face))
    X = list()
    global detector
    faces = detector.detect_faces(frame)
    print("Predicting face")
    for result in faces:
        # get coordinates
        x, y, width, height = result['box']
        x2, y2 = x + width, y + height
        # create the shape
        if (y2 - y < 120 and x2-x < 120):
            continue
        rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 1)         
        image = Image.fromarray(np.asarray(frame[x:x2, y:y2]))
        image = image.resize((160, 160),  Image.NEAREST)
        face_array = np.asarray(image)
        trainX = get_embedding(np.asarray(face_array))
        X.append(trainX)
        # predict
        #yhat_class = model.predict(X)
        # global out_encoder
        # predict_names = out_encoder.inverse_transform(yhat_class)
        #predict_embedding(trainX);
        cv2.putText(frame, str(predict_embedding(trainX)), (y + 30, x2+30   ), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1,
                      color=(0, 255, 0))
        #print (predict_names)

