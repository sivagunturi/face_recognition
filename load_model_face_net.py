import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
# example of loading the keras facenet model
import os
from keras.models import load_model
os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit'
# load the model
model = load_model('/home/schevala/dl/wip/keras/face_recognition/facenet_keras.h5')
# summarize input and output shape
print(model.inputs)
print(model.outputs)
