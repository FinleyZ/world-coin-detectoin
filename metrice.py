import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import os, cv2, itertools
import numpy as np
import matplotlib.pylab as plt
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

model = load_model('model.h5')

valid_datagen = ImageDataGenerator(
    rescale=1 / 255.0,
)
val_generator = valid_datagen.flow_from_directory(
    'test',
    target_size=(224, 224),
    batch_size=64,
    class_mode='categorical',
    shuffle=False,
)

y_pred_score, y_test = model.predict(val_generator,verbose=1), val_generator.classes
y_test, y_pred = np.array(y_test), np.array(np.argmax(y_pred_score, axis=-1))
print(f'test set accuracy:{accuracy_score(y_test, y_pred):.3f} precision:{precision_score(y_test, y_pred, average="macro"):.3f} recall:{recall_score(y_test, y_pred, average="macro"):.3f} f1_score:{f1_score(y_test, y_pred, average="macro"):.3f}')