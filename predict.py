import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import os, cv2, json
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

with open('cat_to_name.json', 'rb') as f:
    label = json.load(f)

model = load_model('model.h5')

while True:
    img_path = input('Input Image Name:')
    try:
        img = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = np.expand_dims(img, axis=0) / 255.0
        pred = np.argmax(model.predict(img)[0])

        plt.figure(figsize=(6, 6))
        plt.imshow(plt.imread(img_path))
        plt.axis('off')
        plt.title('pred label:{}'.format(label[str(pred + 1)]))
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print('Error, Try Again! {}'.format(e))