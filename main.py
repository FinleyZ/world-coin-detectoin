import math, os
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
    
from tensorflow import keras
from keras import backend as K
from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from keras.losses import CategoricalCrossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2

if __name__ == '__main__':
    train_datagen = ImageDataGenerator(
        rotation_range=30,
        height_shift_range=0.1,
        width_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        rescale=1 / 255.0, 
    )

    test_datagen = ImageDataGenerator(
        rescale=1 / 255.0,
    )

    train_generator = train_datagen.flow_from_directory(
        'train', 
        target_size=(224, 224), 
        batch_size=32,
        class_mode='categorical',
        shuffle=True,
    )

    val_generator = test_datagen.flow_from_directory(
        'val',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False,
    )

    inputs = Input(shape=(224, 224, 3))
    mobilenetv2 = MobileNetV2(include_top=False, input_tensor=inputs)
    cnn = GlobalAveragePooling2D()(mobilenetv2.layers[-1].output)
    cnn = Dropout(0.3)(cnn)
    cnn = Dense(512, activation='relu')(cnn)
    cnn = Dense(211, activation='softmax')(cnn)
    model = Model(inputs=inputs, outputs=cnn)
    model.summary()
    model.compile(optimizer=Adam(1e-4), loss=CategoricalCrossentropy(label_smoothing=0.1), metrics=['accuracy'])
    model.fit_generator(train_generator, verbose=1, epochs=100, validation_data=val_generator, callbacks=[
        CSVLogger('train.log'),
        ModelCheckpoint('model.h5', save_best_only=True, verbose=2, monitor='val_accuracy'),
        ReduceLROnPlateau(factor=0.5, patience=10, verbose=2)
    ])