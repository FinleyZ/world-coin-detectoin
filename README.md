# world-coin-detectoin
# Introduction
This project uses the World Coins image dataset available on Kaggle (https://www.kaggle.com/datasets/wanderdust/coin-images). 

The model is already trained, and all the relevant data used to evaluate and plot information about the model are stored in `model.h5`, `train.log`, and `curve.png`. 

### Windows system
If using Windows, make the following changes to `main.py`:
1. on line 14
    > `from import keras.optimizers import adam_v2`
2. on line 18
   >  `from keras.applications import mobilenet_v2`
3. on line 51
    > `mobilenetv2 = mobilenet_v2.MobileNetV2(include_top=False, input_tensor=inputs)`
4. on line 58
   > `model.compile(optimizer=adam_v2.Adam(learning_rate=1e-4), loss=CategoricalCrossentropy(label_smoothing=0.1), metrics=['accuracy'])`

# Model
Run the following scripts to

## Train the model
The `main.py` script is used to train the model. Stores the model in `model.h5`, and the training and validation to metrics are stored in `train.log`
Note: This may take close to 16 hours to train.

## Evaluate the model: 
The `metrice.py` script is used to display the metrics of the trained model. It uses information stored in the `model.h5` to calculate the metrics of the trained model. 

## Plot graphs: 
The `plot_curve.py` script is used to plot the accuracy and loss graph of the trained model. It uses information stored in the `train.log` to plot the graphs. These graphs are saved as `curve.png`.

## Predict using the trained model: 
The `predict.py` script is used to predict the class of individual images. 
On running the script, it asks for an input image on the terminal. 
