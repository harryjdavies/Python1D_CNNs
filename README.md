# Python1D_CNNs
These examples use human activity accelerometer data available at https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
The aim is to classify the data segments into one of six catagories: Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying
The data has 9 channels of accelerometer and gyroscope data:  total acceleration (x,y,z), body acceleration (x,y,z), and body gyroscope (x,y,z).

KERAS EXAMPLE: adapted from https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/
The default model involves two 1D convolutional layers, a maxpool layer of size 2, a flattening layer, a dense layer to compress to 100 hidden features, and a final dense layer to compress into the 6 outputs.
4 varients of this model are created, the final having a single 1D convolution later and a maxpool of size 10. In an example such as this, there is a lot of redundant data. Thus, narrowing down the features into a few important features (maxpool 10) increases the focus on the important features, and discards redundant features that could be a source of error.

PYTORCH EXAMPLE: the data extraction is the same as in the keras example. The 1D convolutional neural network is built with Pytorch, and based on the 5th varient from the keras example - a single 1D convolutional layer, a maxpool layer of size 10, a flattening layer, a dense/linear layer to compress to 100 hidden features and a final linear layer to compress to the 6 outputs. 
