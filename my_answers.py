import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras
import re #need for regex. 


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    for i in range(0, (len(series)-window_size)):
        end_window = i + window_size
        X.append(series[i:end_window])
        #straightforward for output labels.
        y = series[window_size:] 
    # reshape each
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape= (window_size, 1)))
    model.add(Dense(1)) # dense output layer with size 1 to match size of input. 
       
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    # Used a regex courtesy of this resource: https://www.reddit.com/r/learnprogramming/comments/2ypirj/how_can_i_remove_all_non_alphabetic_characters/
    #Modified below to also include the punctuation. Eliminate .split() bc output error. 
    text = re.sub(r'[^a-zA-Z!,.:;?]', ' ', text)
    
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    for j in range(0, (len(text)-window_size), step_size):
        end_window = j + window_size
        inputs.append(text[j:end_window])
        outputs.append(text[end_window])

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars, activation = 'softmax'))
    return model 
