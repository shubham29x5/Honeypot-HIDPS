import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
#import tensorflow as tf
#import keras
#from sklearn.metrics import recall_score
#from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model, load_model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import keras.backend as K
import pickle
'''
def precision(y_true, y_pred):
    return K.mean(y_pred)
'''
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def generate_token_sent(s):
    # Convert to lowercases
    if type(s) == str:
    	s = s.lower()
    else:
        s = str(s)
        s = s.lower()
    
    # Replace all none alphanumeric characters with spaces
    s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
    
    
    # Break sentence in the token, remove empty tokens
    tokens = [token for token in s.split(" ") if token != ""]
    
    #construct new sentence with tokens
    sent = ""
    for i in range(len(tokens)):
        if i == 0:
            sent = sent + tokens[0]
        #elif i != (len(tokens)-1):
        #    sent = sent + " " + tokens[i]
        else:
            sent = sent + " " + tokens[i]#sent = sent + tokens[i]

    return sent
    

def generate_token_sent_col(arr):
    new_arr = []
    for i in range(arr.size):
        new_arr.append(generate_token_sent(arr[i]))
    new_arr = np.array(new_arr)
    return new_arr

#Training data
df = pd.read_csv('data.csv',delimiter=',',encoding='latin-1')

train_data = np.column_stack((df.payload,df.type))
#print(train_data[:5, :])

X = train_data[:,0]
X = generate_token_sent_col(X)
Y = train_data[:,1]

le = LabelEncoder()
Y = le.fit_transform(Y)
Y = Y.reshape(-1,1)

ol = train_data
print('ol',ol.shape)
nX, nXt, nY, nYt = train_test_split(ol[:,0], ol[:,1], test_size=0.33, random_state=42)

nX = generate_token_sent_col(nX)

le = LabelEncoder()
nY = le.fit_transform(nY)
nY = nY.reshape(-1,1)

nXt = generate_token_sent_col(nXt)

le = LabelEncoder()
nYt = le.fit_transform(nYt)
nYt = nYt.reshape(-1,1)


X_train,X_test,Y_train,Y_test = nX, nXt, nY, nYt
#print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)
#X_train,X_test,Y_train,Y_test = X, xstx, Y, xst


max_words = 10000
max_len = 1500
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)

# saving tokenizer
with open('lstm_tokenizer.pickle', 'wb') as f:
    pickle.dump(tok, f, protocol=pickle.HIGHEST_PROTOCOL)

# loading tokenizer
# with open('lstm_tokenizer.pickle', 'rb') as f:
#    tok = pickle.load(f)


sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)


def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model


model = RNN()
model.summary()
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy', precision])
#model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=[precision, recall])

model.fit(sequences_matrix,Y_train,batch_size=128,epochs=10, validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])

model.save('lstm.h5')

# code for loading model
# model = load_model('lstm.h5')

test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)


accr = model.evaluate(test_sequences_matrix,Y_test)


print('Test set\n  Loss: {:0.4f}\n  Accuracy: {:0.4f}\n precision: {:0.4f}'.format(accr[0],accr[1],accr[2]))
