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
with open('normalTrafficTraining.txt') as f:
    ctnt = f.readlines()
    
ctnt = [x.strip() for x in ctnt]
ctnt = np.array(ctnt)

payload = []
for each in ctnt:
    try:
        if ord(each[0][0])>=99 and ord(each[0][0])<=124 :
            payload.append(each)
    except:
        pass

pyld1 = np.array(payload)

with open('normalTrafficTest.txt') as f:
    ctnt = f.readlines()
    
ctnt = [x.strip() for x in ctnt]
ctnt = np.array(ctnt)

payload = []
for each in ctnt:
    try:
        if ord(each[0][0])>=99 and ord(each[0][0])<=124 :
            payload.append(each)
    except:
        pass

pyld2 = np.array(payload)

payload = np.concatenate((pyld1,pyld2))

df = pd.read_csv('payload_train.csv',delimiter=',',encoding='latin-1')
pyld = df.payload
atype = df.attack_type
data = np.column_stack((pyld,atype))
normx = data[data[:,1]=='norm']
normpyld = normx[:,0]

payload = np.concatenate((payload,normpyld))

norm = np.full(payload.size, 'norm')
norm = np.column_stack((payload,norm))


df = pd.read_csv('xssed_payload.csv',delimiter=',',encoding='latin-1')
xss = np.full(df.values.size, 'xss')
xss_a = np.reshape(df.values, (df.values.size,))

xss_attck = np.column_stack((xss_a,xss))

train_data = np.concatenate((xss_attck,norm))

X = train_data[:,0]
X = generate_token_sent_col(X)
Y = train_data[:,1]

le = LabelEncoder()
Y = le.fit_transform(Y)
Y = Y.reshape(-1,1)


#Test data
with open('xss_vectors.txt') as f:
    content = f.readlines()
    
content = [x.strip() for x in content]
content = np.array(content)

xss = np.full(content.shape, 'xss')

xs = np.column_stack((content,xss))

dft = pd.read_csv('payload_test.csv',delimiter=',',encoding='latin-1')
payload_t = dft.payload
atype_t = dft.attack_type
data_t = np.column_stack((payload_t,atype_t))
normx_t = data_t[data_t[:,1]=='norm']
xssx_t = data_t[data_t[:,1]=='xss']
x_t = np.concatenate((normx_t, xssx_t), axis=0)
test_data = np.concatenate((x_t, xs), axis=0)

X_t = test_data[:,0]
X_t = generate_token_sent_col(X_t)
Y_t = test_data[:,1]

le = LabelEncoder()
Y_t = le.fit_transform(Y_t)
Y_t = Y_t.reshape(-1,1)

ol = np.concatenate((train_data,test_data))
#print('ol',ol.shape)
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
