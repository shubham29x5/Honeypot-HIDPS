import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
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
import urllib
import os

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def decode_back(s):
    try:
        s = urllib.parse.unquote(s)
        s = str.encode(s)
    except:
        return s
    return s.decode('utf8')

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

def generate_token_sent_col1(arr):
    new_arr = []
    for i in range(arr.size):
        new_arr.append(generate_token_sent(arr[i]))
    new_arr = np.array(new_arr)
    return new_arr

# saving tokenizer
#with open('lstm_tokenizer.pickle', 'wb') as f:
#    pickle.dump(tok, f, protocol=pickle.HIGHEST_PROTOCOL)

# loading tokenizer
with open('lstm_tokenizer.pickle', 'rb') as f:
    tok = pickle.load(f)

#model.save('lstm.h5')

# code for loading model
model = load_model('lstm.h5', custom_objects={'precision':precision})

#x="%3cscript%3ealert(%22hello%22)%3c/script%3e"

def predict(x):
    testpayload = np.array([decode_back(x)])
    testpayload_new = generate_token_sent_col1(testpayload)
    test_sequences = tok.texts_to_sequences(testpayload_new)
    test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=1500)
    print('Prediction for', st,' is ', model.predict(test_sequences_matrix)[0,0])
    return model.predict(test_sequences_matrix)[0,0]


def route(data):
    if isinstance(data, list):
        for each in data:
            try:
                s = each.lower()
            except Exception as ex:
                s = str(each).lower()
            if 'name' in s: 
                pload = s.split()[6][16:]
                ip=s.split()[0]
                score = predict(pload)
                if score >= 0.5:
                    os.system('iptables -t nat -A PREROUTING -p tcp --dport 80 -j DNAT --to-destination 192.168.1.114:80')
                    os.system('iptables -t nat -A POSTROUTING -p tcp -d 192.168.1.114 --dport 80 -j SNAT --to-source 192.168.1.113')
                    print(ip)
                else:
                    print('not XSS')

    else:
        print('Route Function: Data type error')

fname = '/var/log/apache2/access.log'
while(True):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    if len(content) > 0:
        os.system('cat /var/log/apache2/access_reset.log > /var/log/apache2/access.log')
        route(content)
