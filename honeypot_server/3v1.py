import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.patches as mpatches
import warnings
from math import sqrt
from collections import Counter
style.use('fivethirtyeight')
import pandas as pd 
import random

df = pd.read_csv('x.txt')
df = pd.DataFrame(df)
full_data = df.values.tolist()
plt.xlabel('duration(seconds)')
plt.ylabel('data(bytes)')
in_data=[]
for i in full_data:
    in_data.append(i[3])
out_data=[]
for i in full_data:
    out_data.append(i[5])
total_bytes=[]
for i in full_data:
    total_bytes.append(i[7])
duration=[]
for i in full_data:
    duration.append(i[-1])
plt.plot(duration, in_data, '-ro', linewidth=2, label="in")
plt.plot(duration, out_data, '-go', linewidth=2, label="out")
plt.plot(duration, total_bytes, '-bo', linewidth=2, label="total")

#plt.show()
plt.savefig('./band.png')

