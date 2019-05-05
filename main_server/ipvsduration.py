import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('final.txt')
df = pd.DataFrame(df)
full_data = df.values.tolist()
k=[]
for i in full_data:
	k.append(i[0])
p=[]
for i in full_data:
	p.append(i[-1])
plt.plot(k,p)
plt.savefig('./ipvsdur.png')
