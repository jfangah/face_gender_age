import numpy as np
import pandas as pd

temp = []
with open('train.txt') as train:
	for i in train.readlines():
		temp.append(i.split())
with open('val.txt') as val:
	for i in val.readlines():
		temp.append(i.split())
df = pd.DataFrame(np.array(temp),columns=['id', 'gender'])
df.index = df['id']
df = df.drop(columns=['id'], axis=1)
df.to_csv('gender.csv')