import numpy as np
import scipy as sp 
import math
import pandas as pd
from sklearn import preprocessing

from hw2_data import hw2_data

def normalization(train, test):
	train_feature = []
	test_feature = []
#	train_feature = map(lambda x: train_feature.append(x.values),train)
#	test_feature = map(lambda x: test_feature.append(x.values),test)
	for i in train:
		train_feature.append(list(i.values()))
	for i in test:
		test_feature.append(list(i.values()))
	length = len(train_feature)
#	print (train_feature)
#	print (list(train.feature_dict.values())
	
	temp = np.asarray(train_feature + test_feature).astype(float)

	scaler = preprocessing.StandardScaler().fit(temp)

	out = scaler.transform(temp)

#	print(out.shape)

	return(out[:length], out[length:])

def handmade_normalization(train, test):
	train_feature = []
	test_feature = []
	for i in train:
		train_feature.append(list(i.values()))
	for i in test:
		test_feature.append(list(i.values()))
	length = len(train_feature)

	temp = np.asarray(train_feature + test_feature).astype(float)
	mu = (sum(temp)/ temp.shape[0])
	sigma = np.std(temp, axis = 0)
	mu = np.tile(mu, (temp.shape[0],1))
	sigma = np.tile(sigma, (temp.shape[0],1))
	out = (temp - mu)/ sigma
	
	return (out[:length], out[length:]) 


def change_dict(my_dict, numbers):
	count = 0
	for i in my_dict:
		for j in i:
			j = numbers[count]
			count += 1


def main():
	train_data = pd.read_csv('train.csv', header = None)
	raw_train_data = train_data.values[:, :]
	test_data = pd.read_csv('test.csv', header = None)
	raw_test_data = test_data.values[:, :]

#	print(raw_test_data)

	new_train_data = []
	new_test_data = []

	for row in raw_train_data:
		new_train_data.append([i.replace(' ', '') for i in row])
	
	for row in raw_test_data:
		new_test_data.append([i.replace(' ', '') for i in row])

#	print(new_train_data)

	train = hw2_data(np.transpose(np.asarray(new_train_data))) 
	
	
	

	test = hw2_data(np.transpose(np.asarray(new_test_data)))

	temp = normalization(train.feature_dict, test.feature_dict)

	temp2 = handmade_normalization(train.feature_dict, test.feature_dict)
	
	new_train = temp[0]

	new_test = temp[1]
	print (new_train.shape)
	print (new_test.shape)

#	print('-----------------')

	print (temp[0])

if __name__ == "__main__":
	main()
