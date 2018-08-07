import numpy as np
import pandas as pd
import math
import scipy as sp
import sklearn


from hw2_normalize import normalization
from hw2_data import hw2_data
from hw2_Logistic import logistic_model


def sigmoid(input_):
	output = 1/(1.0+np.exp(input_))
	return np.clip(output, 1e-8, 1-(1e-8))

def main():
#	data = pd.read_csv('test.csv', header = None)
#	raw_data = data.values[:, :]

#	new_data = []

#	for row in raw_data:
#		new_data.append([i.replace(' ', '') for i in row])
	
#	raw_data_v = np.transpose(np.asarray(new_data))

#	test_data = hw2_data(raw_data_v)

#	test_model = logistic_model(test_data)
	
#	print (test_model.x_train)
	
#	print (test_model.x_train[0])

	weight = np.loadtxt('weight')
	bias = np.loadtxt('bias')

	answer = pd.read_csv('correct_answer.csv', header = None)
	raw_answer = answer.values[:, :]
	
	real_answer = np.transpose(raw_answer)[1][1:]
	real_answer = real_answer.astype(float)

	normalized_test = np.loadtxt('normalized_test')

	print(normalized_test.shape)

	z = np.dot(normalized_test, np.transpose(weight)) + bias
	y = sigmoid(z)

	
	y_ = np.around(y)
#	y_ = np.squeeze(y)
	print(y_)
	print(y_.shape)
	count = 0
	for index, i in enumerate(y_):
		if i == real_answer[index]:
			count += 1

	acc = count/float(y_.shape[0])
	print(acc)

	#make answer_csv
	ans_mat = []

	for index, i in enumerate(y_):
		ans_mat.append([index+1, i])

	columns = ['id', 'label']
	outputDf = pd.DataFrame(np.asarray(ans_mat).astype(int), columns = columns)
	outputDf.to_csv('output.csv', index = False)

if __name__ == "__main__":
	main()
