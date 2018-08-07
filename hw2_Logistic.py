import pandas as pd
import numpy as np
import math
import scipy as sp
from sklearn import preprocessing


from hw2_normalize import normalization
from hw2_data import hw2_data


class logistic_model:
	
	def __init__(self, data):
		self.data = data
		self.x_train = np.asarray([np.asarray(list(i.values())) for i in self.data.feature_dict])
		print(type(self.x_train[0]))
#		print(np.asarray(list(x_train.values())))
		self.normlalized_x_train = None
		self.train_y = self.data.train_y
		self.weight = np.zeros(len(self.data.feature_dict[0]))

	def get_normalized_train(self, input_):
		self.normalized_x_train = input_
	
	def _shuffle(self, x_train, y_train):
		temp = np.arange(len(x_train))
		np.random.shuffle(temp)
		return (x_train[temp], y_train[temp])

	def split_valid(self, x_train_data, y_train_data, percentage):
		data_size = len(x_train_data)
		valid_data_size = int(math.floor(data_size*percentage))

		train_data, test_data = self._shuffle(x_train_data, y_train_data)

		train, test = train_data[:valid_data_size], test_data[:valid_data_size]
		train_valid , test_valid = train_data[valid_data_size:], test_data[valid_data_size:]
		return train, test, train_valid, test_valid

	def sigmoid(self, z):
		output = 1/(1.0+np.exp(-z))
		return np.clip(output, 1e-8, 1-(1e-8))

	def valid(self, w, b , x_valid, y_valid):
		valid_data_size = len(x_valid)

		z = (np.dot(x_valid, np.transpose(w))+b)
		y = self.sigmoid(z)
		y_ = np.around(y)
		result = np.asarray([1 if i == j else 0 for i, j in zip(np.squeeze(y_valid), y_)])
		print(result)
		print('Validation acc = %f' %(float(result.sum())/ valid_data_size) )


	def logistic_regression(self, X_train, Y_train):
		valid_pa = 0.1
		l_rate = 0.12
		batch_size = 64
		epoch_num = 2400
		save_parameter = 50

#		print(x_train[0])
		x_train, y_train, x_valid, y_valid = self.split_valid(X_train, Y_train, valid_pa)

		print(y_valid)

		train_size = len(x_train)
		weight = np.zeros((106,))
#		print(weight)
		s_gra = np.zeros((106,))		
		bias = np.zeros(1,)
		s_gra_b = np.zeros(1,)

		iter_num = int(math.floor(train_size/batch_size))

		total_loss = 0.0

		for epoch in range(1, epoch_num):

			print('--------' + str(epoch) + '-----------')

			if epoch % save_parameter == 0:
				print('=====Saving Param at epoch %d ====' % epoch)
				np.savetxt('weight', weight)
				np.savetxt('bias', bias)
				print('epoch avg loss = %f' %(total_loss/ (float(save_parameter)*train_size)))
				total_loss = 0.0
				self.valid(weight,bias, x_valid, y_valid)

			x_train, y_train = self._shuffle(x_train, y_train)

			for batch in range(iter_num):
#				print('----------' + str(batch) + '---------')
				
				X = x_train[batch*batch_size:(batch+1)*batch_size]
				Y = y_train[batch*batch_size:(batch+1)*batch_size]

#				print(X[0].shape)
#				print(weight.shape)
				z = np.dot(X, np.transpose(weight)) + bias
				y = self.sigmoid(z)

				cross_entropy = -(np.dot(Y, np.log(y))) + np.dot((1 - Y), np.log(1-y))
				total_loss += cross_entropy

				w_grad = np.mean(-1 * X * (Y- y).reshape((batch_size, 1)), axis = 0)
#				print("w_grad =" , w_grad)
				b_grad = np.mean(-1 * (np.squeeze(Y) - y))
#				print("b_grad = ", b_grad)
				s_gra += w_grad**2
				ada = np.sqrt(s_gra)
				s_gra_b += b_grad**2
				ada_b = np.sqrt(s_gra_b)

				weight = weight - l_rate * w_grad/ada
				bias = bias - l_rate * b_grad/ada_b

		return (weight, bias)

def main():
	
	data_train = pd.read_csv('train.csv', header = None)
	raw_train_data = data_train.values[:, :]
	data_test = pd.read_csv('test.csv', header = None)
	raw_test_data = data_test.values[:, :]

	print(raw_test_data.shape)

	new_train_data = []
	new_test_data = []

	for row in raw_train_data:
		new_train_data.append([i.replace(' ','') for i in row])
	for row in raw_test_data:
		new_test_data.append([i.replace(' ','') for i in row])

	raw_data_v = np.transpose(np.asarray(new_train_data))

	train_data = hw2_data(raw_data_v)
	test_data = hw2_data(np.transpose(np.asarray(new_test_data)))

	normalized_data = normalization(train_data.feature_dict, test_data.feature_dict)
#	print(len(train_data.feature_dict))

	model = logistic_model(train_data)

	model.get_normalized_train(normalized_data[0])

	model_pair = model.logistic_regression(model.normalized_x_train, model.train_y)

	print(normalized_data[1].shape)

	np.savetxt('normalized_test', normalized_data[1])
#	print(model_pair[0])

if __name__ == "__main__":
	main()
