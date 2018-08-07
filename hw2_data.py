import pandas as pd
import numpy as np
import scipy as sp
import math

class hw2_data:

	def __init__(self, data):
		
		self.raw_data = data	
		self.feature_template = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked', 'Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool', 'Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse', 'Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces', 'Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried', 'White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black', 'United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands', 'capital_gain', 'capital_loss', 'hours_per_week', 'education_num', 'age', 'fnlwgt', 'sex', '?_workclass', '?_native_country']
		self.feature_continous = ['capital_gain', 'capital_loss', 'hours_per_week', 'education_num', 'age', 'fnlwgt']
		self.train_y = self.get_result(self.raw_data) 
		self.feature_dict = self.make_dic(self.raw_data, self.feature_template)
		self.feature_dict = self.remove_none(self.feature_dict)
		self.wrong = self.get_wrong(self.feature_dict, self.feature_template)

	def get_wrong(self, dict_1, b):
		a = dict_1[0].keys()
		return list(set(a) - set(b))

#	def normalization(self, data):


	def remove_none(self, dict_list):
		output = []
#		print(len(dict_list[0]))
		for i in dict_list:
			output.append({ k:v if v is not None else float(0) for k, v in i.items()})
#		print ('---------')
#		print(len(output[0]))
#		print (output[0])
		return output

	
	def get_result(self, data):
		temp = data[-1][1:]
#		print (temp)
		result = [1 if '<=50K' in i else 0  for i in temp]

		return np.asarray(result).astype(float)
	
	def make_dic(self, data, f_name):
		
		output_dict_list = [dict.fromkeys(self.feature_template) for i in range(data.shape[1]-1)]
#		print(len(output_dict_list[0]))
		for i in data[:-1]:
#			print (i[0])
			if i[0] in self.feature_continous:
#				print(i[0])
				for index,feature in enumerate(i[1:]):
					output_dict_list[index][i[0]] = float(feature)
#				print(len(output_dict_list[0]))
			elif i[0] == 'sex':
				for index,feature in enumerate(i[1:]):
					if feature == 'Male':
#						print('isMale')
						output_dict_list[index][i[0]] = float(1)
					else:
						output_dict_list[index][i[0]] = float(0)
#				print(len(output_dict_list[0]))
			else:
				for index,feature in enumerate(i[1:]):
					if feature == '?':
						if i[0] == 'workclass':
							output_dict_list[index]['?_workclass'] = float(1)
						else:
							output_dict_list[index]['?_native_country'] = float(1)
					else:
						output_dict_list[index][feature] = float(1)
#				print(len(output_dict_list[0]))
		
		return output_dict_list
				
				
		

def main():

	data = pd.read_csv('train.csv', header = None)

	raw_data = data.values[:, :]

	new_data = []
	for row in raw_data:
		new_data.append([i.replace(' ','') for i in row])
			

#	raw_data_v = np.transpose(np.asarray([i.replace(' ','') for i in row for row in raw_data]))

	raw_data_v = np.transpose(np.asarray(new_data))

#	for i in raw_data_v:
#		print (i[0])

#	print(raw_data_v)

	used = hw2_data(raw_data_v)
	
	print(used.train_y)

if __name__ == "__main__":
	main()
