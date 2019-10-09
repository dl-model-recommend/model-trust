'''
Calculate the model trust score given ideal trust matrix and predicted trust matrix
'''

import pandas as pd
import trust_params

def calculate():
	"""Method to calculate the trust score given the ideal trust matrix and the generated trust matrix for the DL model"""
	df_trust = pd.read_csv(trust_params.OUTPUT_MATRIX_FILE_NAME, index_col=0)
	df_ideal = pd.read_csv(trust_params.IDEAL_MATRIX_FILE_NAME, index_col=0)

	total_count = 0
	total_diff = 0
	for index, row in df_trust.iterrows():
		print("Index row: ", index)
		for col, value in row.iteritems():
			diff_value = round(abs(value - df_ideal.loc[index,col]),4)
			if(index == col):
				total_count += 4
				total_diff += 4*diff_value
			else:
				total_count += 1
				total_diff += 1*diff_value

			print(col, value, df_ideal.loc[index,col], diff_value)

	print("Total count: ", total_count)
	print("Total diff: ", total_diff)
	print("Model trust score: ", total_diff/total_count*1.0)

if __name__ == '__main__':
	calculate()
