import common as cm
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

def breakdown(x_data, y_data):
	n, w, h, d = np.shape(x_data)
	x = []
	y = []
	for i in range(n):
		#if np.isnan(y_data[i]).any():
		#	continue
		x.append(x_data[i].flatten())

		y_imputed = y_data[i]
		y_imputed[np.isnan(y_imputed)] = 0
		y.append(y_imputed)

	return x, y

def metrics(y_true, y_pred, text=''):
	print(text, "mean_squared_error", mean_squared_error(y_true, y_pred))
	print(text, "mean_absolute_error", mean_absolute_error(y_true, y_pred))

def train(loader):
	estimator = LinearRegression()
	x_train, y_train = loader.get_train_data()	
	print("train: ", np.shape(x_train), " -> ", np.shape(y_train))
	x_train, y_train = breakdown(x_train, y_train)
	print("train after breakdown: ", np.shape(x_train), " -> ", np.shape(y_train))

	estimator = estimator.fit(x_train, y_train)
	metrics(y_train, estimator.predict(x_train), "train:")

	return estimator

def validate_model(loader, estimator):
	x_val, y_val = loader.get_val_data()
	print("validation: ", np.shape(x_val), " -> ", np.shape(y_val))
	x_val, y_val = breakdown(x_val, y_val)
	print("validation after breakdown: ", np.shape(x_val), " -> ", np.shape(y_val))

	metrics(y_val, estimator.predict(x_val), "validation:")
	
def clip_to_boundaries(val):
	if val < 0:
		return 0
	if val > 96:
		return 96
	return val

def write_test_results_all_points(loader, estimator):
	x_test, y_test = loader.get_test_data()
	print("test: ", np.shape(x_test), " -> ", np.shape(y_test))
	x_test, y_test = breakdown(x_test, y_test)
	print("test after breakdown: ", np.shape(x_test), " -> ", np.shape(y_test))
	predictions = estimator.predict(x_test)
	row_id = []
	pred_location = []
	i = 0
	line, point = np.shape(predictions)
	print("predictions: ", np.shape(predictions))
	for l in range(line):
		i+=1
		for p in range(point):
			row_id.append(i)
			pred_location.append(clip_to_boundaries(predictions[l][p].astype(int)))
			
	res = pd.DataFrame({"RowId": row_id, "Location": pred_location}) # write output forcing a specific order
	res[["RowId", "Location"]].to_csv('submission.csv', index=False)

def main():
	
	# Preprocess
	loader = cm.BatchDataLoader(train_val_split = 0.7, randomize_indices = True)

	# Train
	estimator = train(loader)

	# Validate
	#validate_model(loader, estimator)
	
	# Test and write output
	write_test_results_all_points(loader, estimator)


if __name__ == '__main__':
	main()