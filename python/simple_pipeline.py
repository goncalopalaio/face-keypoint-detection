import common as cm
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

SMALL_DATASET = True

def breakdown(x_data, y_data):
	print("\t-> ", np.shape(x_data), " -> ", np.shape(y_data))
	#print("x-> ", x_data)
	#print("y-> ", y_data)
	n, w, h, d = np.shape(x_data)
	x = []
	y = []
	for i in range(n):
		# ignoring lines with missing values
		if np.isnan(y_data[i]).any():
			continue
		#print("Y: ", len(y_data[i]))
		x.append(x_data[i].flatten())
		
		#norm_y = y_data[i] / 95.0
		norm_y = y_data[i]
		
		#print("normy-> ", norm_y)
	
		y.append(norm_y)
		
		#y_imputed = y_data[i]
		#y_imputed[np.isnan(y_imputed)] = 0
		#y.append(y_imputed)

	print(" <- ", np.shape(x), " -> ", np.shape(y))

	return x, y

def metrics(y_true, y_pred, text=''):
	mean_squared_error_res  = mean_squared_error(y_true, y_pred)
	mean_absolute_error_res = mean_absolute_error(y_true, y_pred)
	print(text, "mean_squared_error", mean_squared_error_res, "mean_squared_error", mean_absolute_error_res)
	return mean_squared_error_res, mean_absolute_error_res
def extract_point(data, index):
	res = []
	rows = np.shape(data)[0]
	for i in range(rows):
		res.append(data[i][index])
	return res

def get_train_data(loader):
	if SMALL_DATASET:
		x,y = next(loader.train_batch_generator(10, 1)) #get_train_data(loader)
	else:
		x,y = loader.get_train_data()
	return breakdown(x, y)

def get_val_data(loader):
	if SMALL_DATASET:
		x,y = next(loader.val_batch_generator(100))
	else:
		x,y = loader.get_val_data()
	return breakdown(x,y)

def get_test_data(loader):
	if SMALL_DATASET:
		x,y = next(loader.test_batch_generator(100))
	else:
		x,y = loader.get_test_data() #
	return breakdown(x,y)

def save_linear_regression_data(estimators):
	with open("linear_regression_weights", 'w') as f:
		for i, est in enumerate(estimators):
			f.write("# estimator {} coef {} intercept {}".format(i, est.coef_.shape, est.intercept_.shape))
			f.write('# coef\n')
			# @note couldn't figure out how to use np.savetxt quickly
			for v in est.coef_:
				f.write("{0:.6f}, ".format(v))
			f.write('\n# intercept\n')
			f.write("{0:.6f} ".format(est.intercept_))
			f.write('\n# end\n')
	# quick and dirty json format
	with open("linear_regression_weights_json", 'w') as f:
		f.write('{')
		len_estimators = len(estimators)
		for i, est in enumerate(estimators):
			f.write('"coefs{}":['.format(i))

			ci = 0
			len_coefs = len(est.coef_)
			for v in est.coef_:
				if ci == len_coefs - 1:
					f.write("{0:.6f}".format(v))
				else:
					f.write("{0:.6f},".format(v))
				ci+=1
				#break
			f.write('],')
			if i == len_estimators - 1:
				f.write('"intercept{0}":{1:.6f}'.format(i, est.intercept_))
			else:
				f.write('"intercept{0}":{1:.6f},'.format(i, est.intercept_))
		f.write('}')
	

def pipeline(loader):
	# load train data
	x_train, y_train = get_train_data(loader)

	# params
	print_train_metrics = False
	print_validation_metrics = True
	print_test_metrics = False

	# train model
	estimators = []
	for i in range(30):
		model = LinearRegression()
		
		points = extract_point(y_train, i)
		est = model.fit(x_train, points)
		estimators.append(est)

		if print_train_metrics:
			metrics(points, est.predict(x_train), "point " + str(i) +" train: ")

	# validation
	if print_validation_metrics:
		x_val, y_val = get_val_data(loader)
		acc_mean_squared_error, acc_mean_absolute_error = (0,0)
		for i in range(30):
			mean_squared_error, mean_absolute_error = metrics(extract_point(y_val, i), estimators[i].predict(x_val), "point " + str(i) +" validation: ")
			acc_mean_squared_error += mean_squared_error
			acc_mean_absolute_error += mean_absolute_error
		print("Average acc_mean_squared_error: {} Average mean_absolute_error: {}".format(acc_mean_squared_error/30.0,  acc_mean_absolute_error/30.0))

	# testing
	if print_test_metrics:
		x_test, y_test = get_test_data(loader)
		#res = pd.DataFrame({"RowId": row_id, "Location": pred_location}) # write output forcing a specific order
		#	res[["RowId", "Location"]].to_csv('submission.csv', index=False)
		#

		test_results = np.zeros((len(x_test), 30))
		print("test rests: ", np.shape(test_results))
		for i in range(30):
			predictions = estimators[i].predict(x_test)
			test_results[:,i] = predictions

			print("params: ", i, " --> ", estimators[i].get_params())
			print("params: ", i, " coef --> ", np.shape(estimators[i].coef_), " ", estimators[i].coef_)
			print("params: ", i, " intercept --> ", estimators[i].intercept_)


		# Decision function
		# read the weights and implement this
		print("INPUT: ", x_test[0])
		res = estimators[i].predict(x_test[0].reshape(1,-1))
		res2 = np.dot(x_test[0].T, estimators[i].coef_)  + estimators[i].intercept_
		print("OUTPUT TEST: res: ", res, " res2: ", res2)

		save_linear_regression_data(estimators)
		results_header = "left_eye_center_x,left_eye_center_y,right_eye_center_x,right_eye_center_y,left_eye_inner_corner_x,left_eye_inner_corner_y,left_eye_outer_corner_x,left_eye_outer_corner_y,right_eye_inner_corner_x,right_eye_inner_corner_y,right_eye_outer_corner_x,right_eye_outer_corner_y,left_eyebrow_inner_end_x,left_eyebrow_inner_end_y,left_eyebrow_outer_end_x,left_eyebrow_outer_end_y,right_eyebrow_inner_end_x,right_eyebrow_inner_end_y,right_eyebrow_outer_end_x,right_eyebrow_outer_end_y,nose_tip_x,nose_tip_y,mouth_left_corner_x,mouth_left_corner_y,mouth_right_corner_x,mouth_right_corner_y,mouth_center_top_lip_x,mouth_center_top_lip_y,mouth_center_bottom_lip_x,mouth_center_bottom_lip_y"
		np.savetxt("test_results.csv", test_results, fmt="%d", delimiter=",", header = results_header)


	return estimators

def main():
	loader = cm.BatchDataLoader(train_val_split = 0.7, randomize_indices = True)
	# run
	estimators = pipeline(loader)

if __name__ == '__main__':
	main()

#def train(loader):
#	estimator = LinearRegression()
#	x_train, y_train = loader.get_train_data()	
#	print("train: ", np.shape(x_train), " -> ", np.shape(y_train))
#	x_train, y_train = breakdown(x_train, y_train)
#	print("train after breakdown: ", np.shape(x_train), " -> ", np.shape(y_train))
#
#	estimator = estimator.fit(x_train, y_train)
#	metrics(y_train, estimator.predict(x_train), "train:")
#
#	return estimator
#
#def validate_model(loader, estimator):
#	x_val, y_val = loader.get_val_data()
#	print("validation: ", np.shape(x_val), " -> ", np.shape(y_val))
#	x_val, y_val = breakdown(x_val, y_val)
#	print("validation after breakdown: ", np.shape(x_val), " -> ", np.shape(y_val))
#
#	metrics(y_val, estimator.predict(x_val), "validation:")
#	
#def clip_to_boundaries(val):
#	if val < 0:
#		return 0
#	if val > 96:
#		return 96
#	return val
#
#def write_test_results_all_points(loader, estimator):
#	x_test, y_test = loader.get_test_data()
#	print("test: ", np.shape(x_test), " -> ", np.shape(y_test))
#	x_test, y_test = breakdown(x_test, y_test)
#	print("test after breakdown: ", np.shape(x_test), " -> ", np.shape(y_test))
#	predictions = estimator.predict(x_test)
#	row_id = []
#	pred_location = []
#	i = 0
#	line, point = np.shape(predictions)
#	print("predictions: ", np.shape(predictions))
#	for l in range(line):
#		i+=1
#		for p in range(point):
#			row_id.append(i)
#			pred_location.append(clip_to_boundaries(predictions[l][p].astype(int)))
#			
#	res = pd.DataFrame({"RowId": row_id, "Location": pred_location}) # write output forcing a specific order
#	res[["RowId", "Location"]].to_csv('submission.csv', index=False)
#