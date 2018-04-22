import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
import utils as util

class BatchDataLoader:
	"""Loads data in batches"""
	def __init__(self, train_val_split = 0.7, randomize_indices = True, load_one_channel = False, skip_missing_values = False):
		# let's just assume the images are there
		self.total_train = 7049
		self.total_test = 1783
		self.train_path_format = "../images/train/{}_img.png"
		self.test_path_format = "../images/test/{}_img.png"
		self.load_one_channel = load_one_channel

		path_dataset_folder = "../dataset/"
		path_dataset_train = path_dataset_folder + "training.csv"
		path_dataset_train_simplified = path_dataset_folder + "training_simplified.csv"
		path_dataset_test = path_dataset_folder + "test.csv"

		self.train = pd.read_csv(path_dataset_train_simplified)
		self.test = pd.read_csv(path_dataset_test)

		if skip_missing_values:
			print("Removing missing values :: original Train", np.shape(self.train))
			l, cols = np.shape(self.train)
			keep = []
			for i in range(l):
				if not np.isnan(self.train.loc[i][1:]).any():
					keep.append(i)
				
			self.train = self.train.loc[keep][1:]

			#self.train = self.train[~np.isnan(self.train).any()]
			self.total_train = np.shape(self.train)[0]
			print("Removed missing values :: new Train", np.shape(self.train))
			print("Removed missing values :: New total for train (and val)", self.total_train)

		self.randomize_indices = randomize_indices
		self.train_val_split = train_val_split
		self.test_indices = []
		self.train_indices = []
		self.val_indices = []


	def _build_indices(self, max, randomize_indices = False):
		if randomize_indices:
			return np.random.permutation(max)
		else:
			return [i for i in range(max)] # @note this is kinda pointless

	def _rebuild_indices(self, data_type):
		# Train has 7049 images
		# Test has 1783 images -> 27124 keypoints to predict
		if (data_type == 'train' or data_type == 'val') and len(self.train_indices) == 0:
			self.train_indices = self._build_indices(self.total_train, self.randomize_indices)
			# split validation accordingly
			self.val_indices = self.train_indices[math.floor(self.total_train * self.train_val_split):]
			self.train_indices = self.train_indices[:math.floor(self.total_train * self.train_val_split)]

		if data_type == 'test' and len(self.test_indices) == 0:
			self.test_indices = self._build_indices(self.total_test, False)

	def get_train_data(self):
		self._rebuild_indices('train')
		return next(self.train_batch_generator(len(self.train_indices), 1))
	
	def get_val_data(self):
		self._rebuild_indices('val')
		return next(self.val_batch_generator(len(self.val_indices)))

	def get_test_data(self):
		self._rebuild_indices('test')
		return next(self.test_batch_generator(len(self.test_indices)))

	def train_batch_generator(self, batch_size, total = None):
		self._rebuild_indices('train')
		#print("Training",batch_size, total)
		if total is None:
			total =  len(self.train_indices) // batch_size

		print("batch_size: {} total batches {}".format(batch_size, total))

		return self._batch_generator('train', batch_size, total)

	def val_batch_generator(self, batch_size):
		self._rebuild_indices('val')

		total =  len(self.val_indices) // batch_size
		print("batch_size: {} total batches {}".format(batch_size, total))

		return self._batch_generator('val', batch_size, total)

	def test_batch_generator(self, batch_size):
		self._rebuild_indices('test')

		total =  len(self.test_indices) // batch_size
		print("batch_size: {} total batches {}".format(batch_size, total))
		return self._batch_generator('test', batch_size, total)


	def _batch_generator(self, data_type='train', size = 0, total = 0):
		timer = util.Timer().start("time per {} [{};{}] batch".format(data_type, size, total))	
		str_template = None
		points_data = None
		indices = None

		if data_type == 'train':
			str_template = self.train_path_format
			points_data = self.train
			indices = self.train_indices

			# when training, let the same items be reused but randomize the indices
			if size * total > len(self.train_indices):
				self.train_indices = np.random.shuffle(self.train_indices)

		elif data_type == 'val':
			str_template = self.train_path_format
			points_data = self.train
			indices = self.val_indices
		else:
			str_template = self.test_path_format
			points_data = self.test
			indices = self.test_indices

		len_indices = len(indices)
		i = 0
		for t in range(total):
			x = []
			y = []
			for s in range(size):
				# let other types blow up the program if they try to exceed the limit
				if data_type == 'train' and i >= len_indices:
					i = 0

				idx = indices[i]
				img = plt.imread(str_template.format(idx))
				if self.load_one_channel:
					img = img[:,:,0]
					print(np.shape(img))
				x.append(img)
				y.append(self.train.loc[idx][1:])

				i += 1
			yield x,y

		timer.stop()



def test_totals():
	params = [0.7, 0.5]
	for split in params:
		loader = BatchDataLoader(train_val_split = split, randomize_indices = False)

		val = loader.get_val_data()
		b, n = np.shape(val)
		assert_equals(b, 2)
		assert_equals(n, len(loader.val_indices))

		train = loader.get_train_data()
		b, n = np.shape(train)
		assert_equals(b, 2)
		assert_equals(n, len(loader.train_indices))
	
	params = [
		(1.0, 7100),
		(0.7, 7100),
		(0.2, 234)
	]
	
	for split, batch_size in params:
		loader = BatchDataLoader(train_val_split = split, randomize_indices = False)
		total_train = loader.total_train
		number_of_batches = 2 # might be more than the images we have, randomize indices inside train and loop
		train = [e for e in loader.train_batch_generator(batch_size, 2)]
		
	params = [
		(1.0, False),
		(0.7, False),
		(0.3, False),
		(0.2, False),
		(1.0, True ),
		(0.7, True ),
		(0.3, True ),
		(0.2, True ),
	]

	for split, randomize in params:
		loader = BatchDataLoader(train_val_split = split, randomize_indices = randomize)
		total_train = loader.total_train
		total_test = loader.total_test

		train = [e for e in loader.train_batch_generator(1)]
		val = [e for e in loader.val_batch_generator(1)]
		assert_equals(len(train) + len(val), total_train) 

		test = [e for e in loader.test_batch_generator(1)]
		assert_equals(len(test), total_test)

def assert_equals(actual, expected):
	print("actual: ", actual, " expected: ", expected)
	assert actual == expected

def test():
	test_totals()

if __name__ == '__main__':
	pass
	#test()