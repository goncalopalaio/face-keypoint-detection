import pandas as pd
from PIL import Image
import time
import multiprocessing as mp

def write_image(idx, image_row, path_image):
	w = 96
	h = 96

	vals = [int(v) for v in image_row.split(' ')]

	im = Image.new('RGB', (w, h), "black")
	pixels = im.load()
	for i in range(w):
		for j in range(h):
			val = vals[i + j * w]
			pixels[i, j] = (val, val, val)
	full_path = path_image + str(idx) + "_img.png" 
	print("saving to ", full_path)
	im.save(full_path)

def linear_process(data, path_image):
	number_images = len(data.Image)
	print("Number of images: ", number_images)

	for i in range(number_images):
		write_image(i, data.Image[i], path_image)

def multi_process(data):
	pool = mp.Pool(processes = 8)
	[pool.apply(write_image, args=(idx, img)) for idx, img in enumerate(data.Image)]

if __name__ == '__main__':
	path_dataset_folder = "../dataset/"
	path_dataset_train = path_dataset_folder + "training.csv"
	path_dataset_test = path_dataset_folder + "test.csv"

	linear_process(pd.read_csv(path_dataset_test), "../images/test/")
	linear_process(pd.read_csv(path_dataset_train), "../images/train/")