import framework as fm
import tensorflow as tf
import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
import os


class FaceNet:
	def __init__(self, args):
		self.args = args
		self.args.print()
	
	def load_pretrained_weights(self, session):
		pass

	def start(self):
		with tf.variable_scope("global_epoch"):
			self.global_epoch_tensor = tf.Variable(-1, trainable = False, name = 'global_epoch')
			self.global_epoch_input = tf.placeholder('int32', None, name='global_epoch_input')
			self.global_epoch_assign_op = self.global_epoch_tensor.assign(self.global_epoch_input)

		with tf.variable_scope('global_step'):
			self.global_step_tensor = tf.Variable(-1, trainable = False, name = 'global_step')
			self.global_step_input = tf.placeholder('int32', None, name='global_step_input')
			self.global_step_assign_op = self.global_step_tensor.assign(self.global_step_input)

		with tf.variable_scope('input'):
			self.x = tf.placeholder(tf.float32,
									[self.args.batch_size, 96, 96, 3])

			self.y = tf.placeholder(tf.int32, [self.args.batch_size, 30])

			# note: there's EstimatorSpec() in tensorflow that could make this more flexible
			self.is_training = tf.placeholder(tf.bool)

		with tf.variable_scope('network'):
			flat = tf.reshape(self.x, [-1, 96 * 96 * 3])
			self.network_output = tf.layers.dense(inputs = flat, units = 30, activation=tf.nn.relu)

		with tf.variable_scope('output'):
			#predictions = tf.squeeze(self.network_output, 1)

			self.loss = tf.losses.mean_squared_error(self.y, self.network_output)

			# todo remove this
			self.accuracy = self.loss

			self.train_op = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(self.loss)

		with tf.name_scope("train-summary-per-iteration"):
			tf.summary.scalar('loss', self.loss)
			#tf.summary.scalar('acc', self.accuracy)
			self.summaries_merged = tf.summary.merge_all()
		
class FaceNetInput:
	"""docstring for FaceNetInput"""
	def __init__(self):
		self.batch_size = 4
		self.learning_rate = 0.001
		self.max_to_keep = 5
		self.checkpoint_directory = './checkpoints/facenet'
		self.epochs = 1
		self.total_iterations_per_epoch = 1
		self.save_every_n_epochs = 2
		self.test_every_n_epochs = 1
		self.data_loader = FaceNetDataLoader()

	def print(self):
		print(vars(self))
		
def main():
	main_timer = fm.Timer().start("main_timer")
	
	args = FaceNetInput()

	net = FaceNet(args)
	net.start()

	session = tf.Session()
	summarizer = fm.Summarizer(session, args.checkpoint_directory)
	trainer = fm.Trainer(session, net, args, summarizer)
	trainer.train()

	main_timer.stop()

if __name__ == '__main__':
	main()
