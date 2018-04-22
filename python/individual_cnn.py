import common as cm
import numpy as np
import pandas as pd
import tensorflow as tf

def attach_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)
    		
class BaseCnn:
	"""docstring for BaseCnn"""
	def __init__(self, loader):
		self.arg = {}
		self.loader = loader
	def build(self):

		with tf.variable_scope('input'):
			self.x = tf.placeholder(tf.float32, [None, 96, 96])
			self.y = tf.placeholder(tf.int32, [None, 30])
			self.training = tf.placeholder(tf.bool)
		with tf.variable_scope('network'):
			flat = tf.reshape(self.x, [-1, 96 * 96 * 3])
			self.network_output = tf.layers.dense(inputs = flat, units = 30, activation=tf.nn.relu)

		with tf.variable_scope('output'):
			self.loss = tf.losses.mean_squared_error(self.y, self.network_output)

			self.train_op = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(self.loss)

		with tf.name_scope("summary"):
			attach_summaries(self.loss)
			self.summaries_merged = tf.summary.merge_all()

		return self
	def train(self):
		total_epoch = 4
		last_epoch = 0
		batch_size = 2
		with tf.Session() as session:
			for epoch in range(last_epoch, total_epoch):
				batch_generator = self.loader.train_batch_generator(batch_size)
				for batch_x, batch_y in batch_generator:
					curr_feed_dict = {
						self.x : batch_x,
						self.y : batch_y,
						self.training : True
					}
					session.run([], feed_dict = curr_feed_dict)



	

def run_pipeline(loader):
	model = BaseCnn(loader).build()
	model.train()

if __name__ == '__main__':
	loader = cm.BatchDataLoader(train_val_split = 0.7,
								randomize_indices = True,
								load_one_channel = True,
								skip_missing_values = True)
	estimators = run_pipeline(loader)