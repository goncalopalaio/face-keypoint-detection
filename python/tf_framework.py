import tensorflow as tf
import numpy as np
import math
import timeit
import pandas as pd

'''

Contains generic tensorflow training "pipeline"

- Heavily inspired by MG2033/MobileNet
'''
class Trainer:
	def __init__(self, session, model, arguments, summarizer):

		self.session = session
		self.model =  model
		self.args =  arguments
		self.summarizer =  summarizer
		self.saver = tf.train.Saver(max_to_keep = self.args.max_to_keep,
									keep_checkpoint_every_n_hours=1)
		
		
		self.data_loader = self.args.data_loader

		# Initialize model variables

		self.session.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

		# Restore model from a checkpoint

		self.model.load_pretrained_weights(self.session)

		latest_checkpoint = tf.train.latest_checkpoint(self.args.checkpoint_directory)
		if latest_checkpoint:
			log("loading ", latest_checkpoint)
			self.saver.restore(self.session, latest_checkpoint)
			log("loaded")
		else:
			log("no checkpoint to load")

		self._log_args()

	def _log_args(self):
		log("self.args.max_to_keep {}".format(self.args.max_to_keep))
		log("self.args.epochs {}".format(self.args.epochs))
		log("self.args.total_iterations_per_epoch {}".format(self.args.total_iterations_per_epoch))
		log("self.args.save_every_n_epochs {}".format(self.args.save_every_n_epochs))
		log("self.args.checkpoint_directory {}".format(self.args.checkpoint_directory))
		log("self.args.test_every_n_epochs {}".format(self.args.test_every_n_epochs))


	def train(self):
		total_epochs = self.args.epochs + 1
		current_epoch = self.model.global_epoch_tensor.eval(self.session) - 1


		epoch_logger = ProgressLogger(total_epochs)

		log("starting training from {} to {}".format(current_epoch, total_epochs))
		for epoch_nr in range(current_epoch, total_epochs):
			epoch_logger.update()

			loss_list = []
			accuracy_list = []

			total_iterations = self.args.total_iterations_per_epoch

			iteration_logger = ProgressLogger(total_iterations)

			current_step = 0
			batch_generator = self.data_loader.batch_generator(type='train', size = self.args.batch_size, total = total_iterations)
			for x, y in batch_generator:
				current_step = self.model.global_step_tensor.eval(self.session)

				iteration_logger.update(message='current_step: {}'.format(current_step))

				curr_feed_dict = {
					self.model.x: x,
					self.model.y: y,
					self.model.is_training: True
				}

				_, loss, accuracy, summaries_merged = self.session.run(
					[self.model.train_op, self.model.loss, self.model.accuracy, self.model.summaries_merged],
					feed_dict = curr_feed_dict
					)

				loss_list += [loss]
				accuracy_list += [accuracy]

				# Update global step inside the model
				self.model.global_step_assign_op.eval(
					session = self.session,
					feed_dict = {
						self.model.global_step_input: current_step + 1
					}
					)

				self.summarizer.add_summary(current_step, summaries_merged = summaries_merged)
					
			# Compute averages and save them in the summarizer for this batch

			summary_avg = {"loss": np.mean(loss_list), "accuracy": np.mean(accuracy_list)}
			self.summarizer.add_summary(current_step, summaries_dict = summary_avg)

			# Save a checkpoint

			if epoch_nr % self.args.save_every_n_epochs == 0:
				log("saving model in epoch {} ".format(epoch_nr))
				self.saver.save(self.session, self.args.checkpoint_directory, self.model.global_step_tensor)
				log("finished")

			# Check if it is time to evaluate with the validation set
			if current_epoch % self.args.test_every_n_epochs == 0:
				self.test("val")

			log("Updating global epoch to: {}".format(epoch_nr + 1))
			self.model.global_epoch_assign_op.eval(session=self.session, feed_dict={self.model.global_epoch_input: epoch_nr + 1})

	def test(self, test_type = 'val'):
		pass

class Summarizer:
	"""docstring for Summarizer"""
	def __init__(self, session, directory):
		self.session = session
		self.directory = directory

		self.tag_list = ['loss', 'accuracy']
		self.tags = []
		self.placeholders = {}
		self.ops = {}
		self.writer = tf.summary.FileWriter(self.directory, self.session.graph)

		with tf.variable_scope('train-summary-per-epoch'):
			for tag in self.tag_list:
				self.tags += tag
				self.placeholders[tag] = tf.placeholder('float32', None, name = tag)
				self.ops[tag] = tf.summary.scalar(tag, self.placeholders[tag])
	def add_summary(self, step, summaries_dict = None, summaries_merged = None):
		log("adding summary")
		if summaries_dict:
			summary_list = self.session.run(
				[self.ops[tag] for tag in summaries_dict.keys()],
				{self.placeholders[tag]: value for tag, value in summaries_dict.items()})
			for summary in summary_list:
				self.writer.add_summary(summary, step)

		if summaries_merged:
			self.writer.add_summary(summaries_merged, step)
			

def log(v, custom_end='\n'):
	print(v, end=custom_end)
					
class ProgressLogger:
	"""Prints message periodically"""
	def __init__(self, total = 100, interval = 5):
		self.total = total
		self.interval = interval
		self.reset()

	def reset(self):
		self.current = 0

	def update(self, message = None, increment = 1):
		self.current += increment
		if (self.current % self.interval) == 0:
			log(". ", custom_end = '')
			if message:
				log("\n",message)


"""

Test code - @note does not really belong here

"""

class InputArgs:
	"""
		docstring for Args
	"""
	def __init__(self, batch_size, max_to_keep, data_loader):
		self.batch_size = batch_size
		self.max_to_keep = max_to_keep
		self.checkpoint_directory = './checkpoints/demo'
		self.epochs = 5
		self.total_iterations_per_epoch = 1
		self.save_every_n_epochs = 2
		self.test_every_n_epochs = 1
		self.data_loader = data_loader

	def print(self):
		print(vars(self))

class DataLoader:
	"""
		docstring for DataLoader
	"""
	def __init__(self):
		pass
	def batch_generator(self, type='train', size = 0, total = 0):
		example_x_value = 1

		for t in range(total):
			x = []
			y = []

			for s in range(size):
				example_x_value += 1
				x.append(example_x_value) 
				y.append(math.sin(example_x_value))

			x = np.expand_dims(x, axis=0)
			y = np.expand_dims(y, axis=0)
			yield (x,y)


class SineNet:
	"""
		Example network to check correctness
	"""
	def __init__(self, args):
		self.args = args
		self.args.print()
	
	def load_pretrained_weights(self, session):
		pass

	def start(self):
		with tf.variable_scope("global_epoch"):
			self.global_epoch_tensor = tf.Variable(0, trainable = False, name = 'global_epoch')
			self.global_epoch_input = tf.placeholder('int32', None, name='global_epoch_input')
			self.global_epoch_assign_op = self.global_epoch_tensor.assign(self.global_epoch_input)

		with tf.variable_scope('global_step'):
			self.global_step_tensor = tf.Variable(-1, trainable = False, name = 'global_step')
			self.global_step_input = tf.placeholder('int32', None, name='global_step_input')
			self.global_step_assign_op = self.global_step_tensor.assign(self.global_step_input)

		with tf.variable_scope('input'):
			self.x = tf.placeholder(tf.float32,
									[1, self.args.batch_size])

			self.y = tf.placeholder(tf.int32, [1, self.args.batch_size])

			# note: there's EstimatorSpec() in tensorflow that could make this more flexible
			self.is_training = tf.placeholder(tf.bool)

		with tf.variable_scope('network'):
			hidden = tf.layers.dense(inputs = self.x, units = 100, activation=tf.nn.relu)
			self.network_output = tf.layers.dense(hidden, self.args.batch_size)

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

def main():
	data_loader = DataLoader()

	args = InputArgs(batch_size = 10, max_to_keep = 1, data_loader = data_loader)
	net = SineNet(args)
	net.start()

	session = tf.Session()
	summarizer = Summarizer(session, args.checkpoint_directory)
	trainer = Trainer(session, net, args, summarizer)
	trainer.train()


if __name__ == '__main__':
	main()