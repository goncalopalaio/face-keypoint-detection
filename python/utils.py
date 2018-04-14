import timeit

def log(v, custom_end='\n'):
	print(v, end=custom_end)

class Timer:
	def __init__(self, message = ""):
		self.start_time = None
		self.message = message
	def start(self, message=""):
		self.start_time = timeit.default_timer()
		self.message = message
		return self

	def stop(self):
		elapsed = timeit.default_timer() - self.start_time
		log("timer: {} {:.1f} millis ({:.6f} seconds)".format(self.message, elapsed * 1000.0, elapsed))

#@timing
def timing(f):
	def wrap(*args):
		time1 = time.time()
		ret = f(*args)
		time2 = time.time()
		print('%s function took %0.3f ms' % (f.__name__, (time2-time1)*1000.0))
		return ret
	return wrap
