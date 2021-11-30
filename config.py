# config.py

class Config(object):
	def __init__(self, model_name=None):
		self.model_name = model_name

		if self.model_name == 'roberta-base':
			self.max_epochs = 30
			self.lr = 0.00001
			self.batch_size = 128
			self.cuda = True
			self.trial_size = -1
			self.max_len = 50
		elif self.model_name == 'roberta-ft':
			self.max_epochs = 30
			self.lr = 0.00001
			self.batch_size = 128
			self.cuda = True
			self.trial_size = -1
			self.max_len = 50
		else:
			self.max_epochs = 30
			self.lr = 0.0001
			self.batch_size = 128
			self.cuda = True
			self.trial_size = -1
			self.max_len = 50