class InListPredicate(object):

	def __init__(self, ls):
		self.ls = ls

	def is_true(self, x):
		return x in self.ls
