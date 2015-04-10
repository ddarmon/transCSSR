import numpy
import collections
import sys

def compute_precision(ts_true, ts_prediction, verbose = False):
	numerator = 0    # In precision, the numerator is the number of true
						 # positives which are predicted correctly.
	denominator = 0	 # In precision, the denominator is the total number
					 # of predicted positives.

	for char_ind in xrange(len(ts_true)):
		if ts_prediction[char_ind] == 'N':
			pass
		else:
			if int(ts_prediction[char_ind]) == 1: # We predicted a 1
				denominator += 1

				if int(ts_true[char_ind]) == 1: # We predicted a 1, and it is also the right
										   # answer.
					numerator += 1

	if denominator == 0:
		if verbose:
			print 'Warning: you didn\'t predict any tweets! By convention, set precision to 1.'

		precision = 1
	else:
		precision = numerator/float(denominator)

	return precision

def compute_recall(ts_true, ts_prediction, verbose = False):
	numerator = 0    # In precision, the numerator is the number of true
					 # positives which are predicted correctly.
	denominator = 0	 # In precision, the denominator is the total number
					 # of true positives.

	for char_ind in xrange(len(ts_true)):
		if int(ts_true[char_ind]) == 1: # The true value is a 1
			denominator += 1
			
			if ts_prediction[char_ind] == 'N':
				pass
			else:
				if int(ts_prediction[char_ind]) == 1: # We predicted a 1, and it is also the right
										   		 # answer.
					numerator += 1

	if denominator == 0:
		if verbose:
			print 'Warning: no tweets were in this day! By convention, set recall to 1.'

		recall = 1
	else:
		recall = numerator/float(denominator)

	return recall

def compute_tv(ts_true, ts_prediction):
	running_sum = 0
	count = 0
	
	for char_ind in xrange(len(ts_true)):
		if ts_prediction[char_ind] == None:
			pass
		else:
			running_sum += numpy.abs(int(ts_true[char_ind]) - ts_prediction[char_ind]) + numpy.abs((1 - int(ts_true[char_ind])) - (1 - ts_prediction[char_ind]))
			count += 1

	tv = 0.5*running_sum / float(count)

	return tv

def compute_metrics(ts_true, ts_prediction, metric = None):
	# choices: 'accuracy', 'precision', 'recall', 'F'

	if metric == None or metric == 'accuracy': # By default, compute accuracy rate.
		correct = 0

		for char_ind in xrange(len(ts_true)):
			if ts_prediction[char_ind] == 'N': # We didn't predict, so don't count towards correct.
				pass
			else:
				if int(ts_true[char_ind]) == int(ts_prediction[char_ind]):
					correct += 1

		accuracy_rate = correct / float(len(ts_true))

		return accuracy_rate
	elif metric == 'precision':
		precision = compute_precision(ts_true, ts_prediction)

		return precision

	elif metric == 'recall':
			
		recall = compute_recall(ts_true, ts_prediction)

		return recall

	elif metric == 'F':
		precision = compute_precision(ts_true, ts_prediction)
		recall = compute_recall(ts_true, ts_prediction)

		if (precision + recall) == 0:
			F = 0
		else:
			F = 2*precision*recall/float(precision + recall)

		return F
	elif metric == 'tv':
		tv = compute_tv(ts_true, ts_prediction)
		
		return tv
	else:
		print "Please choose one of \'accuracy\', \'precision\', \'recall\', or \'F\'."

		return None