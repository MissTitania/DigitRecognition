import pickle
import matplotlib.cm as cm
import matplotlib.pyplot as plot
import numpy as np

#Expects the input to always be a 28 by 28 square image of a single color channel
def process_image_sd1(image):
	edgeness = np.zeros((28, 28))
	cornerness = np.zeros((28, 28))
	for x in xrange(28):
		for y in xrange(28):
			total_weight = 0
			structure_tensor = np.asmatrix(np.zeros((2, 2)))
			for x_shift in range(-3, 4):
				for y_shift in range(-3, 4):
					if 0 <= x + x_shift < 28 and 0 <= y + y_shift < 28:
						if 0 == x + x_shift:
							x_derivative = image[1, y + y_shift] - image[0, y + y_shift]
						elif 27 == x + x_shift:
							x_derivative = image[27, y + y_shift] - image[26, y + y_shift]
						else:
							x_derivative = (image[x + x_shift + 1, y + y_shift] - image[x + x_shift - 1, y + y_shift]) / 2
						if 0 == y + y_shift:
							y_derivative = image[x + x_shift, 1] - image[x + x_shift, 0]
						elif 27 == y + y_shift:
							y_derivative = image[x + x_shift, 27] - image[x + x_shift, 26]
						else:
							y_derivative = (image[x + x_shift, y + y_shift + 1] - image[x + x_shift, y + y_shift - 1]) / 2
						weight = np.exp(-(x_shift * x_shift + y_shift * y_shift) / 2.0)
						total_weight += weight
						structure_tensor[0, 0] += weight * x_derivative * x_derivative
						structure_tensor[0, 1] += weight * x_derivative * y_derivative
						structure_tensor[1, 0] += weight * x_derivative * y_derivative
						structure_tensor[1, 1] += weight * y_derivative * y_derivative
			eigenvalues = np.absolute(np.linalg.eigvals(structure_tensor / total_weight))
			edgeness[x, y] = max(eigenvalues)
			cornerness[x, y] = min(eigenvalues)
	return (edgeness, cornerness)

#Same as above, but for twice as large a standard deviation on the Gaussian filter because of double the image size
def process_image_sd2(image):
	edgeness = np.zeros((56, 56))
	cornerness = np.zeros((56, 56))
	for x in xrange(56):
		for y in xrange(56):
			total_weight = 0
			structure_tensor = np.asmatrix(np.zeros((2, 2)))
			for x_shift in range(-6, 7):
				for y_shift in range(-6, 7):
					if 0 <= x + x_shift < 56 and 0 <= y + y_shift < 56:
						if 0 == x + x_shift:
							x_derivative = image[1, y + y_shift] - image[0, y + y_shift]
						elif 55 == x + x_shift:
							x_derivative = image[55, y + y_shift] - image[54, y + y_shift]
						else:
							x_derivative = (image[x + x_shift + 1, y + y_shift] - image[x + x_shift - 1, y + y_shift]) / 2
						if 0 == y + y_shift:
							y_derivative = image[x + x_shift, 1] - image[x + x_shift, 0]
						elif 55 == y + y_shift:
							y_derivative = image[x + x_shift, 55] - image[x + x_shift, 54]
						else:
							y_derivative = (image[x + x_shift, y + y_shift + 1] - image[x + x_shift, y + y_shift - 1]) / 2
						weight = np.exp(-(x_shift * x_shift + y_shift * y_shift) / 8.0)
						total_weight += weight
						structure_tensor[0, 0] += weight * x_derivative * x_derivative
						structure_tensor[0, 1] += weight * x_derivative * y_derivative
						structure_tensor[1, 0] += weight * x_derivative * y_derivative
						structure_tensor[1, 1] += weight * y_derivative * y_derivative
			eigenvalues = np.absolute(np.linalg.eigvals(structure_tensor / total_weight))
			edgeness[x, y] = max(eigenvalues)
			cornerness[x, y] = min(eigenvalues)
	return (edgeness, cornerness)

def process_mnist():
	print "Loading MNIST data set..."
	training_set, validation_set, test_set = pickle.load(open('MNIST.pkl', 'rb'))
	training_input, training_output = training_set
	validation_input, validation_output = validation_set
	test_input, test_output = test_set

	print "Beginning on training set:"
	edgeness = np.zeros((len(training_input), 28, 28))
	cornerness = np.zeros((len(training_input), 28, 28))
	for image_number in xrange(len(training_input)):
		if image_number % 100 == 0:
			print "Processing image %d..." % image_number
		edgeness[image_number], cornerness[image_number] = process_image_sd1(training_input[image_number].reshape((28, 28)))
	print "Pickling processed training compilation..."
	pickle.dump([edgeness, cornerness, training_output], open('TrainingSet.pkl', mode='w'))

	print "Beginning on validation set:"
	edgeness = np.zeros((len(validation_input), 28, 28))
	cornerness = np.zeros((len(validation_input), 28, 28))
	for image_number in xrange(len(validation_input)):
		if image_number % 100 == 0:
			print "Processing image %d..." % image_number
		edgeness[image_number], cornerness[image_number] = process_image_sd1(validation_input[image_number].reshape((28, 28)))
	print "Pickling processed validation compilation..."
	pickle.dump([edgeness, cornerness, validation_output], open('ValidationSet.pkl', mode='w'))

	print "Beginning on test set:"
	edgeness = np.zeros((len(test_input), 28, 28))
	cornerness = np.zeros((len(test_input), 28, 28))
	for image_number in xrange(len(test_input)):
		if image_number % 100 == 0:
			print "Processing image %d..." % image_number
		edgeness[image_number], cornerness[image_number] = process_image_sd1(test_input[image_number].reshape((28, 28)))
	print "Pickling processed test compilation..."
	pickle.dump([edgeness, cornerness, test_output], open('TestSet.pkl', mode='w'))

if __name__ == '__main__':
	process_mnist()