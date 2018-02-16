from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import os.path
import sys
import CNNTrain
import Preprocessor

#Loads the CNN stored in the CNNModel file. If the file's not found, train one anew!
if not os.path.exists('CNNModel.h5'):
	CNNTrain.build_model()
model = load_model('CNNModel.h5')

class Classifier:
	def __init__(self):
		self._target_size = (56,56)

	def process_image(self, img_path):
		self._img_path = img_path
		image_array = image.img_to_array(image.load_img(img_path, target_size=self._target_size))
		average_edgeness = np.zeros((image_array.shape[0], image_array.shape[1]))
		average_cornerness = np.zeros((image_array.shape[0], image_array.shape[1]))

		#On a channel by channel basis computes the features values of all points
		for color_channel in xrange(image_array.shape[2]):
			channel_array = np.zeros((image_array.shape[0], image_array.shape[1]))
			for x in xrange(len(channel_array)):
				for y in xrange(len(channel_array[0])):
					channel_array[x][y] = image_array[x][y][color_channel]
			#Could potentially crash if not given a 56 x 56 image as input for this method
			channel_edgeness, channel_cornerness = Preprocessor.process_image_sd2(channel_array)
			for x in xrange(len(channel_array)):
				for y in xrange(len(channel_array[0])):
					#Next 2 lines utilize "magic numbers" that are the maximum that was
					#found over the MNIST dataset, effectively regularizing the data
					average_edgeness[x][y] += channel_edgeness[x][y] / 0.634789974395 / image_array.shape[2]
					average_cornerness[x][y] += channel_cornerness[x][y] / 0.16428281162 / image_array.shape[2]

		#Scales image size down to 28 by 28 to fit the model
		ensmalled_edgeness = np.zeros((image_array.shape[0] / 2, image_array.shape[1] / 2))
		ensmalled_cornerness = np.zeros((image_array.shape[0] / 2, image_array.shape[1] / 2))
		for x in xrange(len(ensmalled_edgeness)):
			for y in xrange(len(ensmalled_edgeness[0])):
				ensmalled_edgeness[x][y] += average_edgeness[2 * x][2 * y]
				ensmalled_edgeness[x][y] += average_edgeness[2 * x + 1][2 * y]
				ensmalled_edgeness[x][y] += average_edgeness[2 * x][2 * y + 1]
				ensmalled_edgeness[x][y] += average_edgeness[2 * x + 1][2 * y + 1]
				ensmalled_edgeness[x][y] /= 4.0
				ensmalled_cornerness[x][y] += average_cornerness[2 * x][2 * y]
				ensmalled_cornerness[x][y] += average_cornerness[2 * x + 1][2 * y]
				ensmalled_cornerness[x][y] += average_cornerness[2 * x][2 * y + 1]
				ensmalled_cornerness[x][y] += average_cornerness[2 * x + 1][2 * y + 1]
				ensmalled_cornerness[x][y] /= 4.0
		self._image_array = np.expand_dims(np.asarray([ensmalled_edgeness, ensmalled_cornerness]), axis=0)

	def classify_image(self):
		self._predictions = model.predict(self._image_array)

	def get_classification(self):
		max_index = 0
		for index in xrange(1, len(self._predictions[0])):
			if self._predictions[0][index] > self._predictions[0][max_index]:
				max_index = index
		return max_index

	def pipeline(self, img_path):
		self.process_image(img_path)
		self.classify_image()
		return self.get_classification()

if len(sys.argv) == 2:
	classifier = Classifier()
	print "It's a %d!" % classifier.pipeline(sys.argv[1])
else:
	print "Error: must provide exactly 1 command-line argument for the (relative) file path"