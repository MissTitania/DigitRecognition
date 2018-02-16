import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
import os.path
import Regularizer

def build_model():
	#Check for existence of regularized training data.  If not found, construct it
	if not os.path.exists('TrainingSetRegularized.pkl'):
		Regularizer.regularize_data()

	#Load data
	print "Loading training data..."
	training_edgeness, training_cornerness, training_answers = pickle.load(open('TrainingSetRegularized.pkl', 'r'))
	validation_edgeness, validation_cornerness, validation_answers = pickle.load(open('ValidationSetRegularized.pkl', 'r'))
	test_edgeness, test_cornerness, test_answers = pickle.load(open('TestSetRegularized.pkl', 'r'))

	#Reshape data
	print "Reconfiguring data format..."
	training_input = []
	for index in xrange(len(training_edgeness)):
		training_input.append([training_edgeness[index], training_cornerness[index]])
	validation_input = []
	for index in xrange(len(validation_edgeness)):
		validation_input.append([validation_edgeness[index], validation_cornerness[index]])
	test_input = []
	for index in xrange(len(test_edgeness)):
		test_input.append([test_edgeness[index], test_cornerness[index]])
	training_input = np.asarray(training_input)
	validation_input = np.asarray(validation_input)
	test_input = np.asarray(test_input)

	#One hot encode outputs
	training_answers = np_utils.to_categorical(training_answers)
	validation_answers = np_utils.to_categorical(validation_answers)
	test_answers = np_utils.to_categorical(test_answers)

	#Define the model
	print "Initializing training procedure..."
	model = Sequential()
	model.add(Conv2D(30, (5, 5), input_shape=(2, 28, 28), data_format='channels_first', activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(15, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(10, activation='softmax'))

	#Compile the model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	#Fit the model
	print "Beginning training..."
	model.fit(training_input, training_answers, validation_data=(validation_input, validation_answers), epochs=10, batch_size=200)

	#Score the model on test data
	scores = model.evaluate(test_input, test_answers, verbose=0)
	print "Final classification error: %.2f%%" % (100-scores[1]*100)
	model.save('CNNModel.h5')

if __name__ == '__main__':
	build_model()