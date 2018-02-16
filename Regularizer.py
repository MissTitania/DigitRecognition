import pickle
import os.path
import Preprocessor

def regularize_data():
	#Check if the unregularized data exists.  If not, construct it
	if not os.path.exists('TrainingSet.pkl'):
		Preprocessor.process_mnist()

	maxEdgeness = 0
	maxCornerness = 0

	edgeness, cornerness, answers = pickle.load(open('TrainingSet.pkl', 'r'))
	for i in xrange(len(edgeness)):
		for x in xrange(28):
			for y in xrange(28):
				if maxEdgeness < edgeness[i][x][y]:
					maxEdgeness = edgeness[i][x][y]
				if maxCornerness < cornerness[i][x][y]:
					maxCornerness = cornerness[i][x][y]
	edgeness, cornerness, answers = pickle.load(open('ValidationSet.pkl', 'r'))
	for i in xrange(len(edgeness)):
		for x in xrange(28):
			for y in xrange(28):
				if maxEdgeness < edgeness[i][x][y]:
					maxEdgeness = edgeness[i][x][y]
				if maxCornerness < cornerness[i][x][y]:
					maxCornerness = cornerness[i][x][y]
	edgeness, cornerness, answers = pickle.load(open('TestSet.pkl', 'r'))
	for i in xrange(len(edgeness)):
		for x in xrange(28):
			for y in xrange(28):
				if maxEdgeness < edgeness[i][x][y]:
					maxEdgeness = edgeness[i][x][y]
				if maxCornerness < cornerness[i][x][y]:
					maxCornerness = cornerness[i][x][y]

	edgeness, cornerness, answers = pickle.load(open('TrainingSet.pkl', 'r'))
	for i in xrange(len(edgeness)):
		for x in xrange(28):
			for y in xrange(28):
				edgeness[i][x][y] /= maxEdgeness
				cornerness[i][x][y] /= maxCornerness
	pickle.dump([edgeness, cornerness, answers], open('TrainingSetRegularized.pkl', mode='w'))
	edgeness, cornerness, answers = pickle.load(open('ValidationSet.pkl', 'r'))
	for i in xrange(len(edgeness)):
		for x in xrange(28):
			for y in xrange(28):
				edgeness[i][x][y] /= maxEdgeness
				cornerness[i][x][y] /= maxCornerness
	pickle.dump([edgeness, cornerness, answers], open('ValidationSetRegularized.pkl', mode='w'))
	edgeness, cornerness, answers = pickle.load(open('TestSet.pkl', 'r'))
	for i in xrange(len(edgeness)):
		for x in xrange(28):
			for y in xrange(28):
				edgeness[i][x][y] /= maxEdgeness
				cornerness[i][x][y] /= maxCornerness

	pickle.dump([edgeness, cornerness, answers], open('TestSetRegularized.pkl', mode='w'))

if __name__ == '__main__':
	regularize_data()