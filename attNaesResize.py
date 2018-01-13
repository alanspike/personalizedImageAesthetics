import numpy as np
import caffe
import PIL
from PIL import Image

caffe.set_mode_gpu()
caffe.set_device(0)

def score(imgPath, model_def_a, model_weights_a):
	
	# attributes & aesthetics 
	mean10K = [111.452247751, 108.607499157, 100.449740826] # mean for attributes
	mean_value40K = [109.692900367, 104.676436555, 97.6725598888] # mean for aesthetics

	attributeNet = caffe.Net(model_def_a, model_weights_a, caffe.TEST)

	# 10 attributes
	attributesName = ['BalacingElements', 'ColorHarmony', 'Content', 
	'DoF', 'Light', 'Object', 'Repetition', 'RuleOfThirds', 'Symmetry', 'VividColor']

	# ------------------for aesthetic----------------------------------------------------
	tmpImg = Image.open(imgPath).resize((224, 224), PIL.Image.ANTIALIAS)
	tmpImg = np.array(tmpImg).astype('float32')
	tmpImgA = tmpImg - np.array(mean_value40K)
	tmpImgA = tmpImgA[:, :, ::-1].astype('float32') 
	images = np.transpose(tmpImgA, (2, 0, 1)) 

	attributeNet.blobs['data'].data[0] = np.array(images).astype(np.float32)
	# ---------------------for attributes--------------------------------------------
	img = tmpImg - np.array(mean10K)
	img = img[:, :, ::-1].astype('float32')           
	images = np.transpose(img, (2, 0, 1))

	attributeNet.blobs['data_p'].data[0] = np.array(images).astype(np.float32)

	# ---------------------run--------------------------------------------
	output = attributeNet.forward()

	scoreAll = dict()
	scoreAll['Aesthetics'] = output['loss3_new/classifier'][0][0]

	for name in attributesName:
		scoreAll[name] = output['loss3_new/classifier_' + name][0][0]
	return scoreAll 

if __name__ == '__main__':
	# image path
	imgPath = 'test.jpg'
	# prototxt
	model_def = 'attNaesS1.prototxt'
	# model
	model_weights = 'attNaesSmall.caffemodel'
	
	# get the aesthetics score and 10 attributes, return a dictionary
	scoreAll = score(imgPath, model_def, model_weights)
	print '---------------score----------------'
	for key in scoreAll:
		print key, scoreAll[key]






