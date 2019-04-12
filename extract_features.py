import numpy as np
import caffe
import PIL
from PIL import Image



def read_image(fileName):
    '''
    read an image for preprocessing
    '''
    tmpImg = Image.open(fileName)
    tmpImg = tmpImg.resize((224, 224), PIL.Image.ANTIALIAS)
    tmpImg = np.array(tmpImg).astype('float32')

    return tmpImg

def preprocess_content(tmpImg, meanImgNet):
    '''
    preprocess the input image for the content network
    '''
    
    tmpImgA = tmpImg - np.array(meanImgNet)
    tmpImgA = tmpImgA[:, :, ::-1].astype('float32') 
    tmpImgA = np.transpose(tmpImgA, (2, 0, 1))

    return tmpImgA  

def preprocess_aesthetics(tmpImg, mean_value40K, mean10K):
    '''
    preprocess the input image to feed into the network
    imagesAes is for the aesthetics
    imagesAtr is for the attributes
    '''
    
    tmpImgA = tmpImg - np.array(mean_value40K)
    tmpImgA = tmpImgA[:, :, ::-1].astype('float32')   
    imagesAes = np.transpose(tmpImgA, (2, 0, 1)) 

    tmpImgB = tmpImg - np.array(mean10K)
    tmpImgB = tmpImgB[:, :, ::-1].astype('float32')             
    imagesAtr = np.transpose(tmpImgB, (2, 0, 1))

    return imagesAes, imagesAtr

def features(album, model_def_a, model_weights_a, model_def_content, model_weights_content):
    

    mean10K = [111.452247751, 108.607499157, 100.449740826]#mean for attributes
    mean_value40K = [109.692900367, 104.676436555, 97.6725598888]#mean for aesthetics
    meanImgNet = [123, 117, 104] #mean for content

    attributesName = ['BalacingElements', 'ColorHarmony', 'Content', 
    'DoF', 'Light', 'Object', 'Repetition', 'RuleOfThirds', 'Symmetry', 'VividColor']#ten attributes

    attributeNet = caffe.Net(model_def_a, model_weights_a, caffe.TEST)
    
    contentNet = caffe.Net(model_def_content, model_weights_content, caffe.TEST)


    linesAll = album[:]
    imgNum = len(linesAll)

    
    numFeatures = 176
    attributesFeature = np.zeros((imgNum, len(attributesName)))
    predictScoreAll = []

    contentFeature_length = 10
    contentFeature = np.zeros((imgNum, contentFeature_length))

    for idx, fileName in enumerate(linesAll):
                
        img = read_image(fileName)

        imagesAes, imagesAtr = preprocess_aesthetics(img, mean_value40K, mean10K)

        attributeNet.blobs['data'].data[0] = np.array(imagesAes).astype(np.float32)
        attributeNet.blobs['data_p'].data[0] = np.array(imagesAtr).astype(np.float32)

        output = attributeNet.forward()

        predictScoreAll.append(output['loss3_new/classifier'][0][0])

        for i, name in enumerate(attributesName):
            attributesFeature[idx,i] = output['loss3_new/classifier_' + name][0][0] 


        contentNet.blobs['data_p'].data[...] = preprocess_content(img, meanImgNet)
        contentNet.forward()

        contentFeature[idx, :] = contentNet.blobs['prob'].data[0].reshape(contentFeature_length)


    return attributesFeature, predictScoreAll, contentFeature


