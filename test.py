from extract_features import features
import caffe
import argparse


caffe.set_mode_gpu()
caffe.set_device(0)

def parse_args():
    """
    parse input argument
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--filename',
                        help='the file saves testing images',
                        default='testImages.txt',
                        type=str)
    
    parser.add_argument('--model_def_a',
                        help='the caffe prototxt for aesthetics and attributes model',
                        default='attNaesS1.prototxt',
                        type=str)

    parser.add_argument('--model_weights_a',
                        help='the weights for aesthetics and attributes model',
                        default='attNaesSmall.caffemodel',
                        type=str)

    parser.add_argument('--model_def_content',
                        help='the caffe prototxt for content model',
                        default='content.prototxt',
                        type=str)

    parser.add_argument('--model_weights_content',
                        help='the weights for content model',
                        default='content.caffemodel',
                        type=str)
    

    args = parser.parse_args()
    return args

def main(filename, model_def_a, model_weights_a, model_def_content, model_weights_content):


    allLines = open(filename, 'r').readlines()
    album = []

    for item in allLines:
        album.append(item.rstrip())

    attributesFeature, predictScoreAll, contentFeature = features(album, model_def_a, model_weights_a, model_def_content, model_weights_content)

    

    return attributesFeature, predictScoreAll, contentFeature

if __name__ == '__main__':

    args = parse_args()

    attributesFeature, predictScoreAll, contentFeature = main(args.filename, args.model_def_a, args.model_weights_a, 
        args.model_def_content, args.model_weights_content)

    print 'aesthetics attributes: ', attributesFeature
    print 'aestheitcs score: ', predictScoreAll
    print 'content feature: ', contentFeature



