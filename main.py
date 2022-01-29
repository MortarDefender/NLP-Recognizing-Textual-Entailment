import json

from src.classifier import Classifier


def run(trainDataSet, testDataSet, outputFileName = None, verbose = False):

    if verbose:
        print(f"start training on {trainDataSet}")
    
    classifier = Classifier(trainDataSet, testDataSet, outputFileName)
    
    if verbose:
        print("finished training")
    
    predictionResults = classifier.predict(testDataSet)
    print(predictionResults)

    #TODO convert testDataSet to fatures and labels
    testFeatures, testLabels = testDataSet
    classifier.accuracy(testFeatures, testLabels, verbose = True)


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)
    
    run()
