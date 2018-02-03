# -*- coding: utf-8 -*-
"""
Naive bayes from scratch

Adapted from:  https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/

"""

# Example of Naive Bayes implemented from Scratch in Python
import csv
import random
import math

 
filename = 'C:/Users/bsmal725/Documents/My Data/Public Datasets/pima-indians-diabetes.data.csv'

lines = csv.reader(open(filename))
dataset = list(lines)
for i in range(len(dataset)):
    dataset[i] = [float(x) for x in dataset[i]]

splitRatio = 0.67
    
# Start with all data in test set and randomly "pop" rows out of test set and into training set

trainSize = int(len(dataset) * splitRatio)
trainSet = []
testSet = list(dataset) 
while len(trainSet) < trainSize:
	index = random.randrange(len(testSet))
	trainSet.append(testSet.pop(index))
    
print('Split {0} rows into train={1} and test={2} rows'.format(len(dataset), len(trainSet), len(testSet)))

 
# Separate data by class use values in last column

separated = {} # create empty dictionary

for i in range(len(dataset)):
	vector = dataset[i]
	if (vector[-1] not in separated): # if value in last column is not in dictionary, add as key with empty list as value
		separated[vector[-1]] = []
	separated[vector[-1]].append(vector) # add each vector to respective list

 
def mean(numbers):
	return sum(numbers)/float(len(numbers))
 
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)
 
# summarize function uses "zip" to restructure data into a column (tuple) of data for each attribute    

def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1] # no summary is returned for the position representing the class
	return summaries
 
# summary statitics are computed using the full dataset (training and test) 
    
summaries = {}
for classValue, instances in separated.items():
	summaries[classValue] = summarize(instances)

#######################################################################################

# returns probability density for a value x on a normal curve of a given mean and stdev 
def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent


 
def calculateClassProbabilities(summaries, inputVector):
	
    probabilities = {} # probabilities is a dictionary that when returned will have a key for the class
                       # and a corresponding value that represents the probability for that class
    
# outer loop iterates through classes
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        
      # inner loop iterates through attributes, multiplying the class probability
      # by the class probability for that attribute
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities
			
# this function just returns the class label corresponding to the highest probability in
# the "probabilities" dictionary
    
def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.items():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel
 
# this function iterates through the rows of the test set and gets a label for each    

def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions
 
def getAccuracy(testSet, predictions):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i][-1] == predictions[i]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0
 
#########################################################################

# test model
predictions = getPredictions(summaries, testSet)

accuracy = getAccuracy(testSet, predictions)

print('Accuracy: {0}%'.format(accuracy))
 

