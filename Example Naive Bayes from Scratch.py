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
 
def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries
 
summaries = {}
for classValue, instances in separated.items():
	summaries[classValue] = summarize(instances)


# returns probability density for a value x on a normal curve of a given mean and stdev 
def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
 
def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.iteritems():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities
			
def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.iteritems():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel
 
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
 
#################################3333    



dataset = loadCsv(filename)

trainingSet, testSet = splitDataset(dataset, splitRatio)
print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), len(trainingSet), len(testSet))
# prepare model
summaries = summarizeByClass(trainingSet)
# test model
predictions = getPredictions(summaries, testSet)
accuracy = getAccuracy(testSet, predictions)
print('Accuracy: {0}%').format(accuracy)
 

