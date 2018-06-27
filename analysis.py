import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import re
import string 
from word2vec import Word2Vec
from logisticRegression import LogisticRegression

#Look at particular business entries 
def parseBusiness(business_filepath):
	print("this is filepath: " + str(business_filepath))
	business_map = {}
	with open(business_filepath) as data_file:
		for json_obj in data_file:
			business = json.loads(json_obj)
			categories = business['categories']
			business_id = business['business_id']
			for category in categories:
				if category in business_map:
					business_map[category].append(business_id)
				else:
					business_map[category] = [business_id]
		return business_map

#Extract raw json for reviews and "useful" labels
def getReviews(reviews_filepath, num_reviews):
	print("this is filepath: " + str(reviews_filepath))
	i = 0
	reviews = []
	labels = []
	with open(reviews_filepath) as data_file:
		for json_obj in data_file:
			reviews.append(json.loads(json_obj)["text"])
			labels.append(json.loads(json_obj)["useful"])
			i = i + 1
			if(i > num_reviews):
				break 
	labels = [min(x, 1) for x in labels]
	return reviews, labels

#Clean a given string 
def cleanString(text):
	text = re.sub("\n", "", text)
	text = re.sub(r"[0-9]+", "", text)
	text = text.lower() 
	text = text.translate(str.maketrans('', '', string.punctuation))
	return text

#Clean reviews, convert to list of tokens
def cleanReviews(reviews):
	reviews = [cleanString(review) for review in reviews]
	reviews = " ".join(reviews).split(" ")
	return reviews

#Transform reviews into corresponding averaged word vectors based on embeddings
def transformFeatures(reviews, word_index_map, embeddings):
	m, n = embeddings.shape
	count = 0
	avg_vectors = []
	for review in reviews:
		cleaned_review = cleanString(review).split(" ")
		avg_vector = [0 for i in range(n)]
		for word in cleaned_review:
			if(word in word_index_map):
				index = word_index_map[word]
			else:
				index = 0
			word_vector = embeddings[index]
			avg_vector = avg_vector + word_vector
			count += 1
		avg_vector /= count
		avg_vectors.append(avg_vector)
	return avg_vectors

#Classification set up - currently prediction of useful or not (0/1, if at least one "useful" upvote) based on averged word vector per review
def classify(num_reviews):
	
	#Load data and get tokens
	curr_dir = os.path.dirname(os.path.abspath(__file__))
	reviews_filepath = os.path.join(curr_dir,'review.json')
	reviews, labels = getReviews(reviews_filepath, num_reviews)
	tokens = cleanReviews(reviews)

	#Training/testing
	split_num = int(0.8 * num_reviews)
	train_reviews = reviews[:split_num]
	train_text = cleanReviews(train_reviews)
	test_reviews = reviews[split_num:]
	test_text = cleanReviews(test_reviews)

	#Set up Word2Vec and get relevant variables
	w2v = Word2Vec()
	w2v.run(tokens, iterations = 2000)
	final_embeddings = w2v.final_embeddings
	word_index_map = w2v.word_index_map

	#Train logistic regression and then test
	X_train = transformFeatures(train_reviews, word_index_map, final_embeddings)
	Y_train = labels[:split_num]
	Y_test = labels[split_num:]
	X_test = transformFeatures(test_reviews, word_index_map, final_embeddings)
	lr = LogisticRegression(X = X_train, Y = Y_train, steps = 500)
	lr.gradientDescent()
	Y_pred = lr.predict(X_test)

	#Look at results
	print("Correctly predicted count: ")
	total_matches = (sum(i == j for i, j in zip(Y_pred, Y_test)))
	print(total_matches)

	print("Baseline count: ")
	baseline = max(len(Y_test) - sum(Y_test), sum(Y_test))
	print(baseline)


if __name__ == "__main__":
	classify(10000)


