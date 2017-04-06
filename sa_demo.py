"""
The SAR14 dataset contains 233600 IMDb movie reviews along with their associated rating scores on a 1-10 scale. Particularly, 
this dataset consists of 167378 reviews with scores >= 7 out of 10 and 66222 reviews with scores <= 4 out of 10. 
Please find details about the construction of this dataset as well as results of sentiment polarity classification in our paper:

Dai Quoc Nguyen, Dat Quoc Nguyen, Thanh Vu and Son Bao Pham. 2014. Sentiment Classification on Polarity Reviews: An Empirical Study Using Rating-based Features. 
In Proceedings of the 5th Workshop on Computational Approaches to Subjectivity, Sentiment and Social Media Analysis, pp. 128-135.

@InProceedings{nguyenNVP2014,
author    = {Nguyen, Dai Quoc  and  Nguyen, Dat Quoc  and  Vu, Thanh  and  Pham, Son Bao},
  title     = {{Sentiment Classification on Polarity Reviews: An Empirical Study Using Rating-based Features}},
  booktitle = {Proceedings of the 5th Workshop on Computational Approaches to Subjectivity, Sentiment and Social Media Analysis},
  month     = {June},
  year      = {2014},
  address   = {Baltimore, Maryland},
  publisher = {Association for Computational Linguistics},
  pages     = {128--135},
  url       = {http://www.aclweb.org/anthology/W14-2621}
}
	
Main Pipeline for ML Sentiment Analysis:
	1. Preprocess Data
		a. Features
			- Tokenize
			- Remove Stop Words
			- Lemmatize Words
			- Remove Pronunciation

	2. Vectorizer
		a. Tf-idf - Weighted by Vader Analysis

	3. Train - Test Data Split
		a. Reduces overfitting of training data using Cross Validation

	4. Machine Learning Model
		b. Regression - continuous labels
"""

# python natives
import re, sys, time, json, string, pickle
import numpy as np
import pandas as pd

# encoding 
reload(sys)  
sys.setdefaultencoding('utf8')

# nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

# scikit-learn ml libs
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, median_absolute_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

#### GLOBAL VARIABLES / FUNCTIONS ####
analyzer = SentimentIntensityAnalyzer()
stopWords = set(stopwords.words('english'))
wnl = WordNetLemmatizer()
start_time = time.time()


def process_incoming_data(filepath):
	""" uploads data from filepath and returns a dataframe with the data. """
	reviews = []	
	labels = []
	with open(filepath) as fileData:
		i = 0
		for line in fileData:
			if (i+1) % 50000 == 0:
				print 'loading ', i+1,' review...'				
			line = line.encode('utf-8').split('"')
			review = line[1].strip('"')	
			label = float(line[2].strip().strip(',').strip('\n'))
			reviews.append(review)
			labels.append(label)
			i+=1
	return pd.DataFrame({'overall': labels, 'reviewText': reviews})


def review2words(raw_review):
	""" Functions to convert raw review to a string of words """
	no_punc = re.sub("[^a-zA-z!.]", " ", raw_review)
	words = word_tokenize(no_punc.lower())
	no_stops = [wnl.lemmatize(word=w) for w in words if w not in stopWords]
	joined_words = " ".join(no_stops)
	return joined_words


def vader_weight_vectorizer(vectorizer, features):
	""" Take vader score and multiply it by vectorizer weight """
	vocabulary = vectorizer.get_feature_names()
	vader_weights = []
	for word in vocabulary:
		vader_analysis = analyzer.polarity_scores(word)
		weight = vader_analysis['compound']
		vader_weights.append(weight)
	vader_weighted_features = []
	for row in features:
		vader_weighted_features.append(np.multiply(row, np.array(vader_weights)))
	return np.vstack(vader_weighted_features)


def scale_conversion(raw_label):
	""" returns equivalent value according to desired scale """
	pass


def t():
	""" Prints Elapsed Time """
	current = time.time() - start_time
	if current > 60:
		print "--- ~{0} minutes ---".format(current/60), '\n'
	else:
		print "---{0} seconds ---".format(current), '\n'


################
# Main Function
################

if __name__ == '__main__':
	# redirect to logfile	
	logfile = 'sa_demo.log'
	sys.stdout = open(logfile, 'w')

	# Load Amazon Data
	print 'Step One: Loading Amazon data...'
	df = process_incoming_data('review_data/SAR14.txt')
	df, _ = train_test_split(df, test_size=0.99)

	y, x = df.pop('overall'), df.pop('reviewText') # raw features and labels
	t()
	
	# Preprocess Text
	print ('Step Two: Processing the Data for cross validation...')
	num_reviews = x.size
	clean_reviews = []
	i = 0
	for index, row in x.iteritems():
		if (i+1) % 50000 == 0:
			print "Review {0} of {1}...".format(i+1, num_reviews)
		clean_reviews.append(review2words(row))
		i += 1
	t()

	# Convert Labels
	print 'Step Three: Convert Labels according to desired scale...'
	y = y # No Need to convert right now.
	t()
	
	print 'Step Four: Vectorize Features...'
	vec = TfidfVectorizer(analyzer='word', max_features=5000)
	vec_feats = vec.fit_transform(clean_reviews).toarray()
	vec_feats = vader_weight_vectorizer(vec, vec_feats)

	df = pd.DataFrame({
		'labels': y.tolist(),
		'reviews': x.tolist(),
		'features': vec_feats.tolist()
	})
	t()

	# Cross Validation
	print 'Step Five: Cross Validation...'
	trainDF, testDF = train_test_split(df, test_size=0.2)
	t()
	
	# Train ML Model
	print 'Step Six: Train ML Model...'
	ml_model = RandomForestRegressor(n_estimators=150) # n_trees
	ml_model.fit(trainDF['features'].tolist(), trainDF['labels'].tolist())
	t()

	# Evaluate Predictions
	print 'Step Seven: Compare Predictions and Calculate Error Loss...'
	pred = ml_model.predict(testDF['features'].tolist())
	print '*'*10, '\nTrue vs. Predicted Labels:'
	for i in range(1):
		print float(testDF['labels'].tolist()[i]), round(pred[i], 2), testDF['reviews'].tolist()[i]
	
	# Loss Functions
	print '\nCompare Loss Functions:'
	loss = mean_absolute_error(testDF['labels'].tolist(), pred)
	print 'Mean Absolute Error: ', loss
	loss = median_absolute_error(testDF['labels'].tolist(), pred)
	print 'Median Absolute Error: ', loss
	t()

	# Save Models
	print 'Step Seven: Saving Model and Vectorizer...'
	with open('tfidfvectorizer.pk', 'wb') as saveVec:
		pickle.dump(vec, saveVec)
	with open('ml_model.pk', 'wb') as saveMod:
		pickle.dump(ml_model, saveMod)
	t()
	
	# End of Program
	print 'Script Finished'
	t()
