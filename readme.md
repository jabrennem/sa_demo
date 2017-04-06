This is an example sentiment analysis program using the python machine learning library scikit-learn.

The main pipeline for sa_demo.py:
  1. Preprocess Data
		a. Feature
			- Tokenization
			- Remove Stop Words
			- Lemmatize Words
			- Remove Pronunciation

	2. Vectorizer
		a. Tf-idf - Weighted by Vader Analysis

	3. Train - Test Data Split
		a. Reduces overfitting of training data using Cross Validation

	4. Machine Learning Model
		a. Random Forest Regressor - predicts continuous labels

*** The Dataset that was used to prepare this program. ***
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
	
