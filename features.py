from __future__ import division
import pickle
import os
import nltk
import math
import random
from nltk.corpus import stopwords
from nltk.classify import apply_features
import re


features = {}
rx = re.compile('([&#/(),-])')

neg_list = []
def tagcountsfeatures(sent):
	
  newsent = re.sub(' +',' ',rx.sub(' ',sent))
	text = nltk.word_tokenize(newsent)
	feature={}
	tagslist = nltk.pos_tag(text)

	# Bigram features:
	count_positive_pair = 0
	count_negation_pair = 0
	count_negation = 0
	count_exclamation_question_mark= 0
	# feature1: count exclamation and question mark
	for t in tagslist:
		if t[0] in ['!','?']:
			count_exclamation_question_mark +=1

	for (w1,t1), (w2,t2) in nltk.bigrams(tagslist):
		# feature2: count positive word pair
		if (w1.lower(),w2.lower()) in [('no','problems'),
										('no','problem'),
										('no','big'),
										('no','trouble'),
										('no','complaints'),
										('no','issues'),
										('no','other'),
										('no','odor'),
										('not','difficult'),
										('highly','recommended'),
										('I','recommended'),
										('well','worth'),
										('is','worth'),
										('was','worth'),
										('\'s','worth'),
										('centainly','worth'),
										('definitely','worth'),
										('indeed','worth'),
										('it','worth'),										
										]:
			count_positive_pair +=1

		# feature3: count negative word pair
		if (w1.lower(),w2.lower()) in [('no','way'),
										('not','work'),
										('not','compatible'),
										('not','worth'),
										('not','possible'),
										('not','find'),
										('no','manual'),
										('not','allow'),
										('not','buying'),
										('not','easy'),
										('never','get'),
										('stay','away'),
										('not','recommended'),
										]:
			count_negation_pair +=1
		
		# feature4: count negative words followed by ADJ/N/Verb
		if w1.lower() in ['not','never','no','wrong','junk'] and (t2 in ['JJ','JJR','JJS'] or t2 in ['VB','VBD','VBG','VBN','VBP','VBZ'] or t2 in ['NN','NNS']):		
			count_negation +=1

	feature["Positive Pair"]=count_positive_pair	
	feature["Negation Pair"]=count_negation_pair
	feature["Negation Count"]= count_negation		
	feature["Exclamation Question Mark"]=count_exclamation_question_mark
	return feature

if __name__ == '__main__':
	pickelFile = "InputPicklefiles"
	lines = list()
	words = list()
	c = 0
	for p in os.listdir(pickelFile):
	    if p != ".DS_Store":
	    	print "---------------------"
	    	print "Now adding:",p
	    	parsed_reviews = pickle.load(open(pickelFile+"/"+p, "rb" )) 
	    	lines =  lines + parsed_reviews.items()  	
	
	random.shuffle(lines)
	featuresets = [(tagcountsfeatures(sent), orientation) for (sent,orientation) in lines]
	
	split = int(math.floor(len(featuresets)*0.9))
	train_set, test_set = featuresets[split:], featuresets[:split]
	classifier = nltk.NaiveBayesClassifier.train(train_set)
	print nltk.classify.accuracy(classifier, test_set)
