from __future__ import division
import pickle
import os
import nltk
import math
from nltk.corpus import stopwords
import re
import random
import collections, itertools
import nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews, stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
import nltk.collocations
import nltk.corpus
import collections
import string


#parsed_reviews = pickle.load( open( "Diaper Champ.p", "rb" ) )
features = {}
rx = re.compile('([&#/(),-])')
all_words = nltk.FreqDist()

#This method uses normalization
def tagcountsfeatures(sent,inputset,all_words):
	"""
	This is te main method that produces the feature set.
	Input: string
		sent -  Raw sentence taken from the training set.

	Output: dictinary
		features - contains adoctionary with key being the feature name and value being the calculated
		feature value
	Output format: correct=<actual orientation> guess=<predicted orientation> sent=<raw sentence>'

	Features treid out:
	1) Count of past tense verbs in the sentence
	2) Count of past tense adverbs in the sentence
	3) Count of capitalised words in the sentence
	4) Count of adjectives in the sentence
	5) tf-idf of the words

	Feature Testing:
	I tested the featues afer normalizing and stemming.
	Stemming reduced the accuracy.

	Results for normalized features:
	Accuracy :0.562403697997
	Sample error check
	correct=neg      guess=neutral  sent=* lens visible in optical viewfinder .
	correct=neg      guess=neutral  sent=* main dial is not backlit .  
	correct=neg      guess=neutral  sent=1 ) the included lens cap is very loose on the camera .
	correct=neg      guess=neutral  sent=2 ) not very ergonomical - you 'll find even for a point-and-shoot lens )
	correct=neg      guess=neutral  sent=and the body / construction in general has quite a bit of plastic , a disappointment after the stainless steel heft of the s330 .
	correct=neg      guess=neutral  sent=b ) the lens cover is surely loose , i already accidently finger-printed the len a few times , and au lens tigt and cause potential damage .
	correct=neg      guess=neutral  sent=canon 's g3 does it consistently .

	Results for Normalized and Stemmed features:
	Accuracy :0.521223692912
	correct=neg      guess=pos      sent=unfortunately , this digital moment-capturing device called the g3 sometimes captures the moment after the one you wanted .
	correct=neg      guess=pos      sent=when you look through the viewfinder ( not the lcd ) the bottom left corner of the picture ( about 15 % ) is blocked by the lens .
	correct=neg      guess=pos      sent=while light , it will not easily go in small handbags or pockets .
	correct=neutral  guess=pos      sent=	Department of Computer Sicence
	correct=neutral  guess=pos      sent=( again this is my first digital camera and maybe that is just how they all are . )

	Results for Bigram Collocation features:
	Accuracy :0.36144


	"""

	# Replacing ([&#/(),-]) with a space and then replacing multiple spaces with single space
	#newsent = sent #re.sub(' +',' ',rx.sub(' ',sent))
	#Tokenize
	#print "Type: all_words",type(all_words)
	#print "Highest Freq word:",all_words.keys()[0]
	#print "Highest Freq :",all_words[all_words.keys()[0]]
	#print "Type: inputset",type(inputset)
	#return
	print "HERE AA-1"
	newsent = sent.translate(None, string.punctuation)
	feature={}
	count_past_verbs = 0 
	count_present_verbs = 0
	count_upper = 0
	count_adj = 0
	#normalizing
	text = normalize(nltk.word_tokenize(sent))
	#stemming
	text = stemming(normalize(nltk.word_tokenize(sent)))
	tagslist = nltk.pos_tag(text)
	#Calculating feature value
	print "Size of tagslist: ", len(tagslist)
	
	for t in tagslist:
		if t[1] in ['VBD','VBN']:
			count_past_verbs+=1
		if t[1] in ['VBG','VBZ']:
			count_present_verbs+=1
		if (t[0].isupper() and len(t[0]) > 3 and t[1] not in ['NN','NNP','NNPS','NNS','PRP','PRP$']):
			count_upper+=1
		if t[1] in ['JJ','JJR','JJS']:
			count_adj+=1
		#Calculatinf tf-idf
		
		ctd = newsent.lower().count(t[0])

		#max frequency of any word in the sentence
		#fq = nltk.FreqDist(w.lower() for w in stemming(normalize(text)))
		#Below I tried to calculate tf-idf without stemming
		
		print "ctd: ", ctd
		#normalizing
		fq = nltk.FreqDist(w.lower() for w in normalize(text))
		#normalizing and stemming
		#fq = nltk.FreqDist(w.lower() for w in stemming(normalize(text)))
		maxfreq = fq[fq.keys()[0]]
		print "maxfreq: ", maxfreq

		tf = 0.5 + (0.5*ctd)/maxfreq
		#normalizing
		num_sent_with_word = sum([1 for record in inputset if t[0].lower() in normalize(record[0].lower().split())])
		#normalizing and stemming
		#num_sent_with_word = sum([1 for record in inputset if t[0].lower() in stemming(normalize(record[0].lower().split()))])
		print "num_sent_with_word: ", num_sent_with_word

		#idf = log of total number of sentence / total sentences in which the word occurs
		idf = math.log((len(inputset)/(1+num_sent_with_word)),2)

		#feature['tf-idf delta (%s)' % w] = ctd*math.log((1.0+float(nt)/(1.0+float(pt))),2)
		feature['tf-idf (%s)' % t[0].lower()] = tf*idf	
		
	#Assigning feature value
	feature["Past Tense Verb"]=count_past_verbs
	feature["Present Tense Verb"]=count_present_verbs
	feature["Adjectives"]=count_adj
	feature["Capitalised Words"]=count_upper
	
	#Logic to calculate tf-idf, it is not imporving accuracy though	
	#for i,w in enumerate(all_words.keys()[:20]):
	return feature
	

#Normalizing words
def normalize(tokens):
	#print "Reached here normalize"
	#rx = re.compile('([&#/(),-])')
	return [t.translate(None, string.punctuation).lower() for t in tokens if t.lower() not in stopwords.words('english') and len(t)>=3]
	
#Stemming , but not used currently
def stemming(tokens):
	#print "Reached here stemming"
	lancaster = nltk.LancasterStemmer()
	lancasterlist =  [lancaster.stem(t) for t in tokens]
	return lancasterlist

def bigram_word_feats(words, score_fn=BigramAssocMeasures.likelihood_ratio, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn,n)
    return dict([(ngram,True) for ngram in itertools.chain(words,bigrams)])

def inputset(inputdata):
	words = [w for l in inputdata for w in l[0].split()]
	#Normalizing and stemming the words 
	#finallist  = stemming(normalize(words))
	#Below I tried to calculate finallist without stemming
	finallist  = normalize(words)

	all_words = nltk.FreqDist(w.lower() for w in finallist)
	highestfreq = all_words[all_words.keys()[0]]
	print "Highest Fequency word",all_words.keys()[0]
	print "Higest Frequency",highestfreq
	print "Length of final list (words)",len(finallist)

	d = 	{"finallist":finallist,
			"all_words":all_words,
			"highestfreqword":highestfreq
			}
	return d


if __name__ == '__main__':
	pickelFile = "training"
	heldoutFiles = "heldout"
	lines = list()
	heldoutlines = list()
	words = list()
	c = 0
	for p in os.listdir(pickelFile):
	    if p != ".DS_Store":
	    	print "Now adding:",p
	    	parsed_reviews = pickle.load(open(pickelFile+"/"+p, "rb" ))
	    	# Reading data of all files in a single list
	    	print len(parsed_reviews.items())
	    	lines = lines + parsed_reviews.items()
	random.shuffle(lines)

	#Heldout data
	print "Adding test files"
	for p in os.listdir(heldoutFiles):
	    if p != ".DS_Store":
	    	print "Now adding:",p
	    	parsed_reviews_1 = pickle.load(open(heldoutFiles+"/"+p, "rb" ))
	    	# Reading data of all files in a single list
	    	print len(parsed_reviews_1.items())
	    	heldoutlines = heldoutlines + parsed_reviews_1.items()
	random.shuffle(heldoutlines)

	train = lines
	test = heldoutlines


	print "Train set length: ",len(train)
	print "Test set length: ",len(test)
	#Fetching all the words from the combined list
	output_train = inputset(train)
	output_test = inputset(test)

	train_set= [(tagcountsfeatures(sent,train,output_train['all_words']), orientation) for (sent,orientation) in train]
	test_set= [(tagcountsfeatures(sent,test,output_test['all_words']), orientation) for (sent,orientation) in test]

	classifier = nltk.NaiveBayesClassifier.train(train_set)
	print "tags classification"
	print nltk.classify.accuracy(classifier, test_set)

	
	errors = []
	for (sent,orientation) in test:
	    guess = classifier.classify(tagcountsfeatures(sent,test,output_test['all_words']))
	    if guess != orientation:
	        errors.append( (orientation, guess, sent) )
	    
	for (orientation, guess, sent) in sorted(errors): 
	    print 'correct=%-8s guess=%-8s sent=%-30s' % (orientation, guess, sent)
	
	#Testing with Bigram collocation
	featuresets_2 = [(bigram_word_feats(nltk.word_tokenize(sent)), orientation) for (sent,orientation) in train]
	split = int(math.floor(len(featuresets_2)*0.7))
	train_set_2, test_set_2 = featuresets_2[split:], featuresets_2[:split]
	classifier1 = nltk.NaiveBayesClassifier.train(train_set_2)
	print "Bigram tagger"
	print nltk.classify.accuracy(classifier1, test_set_2)
	
