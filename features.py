from __future__ import division
import pickle
import os
import nltk
import math
from nltk.corpus import stopwords
import re


#parsed_reviews = pickle.load( open( "Diaper Champ.p", "rb" ) )
features = {}
rx = re.compile('([&#/(),-])')

def tagcountsfeatures(sent):
	#Process the sentence here, e.g. tagging the words of sentenc
	
	newsent = re.sub(' +',' ',rx.sub(' ',sent))
	text = nltk.word_tokenize(newsent)
	feature={}
	count_past_verbs = 0 
	count_present_verbs = 0
	count_upper = 0
	count_adj = 0
	tagslist = nltk.pos_tag(text)
	for t in tagslist:
		if t[1] in ['VBD','VBN']:
			count_past_verbs+=1
		if t[1] in ['VBG','VBZ']:
			count_present_verbs+=1
		if (t[0].isupper() and len(t[0]) > 3 and t[1] not in ['NN','NNP','NNPS','NNS','PRP','PRP$']):
			count_upper+=1
		if t[1] in ['JJ','JJR','JJS']:
			count_adj+=1

	feature["Past Tense Verb"]=count_past_verbs
	feature["Present Tense Verb"]=count_present_verbs
	feature["Adjectives"]=count_adj
	feature["Capitalised Words"]=count_upper

	all_words = nltk.FreqDist(w.lower() for w in finallist)
	highestfreq = all_words[all_words.keys()[0]]
	print "Highest Fequqency word",all_words.keys()[0]
	print "Higest Frequency",highestfreq
	print "Length of final list",len(finallist)
	for w in set(finallist):
		print w
		print "Sentence:",newsent
		print "count in the sentence",newsent.lower().count(w)
		ctd = newsent.lower().count(w)
		tf = 0.5 + (0.5*ctd)/highestfreq
		pt = sum([1 for wd in posrev if w.strip() in wd.lower().split()])
		#print "pt:",pt
		p = len(posrev)
		nt = sum([1 for wd in negrev if w.strip() in wd.lower().split()])
		#print "nt:",nt
		n =len(posrev)
		neut = sum([1 for wd in neutralrev if w.strip() in wd.lower().split()])
		#print "neut:",neut
		neu =len(neutralrev)
		idf = math.log((len(lines)/(pt+nt+neut)),2)
		print "tf:",tf*idf


		#feature['tf-idf (%s)' % w] = (ctd*math.log((float(nt)/float(pt)),2))
		feature['tf-idf (%s)' % w] = tf*idf
	return feature

    
    #new = set(normalize(text))
    #all_words = nltk.FreqDist(w.lower() for w in finallist)
    #word_features = all_words.keys()[:2000]
    #for word in word_features:
    	#feature['contains(%s)' % word] = (word in new)
    
def frequency(lines):
	cfd = nltk.ConditionalFreqDist((sentiment, word)
		for sentiment in ['pos','neg','neutral']
        for word in lines(lines[1]==sentiment))

def normalize(tokens):
	#print "Reached here normalize"
	rx = re.compile('([&#/(),-])')
	return [rx.sub(' ', t).lower() for t in tokens if t.lower() not in stopwords.words('english') and len(t)>=3 and re.search('^[A-Za-z]+$',t)]
	

def stemming(tokens):
	#print "Reached here stemming"
	lancaster = nltk.LancasterStemmer()
	lancasterlist =  [lancaster.stem(t) for t in tokens]
	return lancasterlist

def display(fl):
	#print set(finallist)
	print len(set(fl))
	all_words = nltk.FreqDist(w.lower() for w in fl)
	word_features = all_words.keys()[:2000]
	print word_features


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
	words = [re.sub(' +',' ',rx.sub(' ',w)) for l in lines for w in re.sub(' +',' ',rx.sub(' ',l[0])).split()]
	cntword = sum([1 for  l in lines for  w in l[0].split()])
	finallist  = normalize(words)

	posrev =  [re.sub(' +',' ',rx.sub(' ',w[0])) for w in lines if w[1]=="pos" ]
	negrev =  [re.sub(' +',' ',rx.sub(' ',w[0])) for w in lines if w[1]=="neg" ]
	neutralrev =  [re.sub(' +',' ',rx.sub(' ',w[0])) for w in lines if w[1]=="neutral" ]


	featuresets = [(tagcountsfeatures(sent), orientation) for (sent,orientation) in lines]
	split = int(math.floor(len(featuresets)*0.9))
	train_set, test_set = featuresets[split:], featuresets[:split]
	classifier = nltk.NaiveBayesClassifier.train(train_set)
	print nltk.classify.accuracy(classifier, test_set)

	    #size = len(featuresets)
	    