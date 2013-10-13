# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import os
import nltk
import pickle
from __future__ import division
from nltk.corpus import stopwords
import re
import random
import math

# <codecell>

os.chdir("/Users/Morgan/Dropbox/School/NLP/classification_project/NLP_classification_project")

# <codecell>

#Combine all the reviews into one dictionary
parsed_reviews={}
# print os.listdir(os.getcwd())
for File in os.listdir(os.getcwd()):
    if File[-2:]==".p": #just get pickled files
        parsed_reviews= dict(pickle.load(open(File, "rb")).items() + parsed_reviews.items())

# <codecell>

print "# of reviews: "+str(len(parsed_reviews.items()))
print 'here is an example (the first item in the dictionary of "parsed_reviews":'
print parsed_reviews.items()[:5]

# <codecell>

#puts the tags with the words
# for sent in parsed_reviews.keys():
#     tag=parsed_reviews[sent]
#     tagged_words = [(word,tag) for word in nltk.word_tokenize(sent)]
# print tagged_words[0:4]

# <codecell>

def number_of_adjectives(sent):
    #Process the sentence here, e.g. tagging the words of sentence
    text = nltk.word_tokenize(sent)
    tags = nltk.pos_tag(text)
#     print tags
    #counts the adjectives in the sentence
    num_adj = len([tag for (word,tag) in tags if tag =="JJ"])
#     print num_adj
    #Add the feature here, e.g. adding # of adjectives
    features["count adjectives"] = num_adj
    return features

# <codecell>

number_of_adjectives(parsed_reviews.keys()[0])

# <codecell>

featuresets = [(number_of_adjectives(sent), tag) for (sent,tag) in parsed_reviews.items()]
# for sent,tag in parsed_reviews:
    


# <codecell>

train_set, test_set = featuresets[500:], featuresets[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print nltk.classify.accuracy(classifier, test_set)

# <codecell>

print classifier.show_most_informative_features(5)

# <codecell>


