# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <rawcell>

# My Features to Implement:
# 
# # of seed positive/negative adjectives - Vanessa
# # of seed positive/negative verbs - Vanessa
# 
# presence of positive/negative adjectives - Vanessa
# presnes of positive/negative verbs - Vanessa

# <codecell>

from nltk.corpus import movie_reviews
import nltk
import random
import string
from nltk.corpus import stopwords
import pickle
import os

files = os.listdir("../Vanessa/")
all_files = [f for f in files if f[-2:] == '.p']
all_reviews = []
random.shuffle(all_reviews)

for f in all_files:
    parsed_reviews = pickle.load( open( f, "rb" ) )
    for i in parsed_reviews.items():
        all_reviews.append(i)

random.shuffle(all_reviews)
training_reviews = all_reviews[:3215]
test_reviews = all_reviews[3215:]
#print len(test_reviews)

# <codecell>

def adj_vrb(reviews):
    neutral_adj = []
    pos_adj = []
    neg_adj = []
    neutral_vrb = []
    pos_vrb = []
    neg_vrb = []
    #stopwords = nltk.corpus.stopwords.words('english')
    
    for sent,label in reviews:
        clean_sent = sent.translate(string.maketrans("",""), string.punctuation).lower().strip().split()
        #clean_sent_no_sw = [w for w in clean_sent if w not in stopwords]     ##Removing stop words before tagging does not work.
        tagged = nltk.pos_tag(clean_sent)
        #print sent,label
        if label == 'neutral':
            for i,j in tagged:    #this is looping over all
                if j[0] == 'J':
                    neutral_adj.append(i)
                if j[0] == 'V':
                    neutral_vrb.append(i)
        if label == 'pos':
            for i,j in tagged:    #this is looping over all
                if j[0] == 'J':
                    pos_adj.append(i)
                if j[0] == 'V':
                    pos_vrb.append(i)
        if label == 'neg':
            for i,j in tagged:    #this is looping over all
                if j[0] == 'J':
                    neg_adj.append(i)
                if j[0] == 'V':
                    neg_vrb.append(i)                

    return [neutral_adj, neutral_vrb, pos_adj, pos_vrb, neg_adj, neg_vrb]
    
seed_words = adj_vrb(training_reviews)

# <codecell>

top_adj = nltk.FreqDist(seed_words[0]).keys()[:100]
top_vrb = nltk.FreqDist(seed_words[1]).keys()[:100]

# <codecell>

neutral_top_adj = nltk.FreqDist(seed_words[0]).keys()
pos_top_adj = nltk.FreqDist(seed_words[2]).keys()
neg_top_adj = nltk.FreqDist(seed_words[4]).keys()

neutral_top_vrb = nltk.FreqDist(seed_words[1]).keys()
pos_top_vrb = nltk.FreqDist(seed_words[3]).keys()
neg_top_vrb = nltk.FreqDist(seed_words[5]).keys()

only_pos_adj = [w for w in pos_top_adj if w not in neutral_top_adj and w not in neg_top_adj]
only_neg_adj = [w for w in neg_top_adj if w not in neutral_top_adj and w not in pos_top_adj]
only_neu_adj = [w for w in neutral_top_adj if w not in neg_top_adj and w not in pos_top_adj]
only_pos_vrb = [w for w in pos_top_vrb if w not in neutral_top_vrb and w not in neg_top_vrb]
only_neg_vrb = [w for w in neg_top_vrb if w not in neutral_top_vrb and w not in pos_top_vrb]
only_neu_vrb = [w for w in neutral_top_vrb if w not in neg_top_vrb and w not in pos_top_vrb]

# <rawcell>

# #Next steps:
# Done --  changed to only use unique words. Could compare using all of the words vs. most frequent    (all words better than most frequent)
# # compare presence vs. counts
# Done -- did not improve things. # could use just JJR or JJS, comparative or superlative
# # could do stemming on these words...

# <codecell>

#Testing on adjectives

def features(sent):
    sent = sent.translate(string.maketrans("",""), string.punctuation).lower().strip().split()
    features = {}
    for word in pos_top_adj:
        features["has %s" % word] = word in sent
    for word in neg_top_adj:
        features["has %s" % word] = word in sent
    for word in neutral_top_adj:
        features["has %s" % word] = word in sent    
    for word in neutral_c_adj:
        features["has %s" % word] = word in sent    
    #for word in only_pos_adj:
        #features["has %s" % word] = word in sent
    #for word in only_neg_adj:
        #features["has %s" % word] = word in sent
    #for word in only_neu_adj:
        #features["has %s" % word] = word in sent
    return features

train_set = [(features(s), r) for (s,r) in training_reviews]
devtest_set = [(features(s), r) for (s,r) in test_reviews]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print nltk.classify.accuracy(classifier, devtest_set)

#classifier.show_most_informative_features(5)

# <rawcell>

# *Presence of a pos/neg/neu adjective of any kind using (only_ sets):*
# only positive adjectives: .5210
# only negative adjectives: .4873
# only neutral adjectives: .5014
# both positive and negative adjectives: .5098
# both positive, negative, and neutral adjecives: .5182
# **positive and neutral: .5266**
# 
# *Presence of a pos/neg/neu adj of any kind using top_adj sets:*
# only pos: .5574
# only neg: .5686
# **only neu: .5938**
# both pos and neg: .5574
# all three: .57703
# both neg and neu: .5882
# both pos and neu: .5798
# 
# #Just using subset of pos/neg/neu adj:
# #First 100: top neu: .5714, only neu: .4986

# <codecell>

# Error analysis

errors = []
for (sent, review) in test_reviews:
    guess = classifier.classify(features(sent))
    if guess != review:
        errors.append( (review, guess, sent) )

for (review, guess, sent) in sorted(errors): 
    print 'correct=%-8s guess=%-8s sent=%-30s' % (review, guess, sent)

# <codecell>

#Testing on verbs

def features(sent):
    sent = sent.translate(string.maketrans("",""), string.punctuation).lower().strip().split()
    features = {}
    for word in pos_top_vrb:
        features["has %s" % word] = word in sent
    for word in neg_top_vrb:
        features["has %s" % word] = word in sent
    for word in neutral_top_vrb:
        features["has %s" % word] = word in sent
    for word in pos_top_adj:
        features["has %s" % word] = word in sent
    for word in neutral_top_adj:
        features["has %s" % word] = word in sent
    for word in neg_top_adj:
        features["has %s" % word] = word in sent
    #for word in only_pos_vrb:
        #features["has %s" % word] = word in sent
    #for word in only_neg_vrb:
        #features["has %s" % word] = word in sent
    #for word in only_neu_vrb:
        #features["has %s" % word] = word in sent
    return features

train_set = [(features(s), r) for (s,r) in training_reviews]
devtest_set = [(features(s), r) for (s,r) in test_reviews]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print nltk.classify.accuracy(classifier, devtest_set)

# <rawcell>

# *Presence of a pos/neg/neu adjective of any kind using (only_ sets):*
# only positive verbs: .5322
# only negative verbs: .5238
# only neutral verbs: .5098
# **both positive and negative verbs: .5518**
# both positive and neutral: .5350
# both negative and neutral: .5210
# all positive, negative, and neutral verbs: .5462
# 
# *Presence of a pos/neg/neu vrb of any kind using top_vrb sets:*
# only pos: .5434
# only neg: .5882
# only neu: .5574
# both pos and neg: .5574
# **both pos and neu: .5938**
# both neg and neu: .5770
# all: .5826
# 
# using both top features??
# both pos and neu verbs and neu adj: .605
# both pos and neu verbs and neg adj: .5966
# both pos and neu verbs and pos adj: .6106
# both pos and neu verbs and pos and neu adj: .61344
# all verbs and all adj: .6106
# **both pos and neu verbs and all adj: .619**

# <codecell>

#Testing based on count of verbs

def features(sent):
    sent = sent.translate(string.maketrans("",""), string.punctuation).lower().strip().split()
    features = {}
    features["pos adj count"] = 0
    features["neg adj count"] = 0
    features["neu adj count"] = 0
    features["pos vrb count"] = 0
    features["neg vrb count"] = 0
    features["neu vrb count"] = 0    
    for word in pos_top_adj:
        features["pos adj count"] += sent.count(word)
    for word in neg_top_adj:
        features["neg adj count"] += sent.count(word)        
    for word in neutral_top_adj:
        features["neu adj count"] += sent.count(word)
    for word in pos_top_vrb:
        features["pos vrb count"] += sent.count(word)
    #for word in neg_top_vrb:
        #features["neg vrb count"] += sent.count(word)        
    for word in neutral_top_vrb:
        features["neu vrb count"] += sent.count(word)
        
    
    return features


train_set = [(features(s), r) for (s,r) in training_reviews]
devtest_set = [(features(s), r) for (s,r) in test_reviews]
#classifier = nltk.NaiveBayesClassifier.train(train_set)
#print nltk.classify.accuracy(classifier, devtest_set)

print train_set[0:10]

#classifier.show_most_informative_features(5)

# <codecell>

a = ['a','a']
a.count('a')

