import pickle
parsed_reviews = pickle.load( open( "Diaper Champ.p", "rb" ) )
features = {}

def tagcountsfeatures(sent):
	#Process the sentence here, e.g. tagging the words of sentence
	tags = nltk.pos_tag(sent)

	#Add the feature here, e.g. adding # of adjectives
	features["count adjectives"] = result


featuresets = [(document_features(sent), orientation) for (sent,orientation) in parsed_reviews.items()]
