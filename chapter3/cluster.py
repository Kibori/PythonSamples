import os
import sys
import scipy as sp



from sklearn.feature_extraction.text import CountVectorizer
from scipy import linalg

DIR = r"DIR"

vectorizer = CountVectorizer(min_df=1)

print(vectorizer)

content = ["How to format my hard disk",
	" Hard disk format problems "]
X = vectorizer.fit_transform(content)
print(vectorizer.get_feature_names())
print(X.toarray().transpose())

posts = [open(os.path.join(DIR,f)).read() for f in os.listdir(DIR)]
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=1)

#section3.2.4###
##vectorizer = CountVectorizer(min_df=1,stop_words="english")
##sorted(vectorizer.get_stop_words())[0:20]
# ###

X_train = vectorizer.fit_transform(posts)
num_samples, num_features = X_train.shape
print("#samples:%d, #features: %d" %(num_samples, num_features))

print(vectorizer.get_feature_names())

new_post = "imaging databases"
new_post_vec = vectorizer.transform([new_post])

print(new_post_vec)

print(new_post_vec.toarray())
print(X_train.getrow(0).toarray())
print(X_train.getrow(1).toarray())
print(X_train.getrow(2).toarray())
print(X_train.getrow(3).toarray())
print(X_train.getrow(4).toarray())
print()

def dist_raw(v1,v2):
	delta = v1-v2
	return sp.linalg.norm(delta.toarray())

def dist_norm(v1,v2):
	v1_normalized = v1/sp.linalg.norm(v1.toarray())
	v2_normalized = v2/sp.linalg.norm(v2.toarray())
	delta = v1_normalized - v2_normalized
	return sp.linalg.norm(delta.toarray())
	
import sys
best_doc = None
best_dist = sys.maxint
best_i = None

for i in range(0, num_samples):
	post = posts[i]
	if post == new_post:
		continue
	post_vec = X_train.getrow(i)
	d = dist_norm(post_vec, new_post_vec)
	print "=== Post %i with dist=%.2f: %s"%(i,d,post)
	if d<best_dist:
		best_dist = d
		best_i = i
print("Best post is %i with dist =%.2f"%(best_i,best_dist))

import nltk.stem
s=nltk.stem.SnowballStemmer('english')
print(s.stem("graphics"))
print(s.stem("imaging"))
print(s.stem("image"))
print(s.stem("imagination"))
print(s.stem("imagine"))

print(s.stem("buys"))
print(s.stem("buying"))
print(s.stem("bought"))






