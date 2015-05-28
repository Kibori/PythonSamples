import os
import sys
import scipy as sp
import nltk.stem



from sklearn.feature_extraction.text import CountVectorizer
from scipy import linalg

import sklearn.datasets

MLCOMP_DIR=r"news"
data = sklearn.datasets.load_mlcomp("20news-18828",mlcomp_root=MLCOMP_DIR)
print(data.filenames)

print(len(data.filenames))
print(data.target_names)

train_data = sklearn.datasets.load_mlcomp("20news-18828","train",
											mlcomp_root=MLCOMP_DIR)

print(len(train_data.filenames))
test_data= sklearn.datasets.load_mlcomp("20news-18828","test",
										mlcomp_root = MLCOMP_DIR)
print(len(test_data.filenames))

groups = ['comp.graphics','comp.os.ms-windows.misc',
'comp.sys.ibm.pc.hardware','comp.sys.mac.hardware',
'comp.windows.x','sci.space']
train_data=sklearn.datasets.load_mlcomp("20news-18828","train",
										mlcomp_root=MLCOMP_DIR,
										categories=groups)

print(len(train_data.filenames))


from sklearn.feature_extraction.text import TfidfVectorizer	
english_stemmer = nltk.stem.SnowballStemmer('english')	
class StemmedTfidfVectorizer(TfidfVectorizer):
	def build_analyzer(self):
		analyzer = super(TfidfVectorizer,self).build_analyzer()
		return lambda doc:(english_stemmer.stem(w) for w in analyzer(doc))

vectorizer = StemmedTfidfVectorizer(min_df=10,max_df=0.5,stop_words="english",charset_error='ignore')
vectorized = vectorizer.fit_transform(train_data.data)
num_samples,num_features=vectorized.shape
print("#samples: %d, #features: %d" % (num_samples,num_features))

num_clusters = 50
from sklearn.cluster import KMeans
km = KMeans(n_clusters=num_clusters, init='random',n_init=1,verbose=1)
km.fit(vectorized)

print(km.labels_)
print(km.labels_.shape)

new_post="Disk drive problems. Hi, I have a problem with my hard disk. After 1 year it is working only sporadically now. I tried to format it, but now it doesn't boot any more. Any ideas? Thanks."

new_post_vec = vectorizer.transform([new_post])
new_post_label = km.predict(new_post_vec)[0]

similar_indices = (km.labels_==new_post_label).nonzero()[0]

similar=[]
for i in similar_indices:
	dist = sp.linalg.norm((new_post_vec - vectorized[i]).toarray())
	similar.append((dist,train_data.data[i]))
similar = sorted(similar)
print(len(similar))


print(similar[0])
print(similar[len(similar)/2])
print(similar[-1])

print("\n")

#section3.4.1
post_group = zip(train_data.data,train_data.target)
z=[(len(post[0]),post[0],train_data.target_names[post[1]])for post in post_group]
print(sorted(z)[5:7])

print("\n")


analyzer = vectorizer.build_analyzer()
print(list(analyzer(z[5][1])))
print(list(analyzer(z[6][1])))

print("\n")


print(list(set(analyzer(z[5][1])).intersection(vectorizer.get_feature_names())))
print(list(set(analyzer(z[6][1])).intersection(vectorizer.get_feature_names())))

for term in ['cs', 'faq', 'thank', 'bh', 'thank']:
	print('IDF(%s)=%.2f'%(term,vectorizer._tfidf.idf_[vectorizer.vocabulary_[term]]))