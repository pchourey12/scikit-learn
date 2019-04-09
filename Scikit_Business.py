
# coding: utf-8

# In[ ]:


####
#Parameters:	
container_path : string or unicode
Path to the main folder holding one subfolder per category

description : string or unicode, optional (default=None)
A paragraph describing the characteristic of the dataset: its source, reference, etc.

categories : A collection of strings or None, optional (default=None)
If None (default), load all the categories. If not None, list of category names to load (other categories ignored).

load_content : boolean, optional (default=True)
Whether to load or not the content of the different files. If true a ‘data’ attribute containing the text information is present in the data structure returned. If not, a filenames attribute gives the path to the files.

shuffle : bool, optional (default=True)
Whether or not to shuffle the data: might be important for models that make the assumption that the samples are independent and identically distributed (i.i.d.), such as stochastic gradient descent.

encoding : string or None (default is None)
If None, do not try to decode the content of the files (e.g. for images or other non-text content). If not None, encoding to use to decode text files to Unicode if load_content is True.

decode_error : {‘strict’, ‘ignore’, ‘replace’}, optional
Instruction on what to do if a byte sequence is given to analyze that contains characters not of the given encoding. Passed as keyword argument ‘errors’ to bytes.decode.

random_state : int, RandomState instance or None (default=0)
Determines random number generation for dataset shuffling. Pass an int for reproducible output across multiple function calls. See Glossary.


# In[47]:


import os
from urllib.request import Request, build_opener

import lxml.html
from lxml.etree import ElementTree
import numpy as np
import os
import re
import codecs
from cogxparser import CSXmlParser as csp

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import sklearn
from sklearn import datasets
from pprint import pprint 
from sklearn.pipeline import Pipeline

docs_to_train = sklearn.datasets.load_files("M:/MASTER/CSI/Cogito/Business/AI_Business/Business/training", description=None, categories=None, load_content=True, shuffle=True, encoding='utf-8', decode_error='strict', random_state=0)

pprint(list(docs_to_train.target_names))


# In[49]:


len(docs_to_train.data)


# In[51]:


len(docs_to_train.filenames)


# In[52]:


print("\n".join(docs_to_train.data[0].split("\n")[:3])


# In[54]:


print(docs_to_train.target_names[docs_to_train.target[0]])


# In[55]:


#Text preprocessing, tokenizing and filtering of stopwords are all included in CountVectorizer, which builds a dictionary of features and transforms documents to feature vectors:
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(docs_to_train.data)
X_train_counts.shape


# In[56]:


#CountVectorizer supports counts of N-grams of words or consecutive characters. Once fitted, the vectorizer has built a dictionary of feature indices:
count_vect.vocabulary_.get(u'algorithm')


# In[58]:


#Both tf and tf–idf can be computed as follows using TfidfTransformer:
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
X_train_tf.shape


# In[61]:


tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape


# In[62]:


#Training a classifier
clf = MultinomialNB().fit(X_train_tfidf, docs_to_train.target)


# In[63]:


docs_new = ['The Africa-EU Strategic Partnership Agreement and Regional Integration in West Africa', 'Bioeconomic Model of Spatial Fishery Management in Developing Countries']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)
for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, docs_to_train.target_names[category]))
    
    #Partnership agreements-4754668 , Business enterprises- 227163
    # AN : 1368033, 1305025


# In[66]:


from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB())])
text_clf.fit(docs_to_train.data, docs_to_train.target) 


# In[67]:


#Building a Pipeline
docs_to_test= sklearn.datasets.load_files("M:/MASTER/CSI/Cogito/Business/AI_Business/Business/testing", description=None, categories=None, load_content=True, shuffle=True, encoding='utf-8', decode_error='strict', random_state=0)
docs_test = docs_to_test.data
predicted = text_clf.predict(docs_test)
np.mean(predicted == docs_to_test.target)


# In[69]:


#Support VEctor Machine
from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([
('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=5, tol=None))])

text_clf.fit(docs_to_train.data,docs_to_train.target)  
predicted = text_clf.predict(docs_test)
np.mean(predicted == docs_to_test.target)  


# In[71]:


from sklearn import metrics
print(metrics.classification_report(docs_to_test.target, predicted,target_names=docs_to_test.target_names))


# In[72]:


metrics.confusion_matrix(docs_to_test.target, predicted)

