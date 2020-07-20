#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
data = pd.read_csv('LDA.csv', error_bad_lines=False);
data=data.astype(str)
data_text = data[['abstract']]
data_text['index'] = data_text.index
documents = data_text


# **the sorted data with combined abstract for a year**

# In[4]:


data.columns=['index','author','year','abstract']

for index,row in data.iterrows():
    if row['year']=="-1":
        row['year']="1996"
data['newid'] = np.arange(len(data))


# In[18]:


data


# In[5]:


print(len(documents))
print(documents[:5])


# In[6]:


import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
stemmer=PorterStemmer()
import numpy as np
np.random.seed(2018)
import nltk
nltk.download('wordnet')


# **cleaning texts**

# In[7]:


def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


# **splitting words**

# In[24]:


words=[]
doc_sample=documents[documents['index']==2644].values[0][0]
for word in doc_sample.split(' '):
    words.append(word)
print(words)


# In[8]:


processed_docs = documents['abstract'].map(preprocess)
processed_docs[:10]


# In[9]:


dictionary = gensim.corpora.Dictionary(processed_docs)
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break


# **keeping first 100000 frequent tokens**
# 

# In[20]:


dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)


# **number of words and their count for each author and year**

# In[30]:


bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
bow_corpus[2644]


# **preview of bag of words**

# In[31]:


bow_doc_2644 = bow_corpus[2644]
for i in range(len(bow_doc_2644)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_2644[i][0], 
                                               dictionary[bow_doc_2644[i][0]], 
bow_doc_2644[i][1]))


# **TF-IDF**

# In[32]:


from gensim import corpora, models
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
from pprint import pprint
for doc in corpus_tfidf:
    pprint(doc)
    break


# **LDA using bag of words**

# In[33]:


lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=50, id2word=dictionary, passes=2, workers=2)


# **words occuring in a topic and its relative weight**

# In[34]:


for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))


# **LDA using TF-IDF**

# In[35]:


lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=50, id2word=dictionary, passes=2, workers=4)
for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))


# **Checking where a given pre-processed abstract would be classified**

# In[36]:


for index, score in sorted(lda_model_tfidf[bow_corpus[2644]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))


# **training lda model**

# In[37]:


#lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=50, id2word=dictionary, passes=2, workers=2)
lda =models.ldamodel.LdaModel(bow_corpus, num_topics=50)
from gensim.test.utils import datapath
temp_file = datapath("model")
lda.save(temp_file)
lda = models.ldamodel.LdaModel.load(temp_file)


# In[38]:


reviewers=np.zeros(shape=(10,50))
finalr=np.zeros(shape=(364,50))
rl=[]


# In[39]:


r=pd.read_csv('reviewers.txt',sep='\t',names=['index','reviewer'])


# In[40]:


r


# In[41]:


for index,row in r.iterrows():
    rl.append(row['reviewer'])


# In[42]:


rl[363]


# In[43]:


rl.sort()
rl


# In[44]:


print(len(rl))


# In[45]:


yr=[1996,1997,1998,1999,2000,2001,2002,2003,2004,2005]
import math


# i=0
# for index,row in data.iterrows():
#     for x in rl:
#         for y in yr:
#             if row['year']==y and row['author']==x:
#                 doc=row['abstract']
#                 
#             vector=lda[doc]
#             #id=y-1996
#             a=2005-y+2
#             for idx,val in vector:
#                 val=val/math.log2(a)
#                 reviewers[id][idx]+=val
#         for j in range(0,10):
#             finalr[i]+=reviewers[j]
#             print(finalr[i])
#         i=i+1

# for it in range(0,2645):
#     vector=lda[bow_corpus[it]]
#     for index,row in data.iterrows():
#         if row['index']==it:
#             yr=row['year']
#             a=2005-yr+2 
#             au=row['author']
#             i=0
#             for r in rl:
#                 if r==au:
#                     rin=i
#                 i=i+1
#             for idx,val in vector:
#                 val=val/math.log2(a)
#                 reviewers[yr-1996][idx]+=val
#             break

# In[46]:


for i in range(0,364):
    t=rl[i]
    for index,row in data.iterrows():
        if row['author']==t:
            yr=int(row['year'],10)
            a=2005-yr+2
            index=row['newid']
            vector=lda[bow_corpus[index]]
            for idx,val in vector:
                val=val/math.log2(a)
                reviewers[yr-1996][idx]+=val
        else:
            for j in range(0,10):
                finalr[i]+=reviewers[j]
            reviewers=np.zeros(shape=(10,50))


# In[60]:


finalr


# In[56]:


np.savetxt("rlda.csv",finalr, delimiter=",")


# In[61]:


rlda=pd.read_csv("rlda.csv")


# In[62]:


rlda


# **evaluating cosine similarity between reviewers**

# In[85]:


cos=np.zeros(shape=(1,364))
cosine=np.zeros(shape=(364,364))


# In[88]:


from sklearn.metrics.pairwise import cosine_similarity


# In[90]:


for i in range(364):
    a=finalr[i].reshape(1,50)
    for j in range(i+1,364):
        b=finalr[j].reshape(1,50)
        cos_i=cosine_similarity(a, b)
        cosine[i][j]=cos_i
        


# In[91]:


cosine


# In[92]:


np.save('outfile_name',cosine)


# In[ ]:




