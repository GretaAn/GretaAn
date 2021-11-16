#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install praw --upgrade praw


# In[ ]:


import praw
from praw.models import MoreComments
import string
import re
import nltk 
from nltk.text import Text
from nltk.draw.dispersion import dispersion_plot
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from nltk.collocations import BigramCollocationFinder


# ## Load raw data

# In[ ]:


def get_post_data():
    reddit = praw.Reddit(client_id='-sgTOKuB8OpYmcvV0gf1DQ',                          client_secret='p3cKCHZ4xhK0PVUzB9KUhupYpNvuHg',                          user_agent='crawler')
    url = "https://www.reddit.com/r/Coronavirus/comments/pbo8yv/misinformation_is_bad_good_information_is_good/"
    data = reddit.submission(url=url)

    return data


# In[ ]:


raw_posts = get_post_data()


# ## Inspect raw data

# In[ ]:


def inspect_raw_posts(raw_posts):
    for top_level_comment in raw_posts.comments:
        print(top_level_comment.body)


# In[ ]:


inspect_raw_posts(raw_posts)


# ## Clean Text

# In[ ]:


def remove_links(post):
    post = re.sub("http\S*\s", "", post)
    return post


# In[ ]:


def lemmatize(post):
    lemmatizer = WordNetLemmatizer()
    lemmatized_post = lemmatizer.lemmatize(post) 
    return lemmatized_post


# In[ ]:


def remove_punctuation(post):
    table = str.maketrans('', '', string.punctuation) 
    post = [w.translate(table) for w in post] 
    return post


# In[ ]:


def remove_stopwords(post):
    stop_words = nltk.corpus.stopwords.words('english')
    new_stopwords = ["would" , "etc" , "url" , "https" , "jpeg"] 
    stop_words.extend(new_stopwords) 
    post = [w for w in post if not w in stop_words] 
    
    return post
  


# In[ ]:


def remove_digits(post):
    post = [w for w in post if w.isalpha()]
    return post


# In[ ]:


def clean_text(post):
   
    post = post.lower()
    post = remove_links(post)
    post = lemmatize(post)
    post = word_tokenize(post) 
    post = remove_punctuation(post)
    post = remove_stopwords(post)
    post = remove_digits(post)
  
    return post
    


# ## Apply text cleaning

# In[ ]:


def apply_text_cleaning(raw_posts):
    cleaned_posts = []
    for top_level_comment in raw_posts.comments[1:]:

        if isinstance(top_level_comment, MoreComments):
            continue #remove second level comments

        raw_post = top_level_comment.body
        cleaned_post = clean_text(raw_post)

        if cleaned_post in ["removed" , "deleted"]: #remove deleted comments
            continue

        cleaned_posts.append(cleaned_post)
        
    return cleaned_posts


# In[ ]:


cleaned_posts = apply_text_cleaning(raw_posts) 


# ## Check results of text cleaning

# In[ ]:


def inspect_cleaned_posts(cleaned_posts):
    for post in cleaned_posts:
        print(post ,"\n" )


# In[ ]:


inspect_cleaned_posts(cleaned_posts)


# ## Prepare tokens for analysing

# In[ ]:


def transform_posts(cleaned_posts):
    transformed_posts = []
    for post in cleaned_posts:
        transformed_posts.extend(post)
    transformed_posts = Text(transformed_posts)
    return transformed_posts


# In[ ]:


transformed_posts = transform_posts(cleaned_posts) 


# ### Lexical dispertion

# In[ ]:


targets=["misinformation" , "information", "vaccination", "vaccine" , "virus" , "bias"]
dispersion_plot(transformed_posts, targets, title='Lexical Dispersion Plot')


# ### Frequency distribution

# In[ ]:



frequency_distribution = FreqDist(transformed_posts)
frequency_distribution.most_common(20)


# In[ ]:


frequency_distribution.plot(20, cumulative=False)


# ### Collocations of "vaccine"

# In[ ]:


bigram_measures = nltk.collocations.BigramAssocMeasures()
word_filter = lambda *w: 'vaccine' not in w
finder = BigramCollocationFinder.from_words(transformed_posts)
finder.apply_ngram_filter(word_filter)
finder.nbest(bigram_measures.likelihood_ratio , 10)


# In[ ]:


transformed_posts.collocations(10)


# In[ ]:




