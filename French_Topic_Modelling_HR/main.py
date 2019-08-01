# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 18:34:33 2019

@author: Hincal
"""

import pandas as pd
import numpy as np
import os
import re
from unidecode import unidecode
import lda
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
import bokeh.plotting as bp
from bokeh.plotting import save
from bokeh.models import HoverTool
import logging
import logging.handlers
import configparser



# ============================================================================
# CONFIG PARSER
# ============================================================================
config = configparser.ConfigParser()
config.read("config.ini")
config.sections()
CSV_FILES_DIR = config["CSV_FILES_DIR"]["dir"]
LOGPATH = config["LOGPATH"]["path"]
STOPWORDS = config["STOPWORDS"]["stopwrds"]
PLOTS = config["PLOTS"]["plots"]

# ============================================================================
# LOGGER CONFIG
# ============================================================================
if not os.path.exists(LOGPATH):
    os.mkdir(LOGPATH)

LOGFILE = f"{LOGPATH}/info.log"

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

# create a file handler
HANDLER = logging.handlers.RotatingFileHandler(
    LOGFILE, maxBytes=100 * 1024 * 1024, backupCount=5)
HANDLER.setLevel(logging.INFO)

# create a logging format
FORMATTER = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
HANDLER.setFormatter(FORMATTER)

# add the handlers to the logger
LOGGER.addHandler(HANDLER)


# ============================================================================
# READ DATA
# ============================================================================


def read_data():
    _lst = os.listdir(CSV_FILES_DIR)
    for csv in _lst:
        data = pd.read_csv(CSV_FILES_DIR + csv)
        df = data[['Regions','Offer']]
    return df

def read_stopwords():
    _stop = os.listdir(STOPWORDS)
    for word in _stop:
        with open(STOPWORDS + word, 'r') as f:
            stopwords=[i for line in f for i in line.split(',')]
                
    return stopwords


stopwords=read_stopwords()


# ============================================================================
# LOWER LETTERS
# ============================================================================
def lower_case():
    df = read_data()
    OfferLower = df["Offer"].map(lambda x: x if type(x)!=str else x.lower())
    OfferLower2=[]
    for i in OfferLower:
        if type(i) == str:
            OfferLower2.append(unidecode(i))
        
    RegionsLower = df["Regions"].map(lambda x: x if type(x)!=str else x.lower())
    RegionsLower2=[]
    for i in RegionsLower:
        if type(i) == str:
            RegionsLower2.append(unidecode(i))
        
    df2 = pd.DataFrame(list(zip(RegionsLower2, OfferLower2)))
    df2 = df2.rename(columns={0: 'Regions', 1: 'Offer'})
    return df2


    
# ============================================================================
# Doing a first cleaning of the texts (including some stopwords)
# ============================================================================
def clean_text(text):
    #text = re.sub("[^a-zA-Z]", " ", str(text))
    text = re.sub(r"[_______\"&>>/»;:+!@%()-<>{}=~|.?,:]", "", text)
    text = re.sub('\s+', ' ', text).strip()
    #text = re.sub(r"d''", "d'", text)
    text = re.sub(r" plm ", " ", text)
    text = re.sub(r" un ", " ", text)
    text = re.sub(r" une ", " ", text)
    text = re.sub(r" de ", " ", text)
    text = re.sub(r" des ", " ", text)
    text = re.sub(r" son ", " ", text)
    text = re.sub(r" en ", " ", text)
    text = re.sub(r" tres ", " ", text)
    text = re.sub(r" a la ", " ", text)
    text = re.sub(r" a ", " ", text)
    text = re.sub(r" d ", " ", text)
    text = re.sub(r" la ", " ", text)
    text = re.sub(r" et ", " ", text)
    text = re.sub(r" aux ", " ", text)
    text = re.sub(r" les ", " ", text)
    text = re.sub(r" du ", " ", text)
    text = re.sub(r" avec ", " ", text)
    text = re.sub(r" par ", " ", text)
    text = re.sub(r" il ", " ", text)
    text = re.sub(r" sur ", " ", text)
    text = re.sub(r" que ", " ", text)
    text = re.sub(r" le ", " ", text)
    text = re.sub(r" vous ", " ", text)
    text = re.sub(r" pour ", " ", text)
    text = re.sub(r" ces ", " ", text)
    text = re.sub(r" ce ", " ", text)
    text = re.sub(r" dos ", " ", text)
    text = re.sub(r" dans ", " ", text)
    text = re.sub(r" est ", " ", text)
    text = re.sub(r" et ", " ", text)
    text = re.sub(r" eu ", " ", text)
    text = re.sub(r" ils ", " ", text)
    text = re.sub(r" je ", " ", text)
    text = re.sub(r" leur ", " ", text)
    text = re.sub(r" ma ", " ", text)
    text = re.sub(r" mes ", " ", text)
    text = re.sub(r" plus ", " ", text)
    text = re.sub(r" v ", " ", text)
    
    return(text)    
    
# ============================================================================
# Cleaning the french stopwords)
# ============================================================================    
def cleaning():
    df2 = lower_case()
    
    clean_offers = []
    for offer in df2['Offer']:
        clean_offers.append(clean_text(offer))
                 
    # Filtering out the stopwords from offers 
    clean_offers2 = {}
    for x, elem in enumerate(clean_offers):
       clean_offers2[x] = " ".join(filter(lambda x: x not in stopwords , elem.split()))
    
    clean_offers2_list = [v1 for v1 in clean_offers2.values()]
   
   # Filtering out the offer words which are too short 
    clean_offers3 = {}
    for i, elem in enumerate(clean_offers2_list):
       clean_offers3[i] = " ".join(filter(lambda i: len(i) > 2, elem.split()))
    
    clean_offers3_list = [v1 for v1 in clean_offers3.values()]
    
    # Create final dictionary which will be used in calculation
    df3 = pd.DataFrame(list(zip(df2.Regions,clean_offers3_list)))
    df3 = df3.rename(columns={0: 'Regions', 1: 'Offers'})
    df3_dict = df3.groupby('Regions')['Offers'].apply(list).to_dict()

    dict_final = {}
    for k,v in df3_dict.items():
       if len(v)>1:
           dict_final[k] = v
   
    return dict_final
    
    
# ============================================================================
# Calculate Term Frequencies and train LDA and TSNE model
# ============================================================================
def tf_lda_tsne(n_topics=10, n_iter=500,n_components=2, verbose=1, random_state=0, angle=.99,init='pca'):
    n_topics = n_topics # number of topics
    n_iter = n_iter # number of iterations
    tf_counts = {}
    vocab = {}
    cvectorizer = CountVectorizer(min_df=1, max_df=4, stop_words=stopwords)
    dict_final = cleaning()
    
    for m,n in enumerate(dict_final):
        tf_counts[n] = cvectorizer.fit_transform(dict_final[n])
        vocab[n] = cvectorizer.get_feature_names()
        
    lda_model = lda.LDA(n_topics=n_topics, n_iter=n_iter)
    X_topics = {}
    topic_word = {}
    for t,elem in enumerate(tf_counts):
        X_topics[elem] = lda_model.fit_transform(tf_counts[elem])
        topic_word[elem] = lda_model.topic_word_ 
    
    # a t-SNE model
    # angle value close to 1 means sacrificing accuracy for speed
    # pca initializtion usually leads to better results 
    tsne_model = TSNE(n_components=n_components, 
                      verbose=verbose, 
                      random_state=random_state, 
                      angle=angle, 
                      init=init)

    tsne_lda = {}
    for a,b in enumerate(X_topics):
        tsne_lda[b] = tsne_model.fit_transform(X_topics[b])

    return X_topics,topic_word,vocab,tsne_lda,dict_final


# ============================================================================
# Prepare Data For
# ============================================================================

def plot_bokeh(n_top_words=5,plot_width=1600,plot_height=700):
    X_topics, topic_word, vocab, tsne_lda, dict_final = tf_lda_tsne()
    _lda_keys_dict = {}
    for m1,m2 in enumerate(X_topics):
        for i in range(X_topics[m2].shape[0]):
            _lda_keys_dict[m2,i] = (X_topics[m2][i].argmax())
        
    _lda_keys_lst = []
    for key,value in _lda_keys_dict.items():
        _lda_keys_lst.append((key[0],value))
    
    _lda_keys_df = pd.DataFrame(_lda_keys_lst)
    _lda_keys = _lda_keys_df.groupby(0)[1].apply(list).to_dict()
      
    n_top_words = n_top_words
    topic_words = {}
    for i, k in enumerate(topic_word):
        for t in range(10):
            topic_words[k,t] = list(np.array(vocab[k])[np.argsort(topic_word[k])][t][:-(n_top_words + 1):-1])
    
    topic_summaries_lst = []
    for key,value in topic_words.items():
        topic_summaries_lst.append((key[0],value))

    topic_summaries_df = pd.DataFrame(topic_summaries_lst)
    topic_summaries = topic_summaries_df.groupby(0)[1].apply(list).to_dict()

    for key,value in topic_summaries.items():
        templist = []
        for single in topic_summaries[key]:
            templist.append(' '.join(single))
        topic_summaries[key] = templist
    
    
    num_example = {}
    for t1,t2 in enumerate(X_topics):
        num_example[t2] = len(X_topics[t2])

    colormap = np.array([
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
    "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5"])
    

    plot_dict = {}
    for h1,h2 in enumerate(tsne_lda):
        plot_dict[h2] = {'x':pd.DataFrame(tsne_lda[h2]).iloc[:,0], 
                         'y':pd.DataFrame(tsne_lda[h2]).iloc[:,1], 
                         'colors':colormap[_lda_keys[h2]][:num_example[h2]],
                         'content':dict_final[h2][:num_example[h2]],
                         'topic_key': _lda_keys[h2][:num_example[h2]]
                        }
    
    
    # declare the source    
    for r1,r2 in enumerate(plot_dict):
        source = bp.ColumnDataSource(data=pd.DataFrame.from_dict(plot_dict[r2]))
        title = r2
        plot_lda = bp.figure(plot_width=plot_width, plot_height=plot_height,
                     title=title,
                     tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
                     x_axis_type=None, y_axis_type=None, min_border=1)
        plot_lda.scatter('x', 'y', color='colors', source=source)
        topic_coord = np.empty((X_topics[r2].shape[1], 2)) * np.nan
        topic_summaries2 = {}
        for topic_num in _lda_keys[r2]:
            if not np.isnan(topic_coord).any():
                break
            topic_coord[topic_num] = tsne_lda[r2][_lda_keys[r2].index(topic_num)]
            topic_coord2 = topic_coord[~np.isnan(topic_coord).any(axis=1)]
            nans = np.where(np.isnan(topic_coord))
            nans_uniq = np.unique(nans[0])
            if nans_uniq.size != 0:
                topic_summaries2[r2] = [j for i, j in enumerate(topic_summaries[r2]) if i not in nans_uniq]
            else:
                topic_summaries2[r2] = topic_summaries[r2]
        # plot crucial words
        for i in range(len(topic_coord2)):
            plot_lda.text(topic_coord2[i, 0], topic_coord2[i, 1], [topic_summaries2[r2][i]])
        # hover tools    
        hover = plot_lda.select(dict(type=HoverTool))
        hover.tooltips = {"content": "@content - topic: @topic_key"}
        # save the plot
        save(plot_lda, f"{PLOTS}"'{}.html'.format(title))
    
    
    
if __name__ == "__main__":
    plot_bokeh()