# Libraries imports
import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
import re
import tashaphyne.normalize as normalize
import nltk
from snowballstemmer import stemmer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
import os

def preprocessing_pipeline(data_path, validation_split, number_features):
    '''
    Args:
        String: Arabic dataset CSV file path. (Expected columns String: id, String: tweet).
        Float: validation percentage parameter for spliting the data. Must be between 0 and 1.
        Int: Number of max features to pass to the vectorizer.
    output
        Four np arrays: x_tarin, x_test, y_train, y_test
    '''
    assert validation_split >= 0 and validation_split <= 1
    start = datetime.now()
    # Create path to input files
    inputs_path = os.path.abspath('../inputs/')
    print('Preprocessing started at: {}'.format(start))
    # Import data
    dialects_data = pd.read_csv(data_path)
    dialects_data = merge_similar_dialects(dialects_data)
    # Balance data using random undersampling method
    dialects_data = tweets_undersampling(dialects_data)
    # Remove unwanted characters
    dialects_data['tweet'] = dialects_data['tweet'].apply(remove_unwanted_characters)
    print('Unwanted charaters removed.')
    # Normalize arabic text
    dialects_data['tweet'] = dialects_data['tweet'].apply(normalize_arabic)
    print('Text normalization applied.')
    # Remove names from tweets
    names_list = remove_arabic_names()
    dialects_data['tweet'] = dialects_data['tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (names_list)]))
    print ('Names removed from tweets!')
    # Remove stop words
    dialects_data = remove_stop_words(dialects_data)
    '''
    # Update: I decided to remove stemming for now as I think that it affects dialects classification negatevially.
    dialects_data['tweet'] = dialects_data['tweet'].apply(stem_text)
    print('Text stemming applied.')
    '''
    # Tagets label encoding
    dialects_data['dialect'] = labels_encoding(dialects_data['dialect'])
    # Splitting data
    tweets = dialects_data['tweet']
    dialect_target = dialects_data['dialect']
    x_train, x_test, y_train, y_test = train_test_split(tweets, dialect_target, test_size = validation_split, random_state = 0)
    print('Data was split with was split with test_size = {}.'.format(validation_split))
    # Apply bag of words technique
    x_train, x_test = data_tf_idf_vectorization(x_train, x_test, number_features)
    print('Preprocessing took: {}.'.format(datetime.now() - start))
    return x_train, x_test, y_train, y_test

def merge_similar_dialects(dialects_data):
    # Merge gulf dialcets
    dialects_data['dialect'].replace(['SA', 'AE', 'QA', 'BH', 'KW', 'OM', 'IQ'], 'GUL', inplace=True)
    #   Merge noth african countries (Except for egyptyian)
    dialects_data['dialect'].replace(['LIB', 'DZ', 'MA', 'TN', 'LY'], 'NA', inplace=True)
    # Merge Levantine countries
    dialects_data['dialect'].replace(['SY', 'PL', 'JO', 'LB'], 'LV', inplace=True)
    print('Dialects merged into geographic areas.')
    return dialects_data

def tweets_undersampling(dialects_data):
    undersampler = RandomUnderSampler(sampling_strategy={'GUL': 50000, 'LV': 50000, 'EG': 50000, 'NA': 50000})
    X_under, y_under = undersampler.fit_resample(dialects_data['tweet'].values.reshape(-1, 1), dialects_data['dialect'])
    X_under = pd.Series(X_under.ravel())
    under_sampled_data = pd.concat([X_under, y_under], axis = 1)
    under_sampled_data.rename(columns={0: 'tweet'}, inplace = True)
    # Drop minority classes
    under_sampled_data.drop(under_sampled_data.loc[under_sampled_data['dialect']=='YE'].index, inplace=True)
    under_sampled_data.drop(under_sampled_data.loc[under_sampled_data['dialect']=='SD'].index, inplace=True)
    print('Undersampling finshed, new number of samples = {}.'.format(under_sampled_data.shape[0]))
    return under_sampled_data

def remove_unwanted_characters(tweet):
    #tweet = re.sub('/#\w+\s*[A-Za-z]+\b','',tweet) # Remove hashtags
    tweet = re.sub('@[^\s]+','',tweet) # Remove user name and mentions
    tweet = re.sub('http[^\s]+','',tweet) # Remove Hyper links
    tweet = re.sub('[^\w\s#@/:%.,_-]','',tweet) # Remove emojis
    tweet = re.sub('\d+','',tweet) # Remove numbers
    tweet = re.sub('[^\w\s]','',tweet) # Remove punctuation
    tweet = re.sub(r'[a-zA-Z]','',tweet, flags=re.I) # Remove english words and add flag to remove dotlees i
    tweet = re.sub(r'[\_-]', '', tweet) # Remove underscores and dashes
    tweet = re.sub(r'\n', '', tweet)
    tweet = tweet.rstrip()
    return tweet

def normalize_arabic(tweet):
    normalized_data = normalize.normalize_searchtext(tweet)
    return normalized_data

def remove_arabic_names():
    # Gather Names from local documents
    arabic_names = pd.read_csv('preprocessing/names_stopwords/arabic_names.csv', encoding = 'utf-8')
    # drop gender column
    arabic_names.drop('Gender', axis = 1, inplace = True)

    names_list = arabic_names.values.ravel()
    # Get list of stopwords from txt file
    with open('preprocessing/names_stopwords/arabic_names.txt', 'r', encoding="utf8") as file:
        names = set(file.read().split())

    # Join list form csv and txt files
    return names.union(names_list)

def remove_stop_words(dialects_data):
    with open('preprocessing/names_stopwords/arabic_stopwords.txt', 'r', encoding="utf8") as file:
        stopwords = file.read().split()
    # Create stop words list by joining nltk stopwords with my own.
    arab_stopwords = set(nltk.corpus.stopwords.words("arabic"))
    joint_stop_words = arab_stopwords.union(stopwords)
    dialects_data['tweet'] = dialects_data['tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in joint_stop_words]))
    print('Stop words removed.')
    return dialects_data

def stem_text(tweet):
    ar_stemmer = stemmer("arabic")
    newtweet = ar_stemmer.stemWord(tweet)
    return newtweet

def labels_encoding(targets):
    le = LabelEncoder()
    encoded_targets = le.fit_transform(targets)
    print('Classes labels encoded.')
    return encoded_targets

def data_tf_idf_vectorization(x_train, x_test, number_features):
    tfidf_vectorizer = TfidfVectorizer(max_features = number_features)
    # Transform data
    X_train_tfidf = tfidf_vectorizer.fit_transform(x_train).toarray()
    X_test_tfidf = tfidf_vectorizer.transform(x_test).toarray()
    return X_train_tfidf, X_test_tfidf
