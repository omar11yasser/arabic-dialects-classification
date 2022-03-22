# Libraries imports
import pandas as pd
import numpy as np
import imblearn
from imblearn.under_sampling import RandomUnderSampler
import re
import tashaphyne.normalize as normalize
import nltk
from snowballstemmer import stemmer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from datetime import datetime

male_names_path = 'inputs/males_ar.csv'
female_names_path = 'inputs/females_ar.csv'

def preprocessing_pipe(data_path):
    start = datetime.now()
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
    # Text stemming
    dialects_data['tweet'] = dialects_data['tweet'].apply(stem_text)
    print('Text stemming applied.')
    # Tagets label encoding
    dialects_data['dialect'] = classes_label_encoding(dialects_data['dialect'])
    # Splitting data
    tweets = dialects_data['tweet']
    dialect_target = dialects_data['dialect']
    x_train, x_test, y_train, y_test = train_test_split(tweets, dialect_target, test_size = 0.2, random_state = 0)
    print('Data was split with was split with test_size = 0.2')
    # Apply bag of words technique
    x_train, x_test = data_vectorization(x_train, x_test)
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
    male_names = pd.read_csv(male_names_path)
    female_names = pd.read_csv(female_names_path)
    # Concat both gender names
    arabic_names = pd.concat([male_names, female_names])
    # drop gender column
    arabic_names.drop('Gender', axis = 1, inplace = True)
    # Normalize arabic names
    arabic_names['Name'] = arabic_names['Name'].apply(normalize_arabic)
    #Put names into a list
    names_list = arabic_names.values.ravel()
    return names_list

def remove_stop_words(dialects_data):
    arab_stopwords = set(nltk.corpus.stopwords.words("arabic"))
    additional_stopwords = {'بس','يوم','ﻣﺎ','و', 'ﻣﻦ', 'ﻓﻲ', 'الي', 'يونيو','اب', 'ام','اه','ابريل','هو','هي','اللي','يا','لما','لو','لذلك'}
    joint_stop_words = arab_stopwords.union(additional_stopwords)
    dialects_data['tweet'] = dialects_data['tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (joint_stop_words)]))
    print('Stop words removed.')
    return dialects_data

def stem_text(tweet):
    ar_stemmer = stemmer("arabic")
    newtweet = ar_stemmer.stemWord(tweet)
    return newtweet

def classes_label_encoding(targets):
    le = LabelEncoder()
    encoded_targets = le.fit_transform(targets)
    print('Classes labels encoded.')
    return encoded_targets

def data_vectorization(x_train, x_test):
    vectorizer = CountVectorizer(analyzer = "word",
                             tokenizer = None,
                             preprocessor = None,
                             stop_words = None,
                             max_features = 5000)
    # fit vectorizer to train data and transform the data
    train_data_features = vectorizer.fit_transform(x_train)
    train_data_features = train_data_features.toarray()
    # Transform testing data with vectorizer
    test_data_features = vectorizer.transform(x_test)
    # ...
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(train_data_features)
    X_test_tfidf = tfidf_transformer.transform(test_data_features)
    print('Data vectorization finished.')
    return X_train_tfidf, X_test_tfidf
