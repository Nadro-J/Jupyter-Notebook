import nltk

nltk.download('punkt')

from nltk.tokenize import word_tokenize
from nltk import pos_tag

nltk.download('stopwords')

from nltk.corpus import stopwords

nltk.download('wordnet')

from nltk.corpus import wordnet

from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()

from textblob import TextBlob

import pandas as pd
import re

# Create pandas dataframe from text.csv
csv_extract = pd.read_csv('twitter-extract.csv', sep=',')


def cleanse_data(text):
    return re.sub('[^A-Za-z]+', ' ', text)


csv_extract['tweet_text_clean'] = csv_extract['tweet_text'].apply(cleanse_data)

pos_dict = {'J': wordnet.ADJ, 'V': wordnet.VERB, 'N': wordnet.NOUN, 'R': wordnet.ADV}


def token_stop_pos(text):
    tags = pos_tag(word_tokenize(text))
    newlist = []

    for word, tag in tags:
        if word.lower() not in set(stopwords.words('english')):
            newlist.append(tuple([word, pos_dict.get(tag[0])]))

    return newlist


csv_extract['POS tagged'] = csv_extract['tweet_text_clean'].apply(token_stop_pos)


def lemmatize(pos_data):
    lemma_rew = " "
    for word, pos in pos_data:
        if not pos:
            lemma = word
            lemma_rew = lemma_rew + " " + lemma
        else:
            lemma = wordnet_lemmatizer.lemmatize(word, pos=pos)
            lemma_rew = lemma_rew + " " + lemma
    return lemma_rew


csv_extract['lemmatize'] = csv_extract['POS tagged'].apply(lemmatize)


# function to calculate subjectivity
def getSubjectivity(review):
    return TextBlob(review).sentiment.subjectivity


# function to calculate polarity
def getPolarity(review):
    return TextBlob(review).sentiment.polarity


# function to analyze the reviews
def analysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'


fin_data = pd.DataFrame(csv_extract[['tweet_text', 'lemmatize']])
fin_data['Polarity'] = fin_data['lemmatize'].apply(getPolarity)
fin_data['Analysis'] = fin_data['Polarity'].apply(analysis)

tb_counts = fin_data.Analysis.value_counts()

print("--- Results ---")
print(f"Number of tweets in 24H: {len(fin_data)}")
print("...")
print(tb_counts)
