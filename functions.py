# dataframe libraries
import numpy as np
import pandas as pd

# twitter scraper
import twint
# fixes runtime errors with twint
import nest_asyncio
nest_asyncio.apply()

# text processing
import nltk
from nltk import RegexpTokenizer
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from textblob import TextBlob
import re
import string
import emoji


# twitter scrape
def twint_search(search, username=None, since=None, until=None, drop_cols=None, limit=None):
    
    '''
    Function to return a Pandas DataFrame of tweets in English containing search terms using twint.
    
    Input a search term as a string. NOTE: use of 'OR', 'AND', and other search commands is supported.
    
    Optional inputs include:
                username: Twitter handle
                since: start date of search
                until: end date of search
                drop_cols: columns of tweet data to drop
                limit: maximum number of tweets to scrape
                
    Output a Pandas DataFrame.
    '''
    
    # instantiate twint object
    c = twint.Config()
    
    # set parameters
    c.Lang = 'en'
    c.Search = search
    c.Username = username
    c.Since = since
    c.Until = until
    c.Limit = limit
    c.Pandas = True
    
    # hide the printing of every tweet during scrape
    c.Hide_output = True
    
    # scrape
    twint.run.Search(c)
    
    # store as pandas dataframe
    df = twint.storage.panda.Tweets_df
    
    # transform date string into datetime object
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    return df


# looping twitter scrape and creating dataframe
def search_loop(start_date, end_date, search, filename, username=None, drop_cols=None, limit=None):
    
    '''
    Function to loop over date range and perform twint_search function for each day, returning one combined dataframe.
    Periodically saves progress to CSV after each daily search.
    
    Input a start date, end date, search time, and file name (where to save CSV file).
    
    Optional inputs include:
                username: Twitter handle
                drop_cols: columns of tweet data to drop
                limit: maximum number of tweets to scrape
                
    Output a Pandas DataFrame. Also saves as CSV.
    '''
    
    # instantiate empty dataframe
    df = pd.DataFrame()
    
    # create panda series of desired dates
    date_range = pd.Series(pd.date_range(start_date, end_date))
    
    # loop over dates
    for d in range(len(date_range) - 1):
        
        # obtain target date
        since = date_range[d].strftime('%Y-%m-%d')
        
        # obtain stop date
        until = date_range[d + 1].strftime('%Y-%m-%d')
        
        # scrape date to temporary dataframe
        day_df = twint_search(search=search, username=username, since=since, until=until, drop_cols=drop_cols, limit=limit)
        
        # drop empty columns
        day_df.drop(columns=drop_cols, axis=1, inplace=True)
        
        # add new daily data to dataframe, delete temporary dataframe, reset index, save to CSV
        df = pd.concat([df, day_df])
        del day_df
        df.reset_index(drop=True, inplace=True)
        df.to_csv(f'Datasets/{filename}.csv')
        
        # notification of successful scrape
        print(datetime.now(), f'{since} Saved!')
        
    return df


# emoticons
def load_dict_emoticons():
    
    '''
    Load a dictionary of emoticons as keys and their word equivalents as values.
    
    Source: https://towardsdatascience.com/twitter-sentiment-analysis-using-fasttext-9ccd04465597    
    '''
    
    return {
        ":‑)": "smiley",
        ":-]": "smiley",
        ":-3": "smiley",
        ":->": "smiley",
        "8-)": "smiley",
        ":-}": "smiley",
        ":)": "smiley",
        ":]": "smiley",
        ":3": "smiley",
        ":>": "smiley",
        "8)": "smiley",
        ":}": "smiley",
        ":o)": "smiley",
        ":c)": "smiley",
        ":^)": "smiley",
        "=]": "smiley",
        "=)": "smiley",
        ":-))": "smiley",
        ":‑D": "smiley",
        "8‑D": "smiley",
        "x‑D": "smiley",
        "X‑D": "smiley",
        ":D": "smiley",
        "8D": "smiley",
        "xD": "smiley",
        "XD": "smiley",
        ":‑(": "sad",
        ":‑c": "sad",
        ":‑<": "sad",
        ":‑[": "sad",
        ":(": "sad",
        ":c": "sad",
        ":<": "sad",
        ":[": "sad",
        ":-||": "sad",
        ">:[": "sad",
        ":{": "sad",
        ":@": "sad",
        ">:(": "sad",
        ":'‑(": "sad",
        ":'(": "sad",
        ":‑P": "playful",
        "X‑P": "playful",
        "x‑p": "playful",
        ":‑p": "playful",
        ":‑Þ": "playful",
        ":‑þ": "playful",
        ":‑b": "playful",
        ":P": "playful",
        "XP": "playful",
        "xp": "playful",
        ":p": "playful",
        ":Þ": "playful",
        ":þ": "playful",
        ":b": "playful",
        "<3": "love"
        }


# self defined contractions
def load_dict_contractions():
    '''
    Load a dictionary of contractions as keys and their expanded words as values.
    
    Source: https://towardsdatascience.com/twitter-sentiment-analysis-using-fasttext-9ccd04465597  
    '''
    
    return {
        "ain't": "is not",
        "amn't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "cuz": "because",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "could've": "could have",
        "daren't": "dare not",
        "daresn't": "dare not",
        "dasn't": "dare not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "d'you": "do you",
        "e'er": "ever",
        "em": "them",
        "'em": "them",
        "everyone's": "everyone is",
        "finna": "fixing to",
        "gimme": "give me",
        "gonna": "going to",
        "gon't": "go not",
        "gotta": "got to",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he he will have",
        "he's": "he is",
        "how'd": "how would",
        "how'll": "how will",
        "how're": "how are",
        "how's": "how is",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i'm'a": "i am about to",
        "i'm'o": "i am going to",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "i've": "i have",        
        "kinda": "kind of",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "may've": "may have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "might've": "might have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "must've": "must have",
        "needn't": "need not",
        "needn't've": "need not have",
        "ne'er": "never",
        "o'": "of",
        "o'clock": "of the clock",
        "o'er": "over",
        "ol'": "old",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",                
        "shalln't": "shall not",
        "shan't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "should've": "should have",
        "so's": "so as",
        "so've": "so have",
        "somebody's": "somebody is",
        "someone's": "someone is",
        "something's": "something is",
        "that'd": "that would",
        "that'd've": "that would have",
        "that'll": "that will",
        "that're": "that are",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there'll": "there will",
        "there're": "there are",
        "there's": "there is",
        "these're": "these are",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "this's": "this is",
        "those're": "those are",
        "to've": "to have",
        "'tis": "it is",
        "tis": "it is",
        "'twas": "it was",
        "twas": "it was",
        "wanna": "want to",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "weren't": "were not",
        "we've": "we have",
        "what'd": "what did",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where're": "where are",
        "where's": "where is",
        "where've": "where have",
        "which's": "which is",
        "will've": "will have",
        "who'd": "who would",
        "who'd've": "who would have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who're": "who are",
        "who's": "who is",
        "who've": "who have",
        "why'd": "why did",
        "why're": "why are",
        "why've": "why have",
        "why's": "why is",
        "won't": "will not",
        "won't've": "will not have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "would've": "would have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have",
        "Whatcha": "What are you",
        "luv": "love",
        "sux": "sucks"
        }


# apply text cleaning techniques
def clean_text(text, stop_words):
    
    '''
    Input text (a tweet) as a string and a list of stop words.
    
    Make text lowercase, remove mentions, remove links, convert emoticons and emojis to words, remove punctuation
    (except apostrophes), tokenize words (including contractions), convert contractions to full words,
    remove stop words.
    
    Output processed text as a string.
    '''
    
    # make text lowercase
    text = text.lower() 
    
    # remove mentions
    text = re.sub("(@[A-Za-z0-9]+)", "", text)
    
    # remove links
    text = re.sub(r'http\S+', '', text) 
    text = re.sub(r'pic\.\S+', '', text)
    
    # convert emoticons
    emoticons = load_dict_emoticons()
    words = text.split()
    words_edit = [emoticons[word] if word in emoticons else word for word in words]
    tweet = ' '.join(words_edit)

    # convert emojis
    text = emoji.demojize(text)
    text = text.replace(':', ' ') # separate emojis-words with space
    
    # remove punctuation
    text = text.replace('...', ' ') # special cases
    text = text.replace('-', ' ')
    text = text.translate(str.maketrans('', '', '!"$%&*()+,./;<=>?@[\\]^_`{|}~')) 
    
    # tokenize words
    tokenizer = RegexpTokenizer("(#?[a-zA-Z]+[0-9]*(?:'[a-zx]+)?)")
    words = tokenizer.tokenize(text)
    
    # convert contractions
    contractions = load_dict_contractions()
    words = text.split()
    words_edit = [contractions[word] if word in contractions else word for word in words]
    text = ' '.join(words_edit)

    # remove stop words and lemmatize
    lemmatizer = WordNetLemmatizer()
    words = tokenizer.tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    text = ' '.join(words)
    
    return text


def lda_getter(x):
    
    '''
    Input a list of tuples containing LDA topic weights.
    
    Convert to a dictionary and find the topic number with the highest weight.
    
    Output LDA topic number.
    '''
    
    # dictionary
    x_dict = dict(x)
    
    # obtain topic with highest weight and convert to integer
    topic = int(max(x_dict, key=x_dict.get))
    
    return topic


def mask_pos_finder(text, word):
    
    '''
    Input text as a string and a target word as a string.
    
    Find and return the part-of-speech (POS) tag for target word.
    
    Output POS tag as a string.
    '''
    
    # analyze text using TextBlob
    pos = TextBlob(text)
    
    # loop over list of POS tags
    for tag in pos.tags:
        
        # find and return POS tag
        if word in tag[0]:
            return tag[1]