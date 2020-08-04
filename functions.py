# dataframe libraries
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import itertools
import emoji
import string
import re
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk import RegexpTokenizer
import nltk
import numpy as np
import pandas as pd

# twitter scraper
import twint

# fixes runtime errors with twint
import nest_asyncio
nest_asyncio.apply()


# twitter scrape
def twint_search(
        search,
        username=None,
        since=None,
        until=None,
        drop_cols=None,
        limit=None):
    
    '''
    Function to scrape Twitter using Twint (`https://github.com/twintproject/twint`).
    Returns a Pandas DataFrame of tweets (in English) that contain search terms
    entered by the user.

    Input
    -----
    search : str
        Desired search term(s).
        Use of 'OR', 'AND', and other search commands is supported.

    Optional input
    --------------
    username : str
        Twitter handle.
        `@` symbol not required.
    since : str
        Start date of search.
        format ==> Y%%%-M%-D%, e.g. `2020-02-29`
    until : str
        End date of search (not inclusive).
        format ==> Y%%%-M%-D%, e.g. `2020-03-01`
    drop_cols : list (str)
        Tweet attributes to drop.
        For a full list of attributes, visit:
        `https://github.com/twintproject/twint/wiki/Tweet-attributes`
    limit : int
        Maximum number of tweets to scrape.

    Output
    ------
    df : Pandas DataFrame
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
def search_loop(
        start_date,
        end_date,
        search,
        filename,
        username=None,
        drop_cols=None,
        limit=None):
    
    '''
    Function to loop over date range and perform twint_search function for
    each day, returning one combined dataframe.

    Periodically saves progress to CSV after each daily search.

    Input
    -----
    start_date : str
        Start date of search.
        format ==> Y%%%-M%-D%, e.g. `2020-02-29`
    end_date : str
        End date of search (not inclusive).
        format ==> Y%%%-M%-D%, e.g. `2020-03-01`
    search : str
        Desired search term(s).
        Use of 'OR', 'AND', and other search commands is supported.
    filename : str
        Path and name for saved csv file.

    Optional inputs
    ---------------
    username : str
        Twitter handle.
        `@` symbol not required.
    drop_cols : list (str)
        Tweet attributes to drop.
        For a full list of attributes, visit:
        `https://github.com/twintproject/twint/wiki/Tweet-attributes`
    limit : int
        Maximum number of tweets to scrape.

    Output
    ------
    df : Pandas DataFrame
        Also saves and updates a CSV file after each day in the search.
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
        day_df = twint_search(
            search=search,
            username=username,
            since=since,
            until=until,
            drop_cols=drop_cols,
            limit=limit)

        # drop empty columns
        day_df.drop(columns=drop_cols, axis=1, inplace=True)

        # add new daily data to dataframe, delete temporary dataframe, reset
        # index, save to CSV
        df = pd.concat([df, day_df])
        del day_df
        df.reset_index(drop=True, inplace=True)
        df.to_csv(f'Datasets/{filename}.csv')

        # notification of successful scrape
        print(datetime.now(), f'{since} Saved!')

    return df


# convert emoticons
def load_dict_emoticons():
    
    '''
    Load a dictionary of emoticons as keys and their word equivalents
    as values.

    Source
    ------
    https://towardsdatascience.com/twitter-sentiment-analysis-using-fasttext-9ccd04465597
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


# convert contractions
def load_dict_contractions():
    
    '''
    Load a dictionary of contractions as keys and their expanded words
    as values.

    Source (modified)
    ------
    https://towardsdatascience.com/twitter-sentiment-analysis-using-fasttext-9ccd04465597
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
        "he'll've": "he will have",
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
        "sux": "sucks",
        "abt": "about",
        "aint": "is not",
        "amnt": "am not",
        "arent": "are not",
        "cant": "cannot",
        "cantve": "cannot have",
        "cuz": "because",
        "couldnt": "could not",
        "couldntve": "could not have",
        "couldve": "could have",
        "darent": "dare not",
        "didnt": "did not",
        "doesnt": "does not",
        "dont": "do not",
        "dyou": "do you",
        "eer": "ever",
        "everyones": "everyone is",
        "gon": "going to",
        "hadnt": "had not",
        "hadntve": "had not have",
        "hasnt": "has not",
        "havent": "have not",
        "hed": "he would",
        "hedve": "he would have",
        "hellve": "he will have",
        "hes": "he is",
        "howd": "how would",
        "howll": "how will",
        "howre": "how are",
        "hows": "how is",
        "id": "i would",
        "idve": "i would have",
        "illve": "i will have",
        "im": "i am",
        "ima": "i am about to",
        "isnt": "is not",
        "itd": "it would",
        "itdve": "it would have",
        "itll": "it will",
        "itllve": "it will have",
        "ive": "i have",
        "lets": "let us",
        "maam": "madam",
        "maynt": "may not",
        "mayve": "may have",
        "mightnt": "might not",
        "mightntve": "might not have",
        "mightve": "might have",
        "mustnt": "must not",
        "mustntve": "must not have",
        "mustve": "must have",
        "neednt": "need not",
        "needntve": "need not have",
        "neer": "never",
        "oclock": "of the clock",
        "oer": "over",
        "ol": "old",
        "oughtnt": "ought not",
        "oughtntve": "ought not have",
        "shallnt": "shall not",
        "shant": "shall not",
        "shantve": "shall not have",
        "shedve": "she would have",
        "shellve": "she will have",
        "shes": "she is",
        "shouldnt": "should not",
        "shouldntve": "should not have",
        "shouldve": "should have",
        "sove": "so have",
        "somebodys": "somebody is",
        "someones": "someone is",
        "somethings": "something is",
        "thatd": "that would",
        "thatdve": "that would have",
        "thatll": "that will",
        "thatre": "that are",
        "thats": "that is",
        "thered": "there would",
        "theredve": "there would have",
        "therell": "there will",
        "therere": "there are",
        "theres": "there is",
        "thesere": "these are",
        "theyd": "they would",
        "theydve": "they would have",
        "theyll": "they will",
        "theyllve": "they will have",
        "theyre": "they are",
        "theyve": "they have",
        "thiss": "this is",
        "thosere": "those are",
        "tove": "to have",
        "wasnt": "was not",
        "wedve": "we would have",
        "wellve": "we will have",
        "werent": "were not",
        "weve": "we have",
        "whatd": "what did",
        "whatll": "what will",
        "whatre": "what are",
        "whats": "what is",
        "whatve": "what have",
        "whens": "when is",
        "whenve": "when have",
        "whered": "where did",
        "wherere": "where are",
        "wheres": "where is",
        "whereve": "where have",
        "willve": "will have",
        "whod": "who would",
        "whodve": "who would have",
        "wholl": "who will",
        "whollve": "who will have",
        "whos": "who is",
        "whove": "who have",
        "whyd": "why did",
        "whyre": "why are",
        "whyve": "why have",
        "whys": "why is",
        "wont": "will not",
        "wontve": "will not have",
        "wouldnt": "would not",
        "wouldntve": "would not have",
        "wouldve": "would have",
        "yall": "you all",
        "yalld": "you all would",
        "yallre": "you all are",
        "yallve": "you all have",
        "youd": "you would",
        "youdve": "you would have",
        "youll": "you will",
        "youllve": "you will have",
        "youre": "you are",
        "youve": "you have",
        "yr": "your"
    }


# apply text cleaning techniques
def clean_text(text, stop_words):
    
    '''
    Function to make tweets lowercase, remove mentions, remove links,
    convert emoticons and emojis to words, remove punctuation (except
    apostrophes), tokenize words (including contractions), convert
    contractions to full words, and remove stop words.

    Input
    -----
    text : str
        Text to be cleaned.
    stop_words : list (str)
        Words to remove from text.

    Output
    ------
    text : str
        Processed text.
    '''

    # make text lowercase
    text = text.lower()

    # remove mentions
    text = re.sub('(@[A-Za-z0-9]+)', '', text)

    # remove links
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'pic\.\S+', '', text)

    # convert emoticons
    emoticons = load_dict_emoticons()
    words = text.split()
    words_edit = [emoticons[word]
                  if word in emoticons else word for word in words]
    tweet = ' '.join(words_edit)

    # convert emojis
    text = emoji.demojize(text)
    text = text.replace(':', ' ')  # separate emojis-words with space

    # remove punctuation
    text = text.replace('...', ' ')  # special cases
    text = text.replace('-', ' ')   # separate words with space
    text = text.translate(
        str.maketrans(
            '',
            '',
            '!"$%&*()+,./;<=>?@[\\]^_`{|}~'))

    # tokenize words -- includes hashtags, words with numbers, and contractions
    tokenizer = RegexpTokenizer("(#?[a-zA-Z]+[0-9]*(?:'[a-z]+)?)")
    words = tokenizer.tokenize(text)

    # convert contractions
    contractions = load_dict_contractions()
    words = text.split()
    words_edit = [contractions[word]
                  if word in contractions else word for word in words]
    text = ' '.join(words_edit)

    # lemmatize, remove stop words, and remove words with fewer than two
    # characters
    lemmatizer = WordNetLemmatizer()
    words = tokenizer.tokenize(text)
    words = [lemmatizer.lemmatize(word)
             for word in words if word not in stop_words]
    words = [word for word in words if len(word) > 2]
    text = ' '.join(words)

    return text


# topic identifier
def lda_getter(x):
    
    '''
    Function to find the LDA topic number with the highest weight.

    Input
    -----
    x : list (tuple)
        LDA topic weights.

    Output
    ------
    topic : int
        LDA topic number.
    '''

    # convert to dictionary
    x_dict = dict(x)

    # obtain topic with highest weight and convert to integer
    topic = int(max(x_dict, key=x_dict.get))

    return topic


# find POS tags
def mask_pos_finder(text, word):
    
    '''
    Function to find the part-of-speech (POS) tag for a target word.

    Input
    -----
    text : str
        Text to be analyzed.
    word : str
        Target word in the text.

    Output
    ------
    tag[1] : str
        POS tag.
    '''

    # analyze text using TextBlob
    pos = TextBlob(text)

    # loop over list of POS tags
    for tag in pos.tags:

        # find and return POS tag
        if word in tag[0]:
            return tag[1]

        
# confusion matrix plotter
def plot_confusion_matrix(
        cm,
        classes,
        normalize=False,
        title='Confusion matrix',
        cmap=plt.cm.Blues):
    
    '''
    This function prints and plots a model's confusion matrix.

    Input
    -----
    cm : sklearn confusion matrix
        `sklearn.metrics.confusion_matrix(y_true, y_pred)`
    classes : list (str)
        Names of target classes.

    Optional input
    --------------
    normalize : bool
        Whether to apply normalization (default=False).
        Normalization can be applied by setting `normalize=True`.
    title : str
        Title of the returned plot.
    cmap : matplotlib color map
        For options, visit:
        `https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html`

    Output
    ------
    Prints a stylized confusion matrix.


    [Code modified from work by Sean Abu Wilson.]
    '''

    # convert to percentage, if normalize set to True
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # plot
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # format true positives and others
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=15,
                 horizontalalignment="center", verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # add axes labels
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# decision tree feature importances plotter
def plot_tree_features(
        model,
        df,
        num_features=10,
        to_print=True,
        to_save=False,
        file_name=None):
    
    '''
    This function plots feature importances for Decision Tree models
    and optionally prints a list of tuples with features and their
    measure of importance.

    Input
    -----
    model : Decision Tree model
        `sklearn.tree.DecisionTreeClassifier()`
    df : Pandas DataFrame
        Features used in model.

    Optional input
    --------------
    num_features : int
        The number of features to plot/print (default=10).
        All feature importances can be shown by setting
        `num_features=X.shape[1]`.
    to_print : bool
        Whether to print list of feature names and their impurity
        decrease values (default=True).
        Printing can be turned off by setting `to_print=False`.
    to_save : bool
        Whether to save graph (default=False).
        A file can be saved off by setting `to_save=True` and
        setting a value for `file_name`.
    file_name : str
        Path and name to save a graph (default=None).
        Required if `to_save=True`.

    Output
    ------
    Prints a bar graph and optional list of tuples.
    '''

    features_dict = dict(zip(df.columns, model.feature_importances_))
    sorted_d = sorted(
        features_dict.items(),
        key=lambda x: x[1],
        reverse=True)[
        :num_features]

    # top 10 most important features
    tree_importance = [x[1] for x in sorted_d]

    # prettify the graph
    plt.figure(figsize=(12, 8))
    plt.title('Decision Tree Feature Importances', fontsize=25, pad=15)
    plt.xlabel('')
    plt.ylabel('Gini Importance', fontsize=22, labelpad=15)
    plt.ylim(bottom=sorted_d[-1][1]/1.75, top=sorted_d[0][1]*1.05)
    plt.xticks(rotation=60, fontsize=20)
    plt.yticks(fontsize=20)

    # plot
    plt.bar([x[0] for x in sorted_d], tree_importance)

    # prepare to display
    plt.tight_layout()

    if to_save:
        # save plot
        plt.savefig(file_name, bbox_inches='tight', transparent=True)

    # show plot
    plt.show()

    if to_print:
        # print a list of feature names and their impurity decrease value in
        # the decision tree
        print('\n\n\n')
        print(sorted_d)
