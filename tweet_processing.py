'''
Module docstring here

'''

import re
import string
import numpy as np

def prcs_rw_twt(raw_tweet):
    '''
    Processes raw tweet text and extracts additional features from tweet.
 	---
 	Input:
 	    raw_tweet: string, full text of twitter tweet
 	Returns:
 	    tweet: string, tweet with non-word elements removed
 	    features: list of additional feature values including...

        rt: binary, if tweet is a retweet
        mntns: binary, if tweet mentions others
        n_mntns: int, number of mentions in tweet
        link: binary, if text includes link
        emoji: binary, if text includes emoji
        n_emoji: int, number of emojis in tweet text
    '''
    # Ensure common case for comparison
    tweet = raw_tweet.lower()

    # Build regular expressions for Twitter entities in text
    retweet = re.compile(r'\srt\s') 
    mention = re.compile(r'@[\w_]+') # alphanumeric and underscores
    link = re.compile(r'http[s]*:\/\/[^(t.co)]')
    emoji = re.compile(r'&#12\d+;') # example: &#128514;
    grammar = re.compile(r'&#\d+;') # example: &#8217; for apostrophe
    grammar2 = re.compile(r'&\w+;') # example: &amp; for &
    alphanumeric = re.compile(r'\w*\d\w*')
    punctuation = re.compile(r'[{}]+'.format(re.escape(string.punctuation)))
    wht_spc = re.compile(r'\s\s+') # remove excess whitespace

    to_search = [retweet, mention, link, emoji]
    to_clean = to_search + [grammar2, alphanumeric, punctuation, wht_spc]
    new_features = []

    # Exctract new regex-based features
    for regex in to_search:
    	new_features.append(len(re.findall(regex, tweet)))
    
    # Scrub excess text
    for regex in to_clean:
    	tweet = re.sub(regex,' ', tweet)

    # Get mean word length in characters and avg word length
    n_char = len(re.sub(r'\s',' ', tweet))
    avg_wrd = np.around(
    	np.mean([len(word) for word in tweet.split()]), 2)
    if np.isnan(avg_wrd): #account for taking mean of empty list
    	avg_wrd = 0
    new_features = new_features+[n_char,avg_wrd]

    return tweet.strip(), new_features

def basic_tweet_scrub(tweet_series):
    '''
    Accepts a series of tweets and returns series w/ string objects
    in lowercase with punctuation and usernames removed.
    '''
    alphanumeric = lambda x: re.sub('\w*\d\w*', ' ', x)
    punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower())
    usernames = lambda x: re.sub('@[\w-]+', ' ', x.lower())
    return tweet_series.map(alphanumeric).map(usernames).map(punc_lower)
    


