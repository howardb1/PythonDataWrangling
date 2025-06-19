import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords 
from collections import Counter 
from nltk.util import ngrams 



nltk.download('punkt')
nltk.download('stopwords')


#reading in the csv file 
df = pd.read_csv("/Users/brianhoward/dev/PythonDataWrangling/example.csv", encoding='latin-1', header=None)
df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']



def clean_tweet(tweet):
    tweet = re.sub(r'http\S+|www\S+', '', tweet)   #Removes URLs
    tweet = re.sub(r'@\w+', '', tweet)             #Removes mentions
    tweet = re.sub(r'#\w+', '', tweet)             #Removes hashtags
    tweet = re.sub(r'[^A-Za-z\s]', '', tweet)      #removes special characters 
    tweet = tweet.lower()
    
    return tweet



stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


def tokenize_and_stem(tweet): 
    tokens = word_tokenize(tweet)
    filtered = [stemmer.stem(word) for word in tokens if word not in stop_words and len(word) > 2]
    return filtered 



#applies cleaning and tokenizing 
df['cleaned'] = df['text'].fillna('').astype(str)
df['tokens'] = df['cleaned'].apply(tokenize_and_stem)


#Flatten all tokens 
all_words = [word for tokens in df['tokens'] for word in tokens]
word_freq = Counter(all_words).most_common(20)



def extract_bigrams(tokens):
    return list(ngrams(tokens, 2))



df['bigrams'] = df['tokens'].apply(extract_bigrams)
all_bigrams = [bg for bigrams in df['bigrams'] for bg in bigrams]
bigrams_freq = Counter(all_bigrams).most_common (20)





#print ("mot Common words:")
#print(word_freq)


#print("\nMost Common Bigrams: ")
#print(bigrams_freq)


df.to_csv("/Users/brianhoward/dev/PythonDataWrangling/cleaned_example.csv", index=False)