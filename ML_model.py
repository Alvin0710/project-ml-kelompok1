#Import library
import pandas as pd
import numpy as np
import pickle


df = pd.read_csv('Dataset.csv')

# Case folding
import re

def caseFolding(Text):
    Text = re.sub(r'http\S+', '', Text) #Remove link
    Text = re.sub(r'www\S+', '', Text) #Remove link
    Text = re.sub("#\S*\s", "", Text) #Remove hashtag
    Text = re.sub("@\S*\s", "", Text) #Remove username
    Text = Text.lower() #Make it lowercase
    Text = Text.strip(" ") #Remove too much space
    Text = re.sub('[^a-zA-Z]',' ',Text) #Remove number
    Text = re.sub(r'[?|$|.|!_:/\'")(---+,#]', "", Text) #Remove character
    return Text

df['Text'] = df['Text'].apply(caseFolding)

# Delete duplicates
data_clean = df.drop_duplicates()

# Tokenizing
import contractions
data_clean['Text'] = data_clean['Text'].apply(lambda x: [contractions.fix(word) for word in x.split()])
data_clean

#Filtering
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

def stopword_removal(comments):
  filtering = stopwords.words('indonesian')
  x = []
  tweet_text = []
  def myFunc(x):
    if x in filtering:
      return False
    else:
      return True
  
  fit = filter(myFunc, comments)
  for x in fit:
    tweet_text.append(x)
  return tweet_text

data_clean['Text'] = data_clean['Text'].apply(stopword_removal)


#Stemming
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

def stemming(comments):
  factory = StemmerFactory()
  stemmer = factory.create_stemmer()
  do = []

  for w in comments:
    dt = stemmer.stem(w)
    do.append(dt)

  d_clean = []
  d_clean = " ".join(do)
  print(d_clean)
  return d_clean

data_clean['Text'] = data_clean['Text'].apply(stemming)


# Label Encoder
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
data_clean['Label'] = labelencoder.fit_transform(data_clean['Label'])
data_clean


#TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer()
text_tf = tf.fit_transform(data_clean['Text'].astype('U'))


pickle.dump(tf, open('tfidf.pkl', 'wb'))

X = text_tf
y = data_clean['Label']


#SMOTE
from imblearn.over_sampling import SMOTE

smote = SMOTE()
X_smote, y_smote = smote.fit_resample(X, y)


# Split into train and validation data
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_smote, y_smote, test_size=.2


from sklearn.naive_bayes import MultinomialNB

mnb_model = MultinomialNB().fit(X_train, y_train)


pickle.dump(mnb_model, open('mnb.pkl', 'wb'))