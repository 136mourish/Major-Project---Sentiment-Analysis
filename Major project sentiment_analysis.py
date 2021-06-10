from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
vs=SentimentIntensityAnalyzer()
#webscraping--beatidul soup 4

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import os
urls=['https://inshorts.com/en/read/world',
              'https://inshorts.com/en/read/sports',
              'https://inshorts.com/en/read/politics',
      'https://inshorts.com/en/read/technology',
      'https://inshorts.com/en/read/entertainment','https://inshorts.com/en/read/automobile']

def build_dataset(urls):
  news_data=[]
  news_category=[]
  for url in urls:
    news_category=url.split('/')[-1]
    data=requests.get(url)
    soup= BeautifulSoup(data.content)
    
    news_articles=[{'news_headline':headline.find('span',attrs={"itemprop":"headline"}).string,
                    'news_article':article.find('div',attrs={"itemprop":"articleBody"}).string,
                    'news_category':news_category}
                   
                   for headline,article in zip(soup.find_all('div',class_=["news-card-title news-right-box"]),
                                               soup.find_all('div',class_=["news-card-content news-right-box"]))
                   ]
    news_articles=news_articles[0:20]
    news_data.extend(news_articles)

  df=pd.DataFrame(news_data)
  df=df[['news_headline','news_article','news_category']]
  return df
df= build_dataset(urls)
import nltk
nltk.download('stopwords')

stopwords_list=nltk.corpus.stopwords.words('english')
stopwords_list.remove('no')
stopwords_list.remove('not')

#remove html tags

def html_tag(text):
  soup=BeautifulSoup(text,"html.parser")
  new_text=soup.get_text()
  return new_text

!pip install contractions

import contractions
def con(text):
  expand=contractions.fix(text)
  return(expand)

import re
def remove_sp(text):
  pattern=r'[^A-Za-z0-9\s]'
  text= re.sub(pattern,'',text)
  return text

from nltk.tokenize.toktok import ToktokTokenizer
tokenizer=ToktokTokenizer
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')


def remove_stopwords(text):
  tokens=word_tokenize(text)
  tokens=[token.strip() for token in tokens]
  filtered_token=[token for token in tokens if token not in stopwords_list]
  filtered_text=' '.join(filtered_token)
  return filtered_text
 
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
  
# stem words in the list of tokenised words
def stem_lemm(text):
    word_tokens = word_tokenize(text)
    #stems = [stemmer.stem(word) for word in word_tokens ]
    #print(stems)
    lem=[lemmatizer.lemmatize(i,pos='v') for i in word_tokens]
    #print(lem)
    t=' '.join(lem)
    return t

#PREPROCESSING  
#lowercase
#html tag
#contraction
#special char
#stopwords

df.news_headline =df.news_headline.apply(lambda x:x.lower())
df.news_article =df.news_article.apply(lambda x:x.lower())

df.news_headline =df.news_headline.apply(html_tag)
df.news_article =df.news_article.apply(html_tag)

df.news_headline =df.news_headline.apply(con)
df.news_article =df.news_article.apply(con)

df.news_headline =df.news_headline.apply(remove_sp)
df.news_article =df.news_article.apply(remove_sp)

df.news_headline =df.news_headline.apply(remove_stopwords)
df.news_article =df.news_article.apply(remove_stopwords)

#df.news_headline =df.news_headline.apply(stem_lemm)
df.news_article =df.news_article.apply(stem_lemm)


df.head()

df['compound']=df['news_article'].apply(lambda x: vs.polarity_scores(x)['compound'])
df['Review']=df['compound'].apply(lambda x: "positive" if x > 0.0 else "negative" ,['review'])

df

df.to_csv('data.csv',index=False)

#SPLIT AND TRAINING

from sklearn.model_selection import train_test_split
X = df.news_article
Y = df.Review
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2,random_state=100)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
model = Pipeline([('vectorizer',TfidfVectorizer()),('classifier',LogisticRegression())])
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
from sklearn.metrics import accuracy_score
print("Accuracy : ", accuracy_score(y_pred, y_test))
from sklearn.metrics import confusion_matrix
confusion_matrix(y_pred, y_test)
from sklearn.metrics import classification_report
print(classification_report(y_pred, y_test))


example = ["not good"]
result = model.predict(example)

print(result)



import joblib
joblib.dump(model,'sentiment')


!pip install streamlit --quiet
!pip install pyngrok==4.1.1 --quiet
from pyngrok import ngrok

%%writefile app.py
import streamlit as st
import joblib
model = joblib.load('sentiment')
st.title('Sentiment Analysis')
ip = st.text_input("Enter the review")
op = model.predict([ip])
if st.button('Predict'):
  st.title(op[0])
!nohup streamlit run app.py &
url= ngrok.connect(port='8501')
print(url)






