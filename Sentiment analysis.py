# Sentiment-analysis-
# Phase 1 data extraction.

import pandas as pd
import tweepy
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
# Oauth keys
consumer_key = "c2rNqpUE2R4SBHPwL32Vsglw4"
consumer_secret = "AWOyMLny3DT4SJ63dBAlKdUuswfvEziBYwnsqIc5qkrI2Ml25B"

callback = 'oob' 
# Authentication with Twitter
auth = tweepy.OAuthHandler(consumer_key, consumer_secret, callback)
redirect_url = auth.get_authorization_url()

user_pin = input("Please enter user pin : ")
auth.get_access_token(user_pin)
api = tweepy.API(auth)

name ="JoeBiden"

#Extractacting replies related to the subject
replies=[]
for tweet in tweepy.Cursor(api.search,q='to:'+name, result_type='recent', timeout = 999999, lang='en', since="2020-11-07").items(2000):
    if hasattr(tweet, 'in_reply_to_status_id_str'):
        replies.append(tweet)
real_replies=[]
for tweet in replies:
    if(tweet.retweet_count == 0):
        real_replies.append(tweet)
         
df = pd.DataFrame([tweet.text for tweet in real_replies], columns=['Tweets'])

#Assigning labels with the help of Afinn lexicon
from afinn import Afinn
af = Afinn()
def sentimentanalysis(tweet):
    analyze = af.score(tweet)
    if (analyze == 0):
        return '1'
    elif (analyze < 0.00):
        return '-1'
    elif (analyze > 0.00):
        return '1'
df['labels'] = df['Tweets'].apply(sentimentanalysis)

df.to_csv('biden_real_replies.csv', index=False, columns=['labels','Tweets'], sep = '\t')

Dataset = pd.read_csv('biden_real_replies.csv', sep='\t')


#Data cleaning / tokenizing / lemmatization

import re 
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
wordnet = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

corpus = []
for i in range(0, len(Dataset)):
    review = re.sub('[^a-zA-Z]', ' ', Dataset['Tweets'][i])
    review = review.lower()
    review = review.split()
    
    review = [wordnet.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import TfidfVectorizer
model = TfidfVectorizer()
X = model.fit_transform(corpus).toarray()

y=pd.get_dummies(Dataset['labels'])
y=y.iloc[:,:-1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train , y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_m = confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)

print(confusion_m)
print(accuracy)
