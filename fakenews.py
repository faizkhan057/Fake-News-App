import pandas as pd

#load datasets
true_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")


#add labels
true_df["label"] = 1 #True news
fake_df["label"] = 0 #Fake news

#combine datasets
data = pd.concat([true_df,fake_df],axis=0)

#shuffle data
data = data.sample(frac=1).reset_index(drop=True)

print(data.head())
print(data["label"].value_counts())


#title+text combine 
data["content"] = data["title"] + "" + data["text"]

X = data["content"]
y = data["label"]



#train-test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#TF-IDF Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english',max_df=0.7)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


#model training (logistic regression)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf,y_train)

#save the model
import pickle

pickle.dump(model,open("model.pkl","wb"))
pickle.dump(vectorizer,open("vectorizer.pkl","wb"))

#prediction and evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = model.predict(X_test_tfidf)

print("Accuracy:",accuracy_score(y_test,y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test,y_pred))

print("\nClassification report:")
print(classification_report(y_test,y_pred))

#imp words
import numpy as np
feature_names = vectorizer.get_feature_names_out()


#model ka weight
coefficients = model.coef_[0]

#top 20 real
top_real_indices = np.argsort(coefficients)[-20:]
top_real_words = feature_names[top_real_indices]

print("top words for real news")
for word in reversed(top_real_indices):
    print(feature_names[word])

#top 20 fake
top_fake_indices = np.argsort(coefficients)[:20]
top_fake_words = feature_names[top_fake_indices]

print("\ntop words for fake news")
for word in top_fake_indices:
    print(feature_names[word])


#custom news test
while True:

    news = input("Enter News Text or exit: ")
    if news.lower()=="exit":
        break

    news_vector = vectorizer.transform([news])
    prediction = model.predict(news_vector)

    if prediction[0]==1:
        print("Real News")
    else:
        print("Fake News")