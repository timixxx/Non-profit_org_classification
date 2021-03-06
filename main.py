import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
import matplotlib.pyplot as plt


# Loading data
df = pd.read_csv("Data/Train_dataset.csv")

df = df.rename(columns={"class": "label"})  # renaming due to python syntax error

columns = ['fullname', 'label']

df = df[columns]  # leave only needed columns

# Checking for classes imbalance
df.groupby('label').fullname.count().plot.bar()
ax = plt.subplot()
ax.set_ylabel('Number of names')
ax.set_title('Data class spread')
plt.show()


X = df.fullname
y = df.label

# Vectoring
stopwords_rus = nltk.corpus.stopwords.words('russian')
vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words=stopwords_rus, min_df=2, ngram_range=(1, 2))

train_corpus, test_corpus, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

X_train = vectorizer.fit_transform(train_corpus)
X_test = vectorizer.transform(test_corpus)

# Fitting models
model_NB = MultinomialNB()
model_NB.fit(X_train, y_train)

model_SVC = LinearSVC()
model_SVC.fit(X_train, y_train)

logit = LogisticRegression(random_state=0, max_iter=120)
logit.fit(X_train, y_train)

knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

gnb = GaussianNB()
gnb.fit(X_train.toarray(), y_train)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)


# Checking the accuracy of models
def accuracy_test(model, display):
    print("Accuracy of model", model, ":")
    if model == gnb:
        y_pred = model.predict_proba(X_test.toarray())
    elif model == model_SVC:
        y_pred = model._predict_proba_lr(X_test)
    else:
        y_pred = model.predict_proba(X_test)

    print(roc_auc_score(y_test, y_pred, multi_class='ovo'))

    if display:
        if model == gnb:
            ConfusionMatrixDisplay.from_estimator(model, X_test.toarray(), y_test, cmap='hot')
        else:
            ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='hot')

        plt.title(model)
        plt.show()


def input_test(model):
    print("Current model:", model, '\nEnter company name: ')
    text = [str(input())]
    pred = model.predict(vectorizer.transform(text))

    return pred


models = [model_NB, model_SVC, logit, knn, gnb, clf]


def accuracy_compare(models):
    roc_scores = []
    names = []
    for model in models:
        model_name = model.__class__.__name__
        names.append(model_name)

        if model == gnb:
            y_pred = model.predict_proba(X_test.toarray())
        elif model == model_SVC:
            y_pred = model._predict_proba_lr(X_test)
        else:
            y_pred = model.predict_proba(X_test)

        roc = roc_auc_score(y_test, y_pred, multi_class='ovo')
        roc_scores.append(roc)


    ax = plt.subplot()
    rect = ax.bar(names, roc_scores, width=0.35)
    ax.set_ylabel('ROC AUC Score')
    ax.set_title('Accuracy of models')
    ax.bar_label(rect, padding=3)
    plt.show()


for model in models:
    accuracy_test(model, False)

accuracy_compare(models)

# Visualization of the best model
accuracy_test(logit, True)

# Testing with manual input
print(input_test(logit))

