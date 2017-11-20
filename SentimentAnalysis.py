import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC

from KaggleWord2VecUtility import KaggleWord2VecUtility
from sklearn.pipeline import Pipeline
import pandas as pd
from collections import defaultdict
import numpy as np

if __name__ == '__main__':
    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'train.tsv'), header=0, \
                    delimiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'test.tsv'), header=0, delimiter="\t", \
                   quoting=3 )

    print 'The first review is:'
    print train["Phrase"][0]

    trainPhrases=[]

    currentSentenceId=0
    for i in xrange(0, len(train["Phrase"])):
        trainPhrases.append((train["Phrase"][i], train["Sentiment"][i]))

    train_sentence = [sentence for sentence, label in trainPhrases]
    train_labels = [int(label) for sentence, label in trainPhrases]

    classifier = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', RandomForestClassifier()),
                         ])
    X_train = np.asarray(train_sentence)
    classifier = classifier.fit(X_train, np.asarray(train_labels))




    testPhrases=[]
    testPhraseIds = []
    for i in xrange(0, len(test["Phrase"])):
        try:
            testPhrases.append((test["Phrase"][i]))
            testPhraseIds.append(test['PhraseId'][i])
        except KeyError as valerr:
            # except ValueError, valerr: # Deprecated?
            print valerr +" i= " + i

    predicted = classifier.predict(np.asarray(testPhrases))
    for i in range(len(predicted)):
        sentiment_label = predicted[i]
        phraseid = testPhraseIds[i]


    output = pd.DataFrame(data={"id": test["PhraseId"], "sentiment": predicted})

    # Use pandas to write the comma-separated output file
    output.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'trial_one.csv'), index=False, quoting=3)
    print "Wrote results to trial_one.csv"