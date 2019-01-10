from collections import Counter

import eli5
import scipy

import utils
from eda import DatasetManager
import numpy as np
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_f1_score
from sklearn_crfsuite import scorers, metrics
from sklearn.model_selection import RandomizedSearchCV, cross_validate, cross_val_score, cross_val_predict
from sklearn.metrics import make_scorer, classification_report
import time


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]


def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))


def main():
    dm = DatasetManager(dataset_path="../Dataset/data.json", data_size=5000)
    sentences = list(dm.sentences.values())
    sent_len = len(sentences)
    train_size = int(sent_len * 0.8)
    print('Num Train size : ', train_size)
    print('Num Test size : ', (sent_len - train_size))
    train_sents = sentences[:train_size]
    test_sents = sentences[train_size:]
    X_train = [sent2features(sent) for sent in train_sents]
    y_train = [sent2labels(sent) for sent in train_sents]
    X_test = [sent2features(sent) for sent in test_sents]
    y_test = [sent2labels(sent) for sent in test_sents]

    # print("X_train size : ", len(X_train))
    # print("X_test size : ", len(X_test))

    start_time = time.time()
    # crf = CRF(algorithm='lbfgs',
    #           c1=0.1,
    #           c2=0.1,
    #           max_iterations=100,
    #           all_possible_transitions=True)
    # crf.fit(X_train, y_train)

    crf = CRF(
        algorithm='lbfgs',
        max_iterations=100,
        all_possible_transitions=True
    )
    labels = list(dm.tags)
    labels.remove('O')
    print("Number Tags: ", len(labels))
    params_space = {
        'c1': scipy.stats.expon(scale=0.5),
        'c2': scipy.stats.expon(scale=0.05),
    }

    # use the same metric for evaluation
    f1_scorer = make_scorer(metrics.flat_f1_score,
                            average='weighted', labels=labels)

    # search
    rs = RandomizedSearchCV(crf, params_space,
                            cv=2,
                            verbose=1,
                            n_jobs=-1,
                            n_iter=5,
                            scoring=f1_scorer,
                            random_state=7)
    rs.fit(X_train, y_train)
    rs_path = "./Archive_Models/rs_crf.model"
    utils.save_sklearn_model(rs, rs_path)
    rs = utils.load_sklearn_model(rs_path)
    print('best params:', rs.best_params_)
    print('best CV score:', rs.best_score_)
    print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))

    y_pred = rs.predict(X_test)
    macro_score = flat_f1_score(y_test, y_pred, average='macro', labels=labels)
    micro_score = flat_f1_score(y_test, y_pred, average='micro', labels=labels)
    weight_score = flat_f1_score(y_test, y_pred, average='weighted', labels=labels)
    print("F1_macro_score : ", macro_score)
    print("F1_micro_score : ", micro_score)
    print("F1_weight_score : ", weight_score)

    # scoring = ["f1_macro", "precision_macro", "recall_macro"]
    # scores = cross_validate(crf, X=X_train, y=y_train, scoring=scoring, cv=2, n_jobs=-1)
    # print("f1_macro        : ", np.mean(scores["test_f1_macro"]))
    # print("precision_marco : ", np.mean(scores["test_precision_macro"]))
    # print("recall_macro    : ", np.mean(scores["test_recall_macro"]))

    # group B and I results
    sorted_labels = sorted(
        labels,
        key=lambda name: (name[1:], name[0])
    )
    print(metrics.flat_classification_report(
        y_test, y_pred, labels=sorted_labels, digits=3
    ))

    print("Top likely transitions:")
    print_transitions(Counter(rs.best_estimator_.transition_features_).most_common(20))

    print("\nTop unlikely transitions:")
    print_transitions(Counter(rs.best_estimator_.transition_features_).most_common()[-20:])

    expl = eli5.explain_weights(rs.best_estimator_, top=10, targets=['O', 'B-per', 'I-per', 'B-geo', 'I-geo'])
    print(eli5.format_as_text(expl))

    exec_time = time.time() - start_time
    print("\nExec Time : {:.2f} seconds".format(exec_time))


if __name__ == "__main__":
    main()
