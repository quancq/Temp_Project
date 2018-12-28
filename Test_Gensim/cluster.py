from gensim.models import Word2Vec, FastText
from gensim.models.keyedvectors import FastTextKeyedVectors, WordEmbeddingsKeyedVectors, KeyedVectors
import utils
import os, time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier


def save_word_vectors():
    we_model_path = "./Model/model_FastText_16000.bin"
    we_model = FastText.load(we_model_path)
    word_vectors = we_model.wv
    wv_path = "./Model/FastText_vectors_{}.bin".format(len(word_vectors.vocab))
    word_vectors.save(wv_path)


def load_products():
    dataset_path = "./Dataset/Preprocess/Train/dataset_2.csv"
    df = utils.load_csv(dataset_path)
    return df["Root Id"].values, df["Model"].values


def cluster_word_vectors():
    start_time = time.time()
    wv_path = "./Model/FastText_vectors_39463.bin"
    word_vectors = FastTextKeyedVectors.load(wv_path)
    print(word_vectors)

    categories, product_names = load_products()
    X_train = word_vectors[product_names]
    y_train = categories
    print("X_train shape: {}, y_train shape: {}".format(X_train.shape, y_train.shape))

    classifier = KNeighborsClassifier(random_state=7, n_jobs=-1)
    pred = classifier.fit_predict(X_train)

    df = pd.DataFrame(dict(Category=categories, Product_Name=product_names))
    df = df[["Category", "Product_Name"]]
    df.sort_values("Category", inplace=True)
    save_result_path = "./Result/test_classifier.csv"
    utils.save_csv(df, save_result_path)

    cluster_model_path = "./Model/Cluster/test_classifier.pkl"
    utils.save_sklearn_model(classifier, cluster_model_path)

    exec_time = time.time() - start_time
    print("Time {:.2f} seconds".format(exec_time))


if __name__ == "__main__":
    pass
    # save_word_vectors()
    cluster_word_vectors()
