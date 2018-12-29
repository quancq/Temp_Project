from gensim.models import Word2Vec, FastText
from gensim.models.keyedvectors import FastTextKeyedVectors, WordEmbeddingsKeyedVectors, KeyedVectors
import nmslib
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
    # dataset_path = "./Dataset/Preprocess/Train/dataset_2.csv"
    dataset_dir = "./Dataset/Preprocess/Train"
    dataset_paths = utils.get_file_paths(dataset_dir)
    df = utils.load_csvs(dataset_paths)
    return df["Root Id"].values, df["Model"].values


def load_test_df():
    dataset_path = "./Dataset/Preprocess/Test/dataset_17.csv"
    df = utils.load_csv(dataset_path)
    return df


def find_ann_word_vectors():
    wv_path = "./Model/FastText_vectors_39463.bin"
    word_vectors = FastTextKeyedVectors.load(wv_path)
    print(word_vectors)

    categories, product_names = load_products()
    test_df = load_test_df()[:100]
    test_docs = test_df["Info"].values.tolist()
    test_product_names = test_df["Model"].values.tolist()
    product_name_vectors = word_vectors[product_names]

    start_time = time.time()

    index = nmslib.init(space="cosinesimil", method="hnsw")
    index.addDataPointBatch(product_name_vectors)
    index.createIndex({"post": 2}, print_progress=True)

    result_names = []
    for i, doc in enumerate(test_docs):
        names_of_doc = []
        tokens = doc.split(" ")
        ngram = utils.get_ngram(tokens, ngram=5, step=7)
        print("\nNumber tokens : {}, Number grams : {}".format(len(tokens), len(ngram)))
        for gram in ngram:
            candidate_name = " ".join(gram)
            if candidate_name not in names_of_doc:
                # candidate_name = "Tủ_lạnh Hitachi 455L"
                # most_sim_product_name = word_vectors.most_similar_to_given(candidate_name, product_names)
                # sim = word_vectors.similarity(candidate_name, most_sim_product_name)

                candidate_vector = word_vectors[candidate_name]
                ids, distances = index.knnQuery(candidate_vector, k=10)
                max_id = np.argmin(distances)
                sim = 1 - distances[max_id]
                most_sim_product_name = product_names[ids[max_id]]

                if sim > 0.7:
                    names_of_doc.append(candidate_name)
                    print("Select Candidate_Name : {}, Product_Name : {} (id={})\n Similarity : {}\n".
                          format(candidate_name, most_sim_product_name, max_id, sim))
        result_names.append(",".join(names_of_doc))
        print("Extract {}/{} docs done".format(i+1, len(test_docs)))

    df = pd.DataFrame(dict(Documents=test_docs, True_Name=test_product_names, Pred_Name=result_names))
    df = df[["Documents", "True_Name", "Pred_Name"]]
    save_result_path = "./Result/test_find_ann.csv"
    utils.save_csv(df, save_result_path)

    exec_time = time.time() - start_time
    print("Time {:.2f} seconds".format(exec_time))


if __name__ == "__main__":
    pass
    # save_word_vectors()
    find_ann_word_vectors()
