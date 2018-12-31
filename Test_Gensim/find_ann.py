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
    we_model_path = "./Model/FastText/model_FastText_16000.bin"
    we_model = FastText.load(we_model_path)
    word_vectors = we_model.wv
    wv_path = "./Model/FastText/FastText_vectors_{}.bin".format(len(word_vectors.vocab))
    word_vectors.save(wv_path)


def load_products():
    # dataset_path = "./Dataset/Preprocess/Train/dataset_2.csv"
    dataset_dir = "./Dataset/Preprocess/Train"
    dataset_paths = utils.get_file_paths(dataset_dir)
    df = utils.load_csvs(dataset_paths)
    return df["Root Id"].values, df["Model"].values


def load_test_df():
    dataset_path = "./Dataset/Preprocess/Test/dataset_17_hard.csv"
    df = utils.load_csv(dataset_path)
    return df


def find_ann_word_vectors():
    model_path = "./Model/FastText/model_FastText_16000_20.bin"
    model = FastText.load(model_path)


    # wv_path = "./Model/FastText/FastText_vectors_39463_20.bin"
    # word_vectors = FastTextKeyedVectors.load(wv_path)
    # print(word_vectors)

    categories, product_names = load_products()
    test_df = load_test_df()[:100]
    test_docs = test_df["Info"].values.tolist()

    tokenized_docs = [doc.split(" ") for doc in test_docs]
    model.build_vocab(tokenized_docs, update=True)
    model.train(tokenized_docs, total_examples=len(test_docs), epochs=10)

    test_product_names = test_df["Model"].values.tolist()
    word_vectors = model.wv
    product_name_vectors = word_vectors[product_names]

    start_time = time.time()

    index = nmslib.init(space="cosinesimil", method="hnsw")
    index.addDataPointBatch(product_name_vectors)
    index.createIndex({"post": 2}, print_progress=True)

    result_names = []
    for doc_idx, doc in enumerate(test_docs):
        names_of_doc = []
        tokens = doc.split(" ")
        ngram = utils.get_ngram(tokens, min_ngram=3, max_ngram=6, step=1)
        print("\nNumber tokens : {}, Number grams : {}".format(len(tokens), len(ngram)))
        for gram_idx, gram in ngram:
            if "." in gram:
                continue
            candidate_name = " ".join(gram)
            if candidate_name not in names_of_doc:
                # candidate_name = "Tủ_lạnh Hitachi 455L"
                # most_sim_product_name = word_vectors.most_similar_to_given(candidate_name, product_names)
                # sim = word_vectors.similarity(candidate_name, most_sim_product_name)

                try:
                    candidate_vector = word_vectors[candidate_name]
                    ids, distances = index.knnQuery(candidate_vector, k=10)
                    max_id = np.argmin(distances)
                    sim = 1 - distances[max_id]
                    most_sim_product_name = product_names[ids[max_id]]

                    if sim > 0.9:
                        names_of_doc.append((gram_idx, candidate_name, sim))
                        # print("Select Candidate_Name : {}, Product_Name : {} (id={})\n Similarity : {}\n".
                        #       format(candidate_name, most_sim_product_name, max_id, sim))
                except:
                    pass
        # Select best names in candidate names
        best_names_of_doc = []
        top_names = len(names_of_doc)
        for _ in range(top_names):
            best_idx = -1
            max_score = 0
            best_name = None
            for candidate_idx, candidate_name, score in names_of_doc:
                if score > max_score:
                    is_valid = True
                    for sel_idx, _, selected_name in best_names_of_doc:
                        start1, end1 = sel_idx, sel_idx + len(selected_name.split(" "))
                        start2, end2 = candidate_idx, candidate_idx + len(candidate_name.split(" "))
                        if not utils.is_separate(start1, end1, start2, end2):
                            is_valid = False
                            break
                    if is_valid:
                        best_idx = candidate_idx
                        max_score = score
                        best_name = candidate_name

            if best_name is not None:
                best_names_of_doc.append((best_idx, max_score, best_name))
            else:
                break

        result_names_of_doc = []
        for idx, score, name in best_names_of_doc:
            result_names_of_doc.append(name)
            print("Select Candidate_Name : {} - Similarity : {}\n".format(name, score))
        result_names.append(",".join(result_names_of_doc))

        print("Extract {}/{} docs done".format(doc_idx+1, len(test_docs)))

    df = pd.DataFrame(dict(Documents=test_docs, True_Name=test_product_names, Pred_Name=result_names))
    df = df[["Documents", "True_Name", "Pred_Name"]]
    save_result_path = "./Result/test_find_ann_hard.csv"
    utils.save_csv(df, save_result_path)

    exec_time = time.time() - start_time
    print("Time {:.2f} seconds".format(exec_time))


if __name__ == "__main__":
    pass
    # save_word_vectors()
    find_ann_word_vectors()
