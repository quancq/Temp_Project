from gensim.models import Word2Vec, FastText
import utils
import pandas as pd
from preprocess import pre_process_pipeline
import time
import os


def train_word2vec(model_name="Word2Vec"):
    # stopwords = utils.load_list("./Dataset/vi_stopwords.txt")
    train_dir = "./Dataset/Preprocess/Train"

    fpaths = utils.get_file_paths(train_dir)
    train_df = utils.load_csvs(fpaths)
    train_docs = train_df["Info"]

    # print(train_docs)
    print("Training {} docs ...".format(len(train_docs)))
    start_time = time.time()

    if model_name == "Word2Vec":
        model = Word2Vec(sentences=train_docs, size=100, window=5, min_count=3, workers=4)
    elif model_name == "FastText":
        model = FastText(sentences=train_docs, size=100, window=5, min_count=3, workers=4)
    else:
        print("Model name {} is not valid".format(model_name))
        return 0

    model.train(sentences=train_docs, total_examples=len(train_docs), epochs=100)

    exec_time = time.time() - start_time
    print("\nTrain {} docs done. Time : {:.2f} seconds".format(len(train_docs), exec_time))
    print(model)

    # summarize vocabulary
    words = list(model.wv.vocab)
    print(words[:50])
    print("Vocab size : ", len(words))

    # save model
    save_path = "./Model/model_{}.bin".format(len(train_docs))
    model.save(save_path)
    # load model
    new_model = Word2Vec.load(save_path)
    print(new_model)


if __name__ == "__main__":
    train_word2vec()

    # # print(train_docs)
    # print("Training {} docs ...".format(len(train_docs)))
    # start_time = time.time()
    # model = Word2Vec(sentences=train_docs, size=100, window=5, min_count=3, workers=3)
    # model.train(sentences=train_docs, total_examples=len(train_docs), epochs=100)
    # exec_time = time.time() - start_time
    # print("\nTrain {} docs done. Time : {:.2f} seconds".format(len(train_docs), exec_time))
    # print(model)
    # # summarize vocabulary
    # words = list(model.wv.vocab)
    # print(words[:50])
    # print("Vocab size : ", len(words))
    # # access vector for one word
    # # print(model['thực_phẩm'])
    # # save model
    # save_path = "./Model/model_16000.bin"
    # model.save(save_path)
    # # load model
    # model = Word2Vec.load(save_path)
    # print(model)
    #
    # r = model.wv.most_similar(positive="Nokia")
    # print(r)

