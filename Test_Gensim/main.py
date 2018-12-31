from gensim.models import Word2Vec, FastText
from gensim.models.wrappers.fasttext import FastText as FT_Wrapper
import fastText
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
    train_docs = train_df["Info"].values
    train_docs = [doc.split(" ") for doc in train_docs]

    print(train_docs[:2])
    print("Training {} docs with model {} ...".format(len(train_docs), model_name))
    start_time = time.time()

    if model_name == "Word2Vec":
        model = Word2Vec(sentences=train_docs, size=100, window=5, min_count=3, workers=4, iter=20)
        # model.train(sentences=train_docs, total_examples=len(train_docs), epochs=100)
    elif model_name == "FastText":
        model = FastText(sentences=train_docs, size=100, window=5, min_count=3, workers=4, iter=20)
        # model.train(sentences=train_docs, total_examples=len(train_docs), epochs=100)
        # model = FT_Wrapper.train(ft_path="/home/quancq/Program/fastText-0.1.0/fasttext",
        #                        corpus_file="./Dataset/Preprocess/Train/dataset_1.csv")

        # model = fastText.train_unsupervised(input="./Dataset/Preprocess/Train/dataset_1.csv")
    else:
        print("Model name {} is not valid".format(model_name))
        return 0

    exec_time = time.time() - start_time
    print("\nTrain {} docs with model {} done. Time : {:.2f} seconds".format(len(train_docs), model_name, exec_time))
    print(model)

    # summarize vocabulary
    words = list(model.wv.vocab)
    print(words[:50])
    print("Vocab size : ", len(words))

    # save model
    save_path = "./Model/{}/model_{}_{}_{}.bin".format(model_name, model_name, len(train_docs), model.iter)
    model.save(save_path)

    word_vectors = model.wv
    wv_path = "./Model/{}/{}_vectors_{}_{}.bin".format(model_name, model_name, len(word_vectors.vocab), model.iter)
    word_vectors.save(wv_path)

    # load model
    new_model = Word2Vec.load(save_path)
    print(new_model)

    words = ["Galaxy Note 9", "Dell Vostro X240", "Đồng hồ nam cá tính", "Honda XS99",
             "Máy giặt tự động Sam sung", "Xe máy sành điệu Honda X2AB",
             "Cà phê Trung Nguyên đóng hộp", "Sữa tươi khuyến mại Mộc Châu"]
    for word in words:
        print("\nWord: ", word)
        try:
            print(new_model.most_similar(positive=word))
        except:
            print("No similar word found")


if __name__ == "__main__":
    # model_name = "FastText"
    model_name = "Word2Vec"
    train_word2vec(model_name=model_name)

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

