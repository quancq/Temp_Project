import utils
import pandas as pd
import numpy as np
import time


class DatasetManager:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.load_data()

    def load_data(self):
        start_time = time.time()
        self.df = utils.load_csv(self.dataset_path, encoding="latin1")
        self.df.fillna(method="ffill", inplace=True)
        print(self.df.info())
        sentences = {}
        for i, row in self.df.iterrows():
            curr_sentence_id = int(row["Sentence #"].split(" ")[1])
            sentence = sentences.get(curr_sentence_id)
            if sentence is None:
                sentence = ([], [], [])
                sentences.update({curr_sentence_id: sentence})

            word, pos, tag = row["Word"], row["POS"], row["Tag"]
            sentence[0].append(word)
            sentence[1].append(pos)
            sentence[2].append(tag)

        sentence_len = [len(sent[0]) for sent in list(sentences.values())]
        min_len, max_len, avg_len = np.min(sentence_len), np.max(sentence_len), np.average(sentence_len)
        print("Min sentence length : ", min_len)
        print("Max sentence length : ", max_len)
        print("Avg sentence length : ", avg_len)

        self.sentences = sentences
        self.words = self.df["Word"].values.tolist()
        self.poses = self.df["POS"].values.tolist()
        self.tags = self.df["Tag"].values.tolist()
        print("Number sentences    : ", len(sentences))
        self.unique_words = list(set(self.words))
        print("Number unique WORDs : ", len(self.unique_words))
        self.unique_poses = list(set(self.poses))
        print("Number unique POSes : ", len(self.unique_poses))
        print(self.unique_poses)
        self.unique_tags = list(set(self.tags))
        print("Number unique TAGs  : ", len(self.unique_tags))
        print(self.unique_tags)

        save_json_path = "./Dataset/data.json"
        utils.save_json(self.sentences, save_json_path)

        exec_time = time.time() - start_time
        print("Load data done. Time {:.2f} seconds".format(exec_time))


if __name__ == "__main__":
    dataset_path = "./Dataset/entity-annotated-corpus/ner_dataset.csv"
    dm = DatasetManager(dataset_path=dataset_path)
