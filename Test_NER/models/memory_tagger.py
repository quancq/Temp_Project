import utils
import pandas as pd
import numpy as np
from eda import DatasetManager
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_validate, cross_val_score
import time


class MemoryTagger(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        '''
        Find tag which most assigned to each word
        :param X: list of words
        :param y: list of tags
        :return:
        '''
        start_time = time.time()
        self.tags = []
        map_word_tag_count = {}
        for word, tag in zip(X, y):
            if tag not in self.tags:
                self.tags.append(tag)

            if word not in map_word_tag_count:
                map_word_tag_count[word] = {tag: 1}
            else:
                map_tag_count = map_word_tag_count.get(word)
                count = map_tag_count.get(tag, 0)
                map_tag_count.update({tag: count+1})

        self.map_word_best_tag = {}
        for word, map_tag_count in map_word_tag_count.items():
            self.map_word_best_tag[word] = max(map_tag_count, key=map_tag_count.get)

        exec_time = time.time() - start_time
        print("Memory Tagger fit done. Time : {:.2f} seconds".format(exec_time))

    def predict(self, X, y=None):
        return [self.map_word_best_tag.get(word, "O") for word in X]


if __name__ == "__main__":
    dataset_path = "../Dataset/entity-annotated-corpus/ner_dataset.csv"
    dm = DatasetManager(dataset_path=dataset_path)
    words, tags = dm.words, dm.tags

    # tagger = MemoryTagger()
    # tagger.fit(words, tags)
    model_path = "./Archive_Models/memory_tagger.model"
    # utils.save_sklearn_model(tagger, model_path)
    # tagger = utils.load_sklearn_model(model_path)

    test_words, test_poses, test_tags = dm.sentences.get(1)
    # print("Test_Words : ", test_words)
    # print("Test_Poses : ", test_poses)
    # print("Test_Tags : ", test_tags)
    #
    # pred_tags = tagger.predict(test_words)
    # print("Pred_Tags : ", pred_tags)
    #
    # test_df = pd.DataFrame(dict(Word=test_words, Pos=test_poses,
    #                             True_Tag=test_tags, Pred_Tag=pred_tags))
    # test_df = test_df[["Word", "Pos", "True_Tag", "Pred_Tag"]]
    # print(test_df)

    scores = cross_validate(MemoryTagger(), X=words, y=tags, scoring=["f1_macro"], cv=5, n_jobs=-1)
    print(scores["test_f1_macro"])
