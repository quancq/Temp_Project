import utils
import pandas as pd
from pyvi import ViTokenizer


def text2csv(text):
    tokens = ViTokenizer.tokenize(text).split(" ")
    df = pd.DataFrame(tokens, columns=["Word"])

    return df


if __name__ == "__main__":
    path = "./Data/NER_input/1"
    text = utils.load_str(path)

    df = text2csv(text)
    save_path = "./Data/NER_output/1.csv"
    utils.save_csv(df, save_path)

    print(df.head())
