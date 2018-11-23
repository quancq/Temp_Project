import pyvi
from pyvi import ViPosTagger, ViTokenizer
import utils
from collections import Counter


if __name__ == "__main__":
    file_path = "./Dataset/sendo_iphone"
    doc = utils.read_file(file_path)

    # print(doc)

    tokens = ViTokenizer.tokenize(doc)
    # print(tokens)

    tok_counts = Counter(tokens.split())
    print(tok_counts)

    # tokens, pos_tags = ViPosTagger.postagging(tokens)
    # for token, pos in zip(tokens, pos_tags):
    #     print("{} - {}".format(token, pos))
