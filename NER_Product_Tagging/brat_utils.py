import utils
import pandas as pd
import os
import time
from pyvi import ViTokenizer


def load_ann_file(path):
    cols = ["TagID", "NerTag", "StartPos", "EndPos", "Text"]
    data = utils.load_list(path)
    df = []
    for line in data:
        split_line = line.split("\t")
        new_line = []
        new_line.append(split_line[0])
        new_line.extend(split_line[1].split())
        new_line.append(split_line[-1])

        df.append(new_line)

    # print(df[0])
    df = pd.DataFrame(df, columns=cols)
    # print(df)
    return df


def convert_news_csv_to_txt(csv_path, dst_dir):
    start_time = time.time()

    df = utils.load_csv(csv_path)
    docs = df["content"].dropna()
    csv_name = os.path.basename(csv_path)
    csv_name = csv_name[:csv_name.rfind(".")]
    num_files = 0
    for i, doc in enumerate(docs):
        if len(doc) == 0:
            continue
        if (i + 1) % 100 == 0:
            print("Process {}/{} files ...".format(i+1, len(docs)))

        file_name = "{}_{}.txt".format(csv_name, i+1)
        save_path = os.path.join(dst_dir, file_name)

        tokens = ViTokenizer.tokenize(doc)
        utils.save_str(tokens, save_path)
        num_files += 1

    exec_time = time.time() - start_time
    print("Convert {} files done. Time {:.2f} seconds".format(num_files, exec_time))


if __name__ == "__main__":
    # path = "./Data/brat/xe/xe_1.ann"
    # df = load_ann_file(path)
    #
    # txt_path = "./Data/brat/xe/xe_1.txt"
    # doc = utils.load_str(txt_path)
    #
    # pos2tag = {}
    # for i, row in df.iterrows():
    #     start_pos, end_pos = int(row["StartPos"]), int(row["EndPos"])
    #     for pos in range(start_pos, end_pos):
    #         pos2tag[pos] = row["NerTag"]
    #
    # start_pos = 0
    # tokens = doc.split()
    # tags = []
    # for token in tokens:
    #     ner_tag = pos2tag.get(start_pos, "O")
    #     tags.append(ner_tag)
    #     start_pos += len(token) + 1
    #
    # result = []
    # for token, tag in zip(tokens, tags):
    #     result.append((token, tag))
    # print(result)

    csv_path = "./Data/brat/input/xe.csv"
    dst_dir = "./Data/brat/output/xe"
    convert_news_csv_to_txt(csv_path, dst_dir)
