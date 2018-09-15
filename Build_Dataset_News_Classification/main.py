import numpy as np
import pandas as pd
import os, json
from datetime import datetime


LABEL_ID_MAP = {
    "Viễn thông": 2,
    "Du lịch": 6,
    "Giao thông": 6,
    "Giáo dục": 7,
    "Ẩm thực": 9,
    "Y tế": 10,
    "Sức khỏe": 10,
    "Làm đẹp": 11,
    "Thời trang": 12,
    "Bất động sản": 13,
    "Nội thất": 14,
    "Nhà": 14,
    "Xe": 16,
    "Tài chính": 17,
    "BHXH và cuộc sống": 17,
    "Pháp luật": 19,
    "Mẹ và bé": 24,
    "Thể thao": 188

}
DATE_TIME_FMT = "%d-%m-%Y_%H-%M-%S"


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)

    return data


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f)
    print("Save json data (size = {}) to {} done".format(len(data), path))


def load_csv(path):
    data = pd.read_csv(path)
    data["time"] = pd.to_datetime(data["time"], format=DATE_TIME_FMT)
    data.sort_values("time", ascending=False, inplace=True)

    return data


def get_category_id(label):
    cate_id = LABEL_ID_MAP.get(label)
    if cate_id is None:
        print("Label : {} dont have category id".format(label))
    return cate_id


def filter_data(data):
    data.dropna(subset=["time"], inplace=True)
    data.fillna("", inplace=True)
    data = data[:200]
    # t = datetime(year=2018, month=8, day=20)
    # data = data[data["time"] > t]
    return data


def transform_data(data):
    transformed_data = []
    for index, row in data.iterrows():
        category = row["category"]
        title = row["title"]
        intro = row["intro"]
        content = row["content"]
        # print("Title = {}, Intro = {}, Content = {}".format(title, intro, content))
        content = '. '.join([title, intro, content])
        transformed_data.append({
            "label": get_category_id(category),
            "content": content
        })

        # break

    return transformed_data


def get_files(dir):
    paths = os.listdir(dir)
    return [os.path.join(dir, path) for path in paths]


def merge_data(merge_dir="./Data/Merge_Data"):
    file_paths = get_files(merge_dir)
    data = []
    for path in file_paths:
        data.extend(load_json(path))

    print(len(data))
    save_json(data, os.path.join(merge_dir, "new_data_train_{}.json".format(len(data))))


def build_new_data():
    dir = "./Data/Origin"
    paths = get_files(dir)
    print("Number file inputs : ", len(paths))
    training_data = []
    test_data = []

    for path in paths:
        data = load_csv(path)
        data = filter_data(data)
        data_size = int(len(data) / 2)
        # print(data.head())
        training_data.extend(transform_data(data[:data_size]))
        test_data.extend(transform_data(data[data_size:]))

    # Save data
    train_save_path = "./Data/new_data_train_{}.json".format(len(training_data))
    save_json(training_data, train_save_path)

    test_save_path = "./Data/new_data_test_{}.json".format(len(test_data))
    save_json(test_data, test_save_path)

    # data = pd.DataFrame.from_dict(transformed_data)
    # data["label"].astype("int")
    # print(data["label"].value_counts())


if __name__ == "__main__":
    # file_path = "./Data/Merge_Data/new_data_train_5986.json"
    # data = load_json(file_path)
    # # data = pd.DataFrame(data)
    # new_data = []
    # remove_label = [18, 20, 22]
    # for elm in data:
    #     if elm["label"] not in remove_label:
    #         new_data.append(elm)
    #
    # save_json(new_data, "./Data/Merge_Data/new_data_train_{}.json".format(len(new_data)))

    path = "./Data/Merge_Data/data_train_4386.json"
    data = load_json(path)
    data = pd.DataFrame(data)
    data = data[data["label"] == 15]
    print(data.head(10).values)
    print("\n")
    # print(data.tail(1).values)