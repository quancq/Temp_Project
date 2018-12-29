import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import math
from sklearn.externals import joblib

DEFAULT_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def get_time_str(time=datetime.now(), fmt=DEFAULT_TIME_FORMAT):
    try:
        return time.strftime(fmt)
    except:
        return ""


def get_time_obj(time_str, fmt=DEFAULT_TIME_FORMAT):
    try:
        return datetime.strptime(time_str, fmt)
    except:
        return None


def transform_time_fmt(time_str, src_fmt, dst_fmt=DEFAULT_TIME_FORMAT):
    time_obj = get_time_obj(time_str, src_fmt)
    time_str = get_time_str(time_obj, dst_fmt)
    return time_str


def mkdirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def make_parent_dirs(path):
    dir = path[:path.rfind("/")]
    mkdirs(dir)


def get_file_paths(parent_dir):
    file_paths = []
    for root, dirs, files in os.walk(parent_dir):
        # print(files)
        files = [os.path.join(root, file) for file in files]
        file_paths.extend(files)

    return file_paths


def get_file_names(parent_dir):
    file_names = [file_name for file_name in os.listdir(parent_dir)
                  if os.path.isfile(os.path.join(parent_dir, file_name))]
    return file_names


def save_list(lst, save_path):
    if len(lst) == 0:
        return
    make_parent_dirs(save_path)

    with open(save_path, "w") as f:
        f.write("\n".join(lst))

    print("Save data (size = {}) to {} done".format(len(lst), save_path))


def save_csv(df, save_path):
    if df.shape[0] == 0:
        return
    make_parent_dirs(save_path)

    df.to_csv(save_path, index=False)
    print("Save data (size = {}) to {} done".format(df.shape[0], save_path))


def save_xlsx(df, save_path):
    if df.shape[0] == 0:
        return
    make_parent_dirs(save_path)
    df.to_excel(save_path, index=False)
    print("Save data (size = {}) to {} done".format(df.shape[0], save_path))


def save_json(data, save_path, mode="w"):
    if len(data) == 0:
        return

    make_parent_dirs(save_path)

    if mode == "a":
        data.update(load_json(save_path))

    with open(save_path, 'w') as f:
        json.dump(data, f, default=MyEncoder)

    print("Save json data (size = {}) to {} done".format(len(data), save_path))


def load_csv(path):
    data = None
    try:
        data = pd.read_csv(path)
        print("Read csv data (size = {}) from {} done".format(data.shape[0], path))
    except:
        print("Error when load csv data from ", path)
    return data


def load_csvs(paths):
    # dataset_paths = get_file_paths(dir)
    df = []
    for path in paths:
        # df.append(utils.load_csv("./Dataset/Data1/dataset_3.csv"))
        df.append(load_csv(path))

    df = pd.concat(df, ignore_index=True, sort=False)
    print("Load {} files (size={}) csv done".format(len(paths), df.shape[0]))
    return df


def load_xlsx(path):
    data = pd.read_excel(path)
    print("Read data (size = {}) from {} done".format(data.shape[0], path))
    return data


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)

    print("Load json data (size = {}) from {} done".format(len(data), path))
    return data


def load_list(path):
    data = []
    with open(path, 'r') as f:
        data = f.read().strip().split("\n")

    print("Load list data (size = {}) from {} done".format(len(data), path))
    return data


def sort_coo(m, axis=0):
    if axis != 0 and axis != 1:
        print("Utils::Sort_coo axis = {} -> Not valid".format(axis))
        axis = 0

    tuples = zip(m.row, m.col, m.data)
    return sorted(tuples, key=lambda x: (x[axis], x[2]), reverse=True)


def save_sklearn_model(model, save_path):
    make_parent_dirs(save_path)
    joblib.dump(model, save_path)
    print("Save sklearn model to {} done".format(save_path))


def load_sklearn_model(path):
    try:
        return joblib.load(path)
    except:
        print("Error when load sklearn model from ", path)
        return None


def get_ngram(tokens, ngram=3, min_ngram=1, max_ngram=5, step=2):
    if ngram is not None and ngram > 0:
        result = [tokens[i: i+ngram] for i in range(0, len(tokens) - ngram + 1, step)]
    else:
        result = [tokens[i: i+ngram] for ngram in range(min_ngram, max_ngram+1)
                  for i in range(0, len(tokens) - ngram + 1, step)]

    return result


if __name__ == "__main__":
    pass
