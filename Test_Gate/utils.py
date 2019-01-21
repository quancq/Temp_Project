import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import math
from sklearn.externals import joblib
from lxml import etree as ET

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


def get_all_file_paths(dir):
    file_paths = []
    for root, dirs, files in os.walk(dir):
        # print(files)
        files = [os.path.join(root, file) for file in files]
        file_paths.extend(files)

    return file_paths


def get_file_paths(parent_dir):
    file_paths = [os.path.join(parent_dir, file) for file in os.listdir(parent_dir)]
    file_paths = [file_path for file_path in file_paths if os.path.isfile(file_path)]
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


def load_csv(path, **kwargs):
    data = None
    try:
        data = pd.read_csv(path, **kwargs)
        print("Read csv data (size = {}) from {} done".format(data.shape[0], path))
    except:
        print("Error when load csv data from ", path)
    return data


def load_csvs(paths):
    df = []
    for path in paths:
        # df.append(utils.load_csv("./Dataset/Data1/dataset_3.csv"))
        df.append(load_csv(path))

    df = pd.concat(df, ignore_index=True, sort=False)
    print("Load {} files (size={}) csv from {} done".format(len(paths), df.shape[0], dir))
    return df


def load_xlsx(path):
    data = None
    try:
        data = pd.read_excel(path)
        print("Read data (size = {}) from {} done".format(data.shape[0], path))
    except:
        print("Error when load csv data from ", path)
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


def load_str(path):
    data = ""
    with open(path, 'r') as f:
        data = f.read().strip()

    return data


def sort_coo(m, axis=0):
    if axis != 0 and axis != 1:
        print("Utils::Sort_coo axis = {} -> Not valid".format(axis))
        axis = 0

    tuples = zip(m.row, m.col, m.data)
    return sorted(tuples, key=lambda x: (x[axis], x[2]), reverse=True)


def convert_df_to_dict(df, col_key=None, col_value=None):
    if col_key is None or col_key not in list(df.columns):
        return {}

    if col_value is None:
        return df.set_index(col_key).to_dict("index")
    else:
        result = {k: v for k, v in zip(df[col_key].values.tolist(), df[col_value].values.tolist())}
        return result


def get_ngram(tokens, ngram=None, min_ngram=1, max_ngram=5, step=2):
    if ngram is not None and ngram > 0:
        result = [(i, tokens[i: i+ngram]) for i in range(0, len(tokens) - ngram + 1, step)]
    else:
        result = [(i, tokens[i: i+ngram]) for ngram in range(min_ngram, max_ngram+1)
                  for i in range(0, len(tokens) - ngram + 1, step)]

    return result


def is_separate(start1, end1, start2, end2, max_overlap_rate=0.5):
    # if start2 < start1:
    #     overlap = min(end1, end2) - start1 + 1
    # elif start2 < end1:
    #     overlap = min(end1, end2) - start2 + 1
    # else:
    #     overlap = 0
    overlap = min(end1, end2) - max(start1, start2) + 1
    overlap = max(0, overlap)

    overlap_rate = overlap / (end1 - start1 + end2 - start2 + 2 - overlap)
    return overlap_rate <= max_overlap_rate


def save_sklearn_model(model, save_path):
    make_parent_dirs(save_path)
    joblib.dump(model, save_path)
    print("Save sklearn model to {} done".format(save_path))


def load_sklearn_model(model_path):
    model = joblib.load(model_path)
    return model


def load_xml(path):
    try:
        result = ET.parse(path)
    except Exception as e:
        print("Error when load xml from ", path)
        print(e)
        result = None

    return result


if __name__ == "__main__":
    pass
