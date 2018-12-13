import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
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


def save_csv(df, save_path):
    make_parent_dirs(save_path)

    df.to_csv(save_path, index=False)
    print("Save data (size = {}) to {} done".format(df.shape[0], save_path))


def save_xlsx(df, save_path):
    make_parent_dirs(save_path)
    df.to_excel(save_path, index=False)
    print("Save data (size = {}) to {} done".format(df.shape[0], save_path))


def save_json(data, save_path, mode="w"):
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


def load_xlsx(path):
    data = pd.read_excel(path)
    print("Read data (size = {}) from {} done".format(data.shape[0], path))
    return data


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)

    print("Load json data (size = {}) from {} done".format(len(data), path))
    return data