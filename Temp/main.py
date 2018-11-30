import pandas as pd
import numpy as np
import math
import os
import utils
from collections import defaultdict


def load_cats():
    fpath = "./Data/Product_Recognition/productcat.xlsx"
    df = pd.read_excel(fpath)

    print(df.head())

    map_id_name = {}
    parent_childs = []
    root_cats = []
    map_child_parent = {}
    map_root_pairs = {}

    for i, row in df.iterrows():
        cat_id = row["cat_id"]
        cat_name = row["cat_name"]
        parent_id = row["parent_id"]
        map_id_name.update({cat_id: cat_name})
        parent_childs.append((parent_id, cat_id))
        map_child_parent.update({cat_id: parent_id})

        if parent_id == 0:
            root_cats.append(cat_id)
            map_root_pairs.update({cat_id: []})

    # Convert id to name
    # parent_childs = [(map_id_name.get(pid), map_id_name.get(cid)) for pid, cid in parent_childs if pid > 0]

    # print(parent_childs[:5])
    print("Num cats: {}, Num root_cats: {}".format(df.shape[0], len(root_cats)))
    print(len(map_child_parent))

    # Build map each root_cats to list tuple relationship of its childrens
    print("Root id : ", map_root_pairs.keys())
    for parent_id, child_id in parent_childs:
        # Find root cat id
        # if parent_id != 0:
        root_id = child_id
        while root_id is not None:
            pid = map_child_parent.get(root_id)
            if pid is None or pid == 0:
                break
            root_id = pid

        pairs = map_root_pairs.get(root_id)
        pairs.append((parent_id, child_id))

    print("\nmap_root_pairs : ", map_root_pairs.get(1))
    return map_root_pairs, map_id_name


def load_cat_id_domains():
    path1 = "./Data/Product_Recognition/Data.xlsx"
    path2 = "./Data/Product_Recognition/Data2.csv"

    data1 = utils.load_xlsx(path1)
    print(data1.head())

    data2 = utils.load_csv(path2)
    print(data2.head())

    data = pd.concat([data1, data2])
    print(data.head())

    map_id_domains = defaultdict(list)

    for i, row in data.iterrows():
        str_cat_id = row["cat_id"]
        domain = row["domain"]
        domain = domain.replace("http://", "")
        domain = domain.replace("https://", "")
        domain = domain.replace("www.", "")
        domain = domain.replace("/", "")

        if isinstance(domain, str) or not math.isnan(float(domain)):
            if isinstance(str_cat_id, str):
                cat_ids = str_cat_id.split(",")
                cat_ids = [int(cat_id.strip()) for cat_id in cat_ids if len(cat_id) > 0]

            if isinstance(str_cat_id, int):
                cat_ids = [str_cat_id]

            for cat_id in cat_ids:
                domains = map_id_domains.get(cat_id, [])
                domains.append(domain)
                map_id_domains.update({cat_id: domains})

    # print("\nMap id domain:")
    # for id, domains in map_id_domains.items():
    #     print("Id : {}, Domains : {}".format(id, domains))

    return map_id_domains


def main():
    data = []
    map_root_pairs, map_id_name = load_cats()
    map_id_domains = load_cat_id_domains()

    for root_id, pairs in map_root_pairs.items():
        root_name = map_id_name.get(root_id)
        print("\nRoot name : ", root_name)

        for parent_id, child_id in pairs:
            child_name = map_id_name.get(child_id)
            domains = map_id_domains.get(child_id, [])
            str_domains = ",".join(domains)

            data.append((child_id, parent_id, child_name, str_domains))

    cols = ["cat_id", "parent_id", "cat_name", "domains"]
    data = pd.DataFrame(data, columns=cols)
    print(data.head())

    save_path = "./Data/Product_Recognition/cat_domains.xlsx"
    utils.save_xlsx(data, save_path)


if __name__ == "__main__":
    pass
    main()
