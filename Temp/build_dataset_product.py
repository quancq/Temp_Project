import utils
import os
import pandas as pd
import math


def get_synthetic_dataset(dir="./Data/Product_Recognition/Original/Archive_2"):
    pass
    file_paths = utils.get_all_file_paths(dir)

    print("Number files : ", len(file_paths))
    file_paths = [fpath for fpath in file_paths if "items" in fpath]
    print("Number files : ", len(file_paths))

    dataset = []
    columns = ["Domain", "Brand", "Category", "Model", "Info"]
    for fpath in file_paths:
        df = utils.load_csv(fpath)
        new_df = pd.DataFrame()
        exist_col = False
        for col in columns:
            if col in df.columns:
                new_df[col] = df[col].values.tolist()
                exist_col = True
            else:
                new_df[col] = ["" for _ in range(df.shape[0])]
        if exist_col:
            dataset.append(new_df)

    dataset = pd.concat(dataset, ignore_index=True)
    print(dataset.shape)

    file_size = 1000
    max_i = int(math.ceil((dataset.shape[0] / file_size)))
    for i in range(1, max_i+1):

        save_path = "./Data/Product_Recognition/Synthetic/{}/dataset_{}.csv".format(utils.get_time_str(), i)
        start_index = (i-1) * file_size
        end_index = start_index + file_size
        utils.save_csv(dataset[start_index: end_index], save_path)


def get_all_category(dir):
    file_paths = utils.get_all_file_paths(dir)
    categories = []
    data = []
    map_cat_domain = {}
    for fpath in file_paths:
        df = utils.load_csv(fpath)
        for i, row in df.iterrows():
            cat, domain = row["Category"], row["Domain"]
            map_cat_domain.update({cat: domain})

        categories.extend(df["Category"].values.tolist())

    categories = list(set(categories))
    data = [(map_cat_domain.get(cat), cat) for cat in categories]
    print(categories[:10])
    print("Number category : ", len(categories))

    df = pd.DataFrame(data, columns=["Domain", "Category"])
    utils.save_xlsx(df, save_path="./Data/Product_Recognition/Full_Category/{}.xlsx".format(utils.get_time_str()))


def get_map_cat_root():
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
        map_id_name.update({cat_id: "{} ({})".format(cat_name, cat_id)})
        parent_childs.append((parent_id, cat_id))
        map_child_parent.update({cat_id: parent_id})

        if parent_id == 0:
            root_cats.append(cat_id)
            map_root_pairs.update({cat_id: []})

    # Convert id to name
    # parent_childs = [(map_id_name.get(pid), map_id_name.get(cid)) for pid, cid in parent_childs if pid > 0]

    # print(parent_childs[:5])
    # print("Num cats: {}, Num root_cats: {}".format(df.shape[0], len(root_cats)))
    # print(len(map_child_parent))

    # Build map each root_cats to list tuple relationship of its childrens
    # print("Root id : ", map_root_pairs.keys())
    map_cat_root = {}
    for parent_id, child_id in parent_childs:
        # Find root cat id
        if parent_id != 0:
            root_id = child_id
            while root_id is not None:
                pid = map_child_parent.get(root_id)
                if pid is None or pid == 0:
                    break
                root_id = pid
            map_cat_root.update({int(child_id): int(root_id)})
            map_cat_root.update({int(parent_id): int(root_id)})
            pairs = map_root_pairs.get(root_id)
            pairs.append((parent_id, child_id))

    # print("\nmap_root_pairs : ", map_root_pairs.get(1))

    return map_cat_root


def build_standard_dataset():
    # Load org dataset
    dir = "./Data/Product_Recognition/Synthetic/2018-12-12 15:05:02"
    file_paths = utils.get_all_file_paths(dir)
    org_df = [utils.load_csv(fpath) for fpath in file_paths]
    org_df = pd.concat(org_df, ignore_index=True)

    # Load map from category name to category id
    path = "./Data/Product_Recognition/Full_Category/2018-12-12 15:22:27.xlsx"
    df = utils.load_xlsx(path)
    map_cat_name_id = {}
    for i, row in df.iterrows():
        cat_id = row["New Category Id"]
        if not math.isnan(cat_id):
            # print(type(cat_id), cat_id)
            map_cat_name_id.update({row["Category"]: int(cat_id)})

    print(map_cat_name_id)
    print(len(map_cat_name_id))

    # Load map from cat id to root cat id
    map_cat_root = get_map_cat_root()
    # print(map_cat_root)
    # print(len(map_cat_root))

    new_df = []
    cols = ["Domain", "Brand", "Category", "Model", "Info", "Category Id", "Root Id"]
    print("cols : ", cols)
    for i, row in org_df.iterrows():
        cat_id = map_cat_name_id.get(row["Category"])
        if cat_id is not None:
            root_id = map_cat_root.get(cat_id)
            new_row = (row["Domain"], row["Brand"], row["Category"],
                       row["Model"], row["Info"], cat_id, root_id)
            new_df.append(new_row)

    new_df = pd.DataFrame(new_df, columns=cols)
    print(new_df.head())
    print(new_df.shape)

    # Save new dataset
    file_size = 1000
    max_i = int(math.ceil((new_df.shape[0] / file_size)))
    for i in range(1, max_i+1):
        save_path = "./Data/Product_Recognition/Synthetic/Standard/{}/dataset_{}.csv".format(utils.get_time_str(), i)
        start_index = (i-1) * file_size
        end_index = start_index + file_size
        utils.save_csv(new_df[start_index: end_index], save_path)


if __name__ == "__main__":
    pass
    # get_synthetic_dataset()
    # get_all_category("./Data/Product_Recognition/Synthetic/2018-12-12 15:05:02")
    build_standard_dataset()
