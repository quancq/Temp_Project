import utils
import os
import pandas as pd


def get_synthetic_dataset(dir="./Data/Product_Recognition/Original"):
    pass
    file_paths = []
    for root, dirs, files in os.walk(dir):
        # print(files)
        files = [os.path.join(root, file) for file in files]
        file_paths.extend(files)

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

    save_path = "./Data/Product_Recognition/Synthetic/dataset_{}.csv".format(dataset.shape[0])
    utils.save_csv(dataset, save_path)


if __name__ == "__main__":
    get_synthetic_dataset()
