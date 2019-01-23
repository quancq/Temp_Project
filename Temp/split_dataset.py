import utils
import os
import pandas as pd


if __name__ == "__main__":
    input_dir = "./Data/Product_Recognition/Original/Archive_2/AutoDaily"
    output_dir = "./Data/Product_Recognition/Splitted/AutoDaily"
    cat_dirs = utils.get_dir_names(input_dir)
    # print(len(cat_dirs))
    for cat_dir_name in cat_dirs:
        cat_dir_path = os.path.join(input_dir, cat_dir_name)
        num_docs = 0
        for ipath in utils.get_all_file_paths(cat_dir_path):
            # print(iname)
            df = utils.load_csv(ipath)
            for i, row in df.iterrows():
                title, intro, content = str(row["title"]), str(row["intro"]), str(row["content"])
                doc = title + ". " + intro + ". " + content
                doc = utils.tokenize(doc)
                num_docs += 1
                output_path = os.path.join(output_dir, cat_dir_name, "{}_{}.txt".format(cat_dir_name, num_docs))
                utils.save_str(doc, output_path)
            break

        # break
