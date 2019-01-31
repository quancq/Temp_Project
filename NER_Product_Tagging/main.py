import utils
import pandas as pd
import os, json, time


def old_main():
    start_time = time.time()
    input_dir = "./Data/input"
    fnames = utils.get_file_names(input_dir)

    for fname_id, fname in enumerate(fnames):
        fpath = os.path.join(input_dir, fname)
        df = utils.load_csv(fpath)
        new_df = []
        for i, row in df.iterrows():
            product_name = row["Model"]
            cat = row["Category"]
            brand = "" if utils.isnan(row["Brand"]) else str(row["Brand"])
            doc = row["Info"]

            ner_cats, ner_brands, ner_models, ner_info = [], [], [], []

            tokens = utils.tokenize(product_name)
            is_start_brand = False
            is_end_brand = False
            for token_id, token in enumerate(tokens):
                replaced_token = token.replace("_", " ")
                if len(brand) > 0:
                    if replaced_token.lower() in brand.lower() or brand.lower() in replaced_token.lower():
                        is_start_brand = True

                if not is_start_brand:
                    ner_cats.append(token)
                else:
                    if replaced_token.lower() not in brand.lower() and brand.lower() not in replaced_token.lower():
                        is_end_brand = True

                    if not is_end_brand:
                        ner_brands.append(token)
                    else:
                        ner_info.append(token)

            new_df.append((cat, product_name, brand, ner_cats, ner_brands, ner_models, ner_info, doc))

            if i > 100:
                break

        columns = ["Category", "Full Product Name", "Brand", "Ner_Cat", "Ner_Bra", "Ner_Mod", "Ner_Inf", "Doc"]
        new_df = pd.DataFrame(new_df, columns=columns)
        save_path = os.path.join("./Data/output/", fname)
        utils.save_csv(new_df, save_path)
        print("Process {}/{} files done".format(fname_id + 1, len(fnames)))

    exec_time = time.time() - start_time
    print("Time : {:.2f} seconds".format(exec_time))


def main():
    start_time = time.time()
    input_dir = "./Data/input"
    fnames = utils.get_file_names(input_dir)

    num_products = 0
    doc_by_lines = []
    tag_result_by_lines = []
    num_output_files = 0

    for fname_id, fname in enumerate(fnames):
        fpath = os.path.join(input_dir, fname)
        df = utils.load_csv(fpath)

        df = df.loc[df["Root Id"] == 16]

        for i, row in df.iterrows():
            num_products += 1
            product_name = row["Model"]
            cat = row["Category"]
            brand = "" if utils.isnan(row["Brand"]) else str(row["Brand"])
            doc = row["Info"]

            # ner_cats, ner_brands, ner_models, ner_info = [], [], [], []

            tokens = utils.tokenize(doc)
            tokens = [token + "," for token in tokens]
            doc_by_lines.append("\n<Product id='{}'>".format(num_products))
            doc_by_lines.extend(tokens)
            doc_by_lines.append("</Product>")

            tag_result_by_lines.append("\n<Product id='{}' full_name='{}'>".format(num_products, product_name))
            tag_result_by_lines.append("<Brand>")
            tag_result_by_lines.append(brand)
            tag_result_by_lines.append("</Brand>")

            tag_result_by_lines.append("<Category>")
            tag_result_by_lines.extend(utils.tokenize(cat))
            tag_result_by_lines.append("</Category>")

            tag_result_by_lines.append("<Model>")
            tag_result_by_lines.append("</Model>")

            tag_result_by_lines.append("<Info>")
            tag_result_by_lines.append("</Info>")

            tag_result_by_lines.append("</Product>")

            if (num_products % 100) == 0:
                num_output_files += 1
                save_path = "./Data/output/Doc_by_lines/doc_{}".format(num_output_files)
                utils.save_list(doc_by_lines, save_path)
                save_path = "./Data/output/Tag_Result/doc_{}".format(num_output_files)
                utils.save_list(tag_result_by_lines, save_path)

                doc_by_lines, tag_result_by_lines = [], []
                break

        print("Process {}/{} files done".format(fname_id + 1, len(fnames)))

    exec_time = time.time() - start_time
    print("Time : {:.2f} seconds".format(exec_time))


if __name__ == "__main__":
    main()
