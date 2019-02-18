import utils
import pandas as pd
import os, json, time
import math
import re
from pyvi import ViPosTagger


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


def get_candidate_tokens(product_name, tokens):
    p = re.compile("[(),?!.-]+")
    # candidate_tokens = [t for t in tokens if t.replace("_", " ") in product_name and not p.search(t)]
    candidate_ids = []
    candidate_tokens = []

    for id, t in enumerate(tokens):
        if t.replace("_", " ").lower() in product_name.lower() and not p.search(t) and id not in candidate_ids:
            candidate_tokens.append(t)
            candidate_ids.append(id)

    return candidate_tokens, candidate_ids


def generate_json_full_doc():
    input_dir = "./Data/input"
    num_products = 0

    df = utils.load_csvs_in_dir(input_dir)
    root_ids = list(set(df["Root Id"].values.tolist()))
    print("Root Ids : ", root_ids)

    for root_id in root_ids:
        data = []
        sub_df = df[df["Root Id"] == root_id]
        sub_df = sub_df.sort_values("Model")
        step = int(math.ceil(sub_df.shape[0] / 500))
        sub_df = sub_df[::step].reset_index()
        print("Process Root Id : {} - Number products : {}".format(root_id, sub_df.shape[0]))
        for i, row in sub_df.iterrows():
            num_products += 1
            product_name = row["Model"]
            product_name = " ".join(utils.tokenize(product_name)).replace("_", " ")
            cat = row["Category"]
            brand = "" if utils.isnan(row["Brand"]) else str(row["Brand"])
            doc = row["Info"]

            tokens = utils.tokenize(doc)

            data.append(dict(product_id=num_products, brand=brand, doc=" ".join(tokens),
                             full_name=product_name, category=cat))

        save_path = "./Data/output/Full_Doc_Json/{}".format(root_id)
        utils.save_json(data, save_path)


def main():
    start_time = time.time()
    input_dir = "./Data/input"
    # fnames = utils.get_file_names(input_dir)

    num_products = 0
    full_doc = []
    tag_result_by_lines = []
    # num_output_files = 0

    df = utils.load_csvs_in_dir(input_dir)
    root_ids = list(set(df["Root Id"].values.tolist()))
    print("Root Ids : ", root_ids)

    # for fname_id, fname in enumerate(fnames):
    #     fpath = os.path.join(input_dir, fname)
    #     df = utils.load_csv(fpath)
    #
    #     df = df.loc[df["Root Id"] == 16]

    for root_id in root_ids:
        sub_df = df[df["Root Id"] == root_id]
        sub_df = sub_df.sort_values("Model")
        step = int(math.ceil(sub_df.shape[0] / 500))
        sub_df = sub_df[::step].reset_index()
        print("Process Root Id : {} - Number products : {}".format(root_id, sub_df.shape[0]))
        full_doc, tag_result_by_lines = ["<Products>"], ["<Products>"]
        for i, row in sub_df.iterrows():
            num_products += 1
            product_name = row["Model"]
            product_name = " ".join(utils.tokenize(product_name)).replace("_", " ")
            cat = row["Category"]
            brand = "" if utils.isnan(row["Brand"]) else str(row["Brand"])
            doc = row["Info"]

            # ner_cats, ner_brands, ner_models, ner_info = [], [], [], []

            tokens = utils.tokenize(doc)
            # tokens = [token + "," for token in tokens]
            full_doc.append("\n<Product id='{}'>".format(num_products))
            full_doc.append("<Full_Name><![CDATA[{}]]></Full_Name>".format(product_name))
            full_doc.append("<Doc><![CDATA[{}]]></Doc>".format(" ".join(tokens)))
            full_doc.append("</Product>")

            tag_result_by_lines.append("\n<Product id='{}' full_name='{}'>".format(num_products, product_name))
            tag_result_by_lines.append("<Ner_Tag Brand='' Category='' Model='' Info=''></Ner_Tag>")
            tag_result_by_lines.append("<Brand>{}</Brand>".format(brand))

            candidate_tokens, candidate_ids = get_candidate_tokens(product_name, tokens)
            tag_result_by_lines.append("<Tokens>")
            for t, token_id in zip(candidate_tokens, candidate_ids):
                prefix = " ".join(tokens[token_id - 4: token_id])[:20]
                suffix = " ".join(tokens[token_id + 1: token_id + 5])[:20]
                # context = " ".join(tokens[token_id - 3: token_id + 4])
                context = "{} {} {}".format(prefix, t, suffix)
                context = "{message:{fill}{align}{width}}".format(message=context, width=70, fill="_", align="<")
                tag_result_by_lines.append("{}, {}, {} ,".format(context, token_id, t))

                # tag_result_by_lines.append("\n<Token>")
                # tag_result_by_lines.append("<Text Ner_Tag=''><![CDATA[{:^20s}]]></Text>".format(t))
                # tag_result_by_lines.append("<Context><![CDATA[{}]]></Context>".format(context))
                # tag_result_by_lines.append("</Token>")
                # tag_result_by_lines.append("<Token prefix='{:20s}'    text='{:20s}'    ner_tag=''    "
                #                            "suffix='{:20s}'></Token>".format(prefix, t, suffix))
            tag_result_by_lines.append("</Tokens>")

            # tag_result_by_lines.append(brand)
            # tag_result_by_lines.append("</Brand>")

            # tag_result_by_lines.append("<Category>")
            # tag_result_by_lines.extend(utils.tokenize(cat))
            # tag_result_by_lines.append("</Category>")
            #
            # tag_result_by_lines.append("<Model>")
            # tag_result_by_lines.append("</Model>")
            #
            # tag_result_by_lines.append("<Info>")
            # tag_result_by_lines.append("</Info>")

            tag_result_by_lines.append("</Product>")

            if (i+1) % 100 == 0:
                print("Process {}/{} products done".format(i+1, sub_df.shape[0]))

        # if (num_products % 100) == 0:
        # num_output_files += 1
        save_path = "./Data/output/Full_Doc/{}.xml".format(root_id)
        full_doc.append("</Products>")
        utils.save_list(full_doc, save_path)
        save_path = "./Data/output/Tag_Result/{}.xml".format(root_id)
        tag_result_by_lines.append("</Products>")
        utils.save_list(tag_result_by_lines, save_path)

        # doc_by_lines, tag_result_by_lines = [], []
            # break

        # print("Process {}/{} files done".format(fname_id + 1, len(fnames)))

    exec_time = time.time() - start_time
    print("Time : {:.2f} seconds".format(exec_time))


def build_ner_dataset():

    # Load full doc json
    full_doc_dir = "./Data/output/Full_Doc_Json"
    fpaths = utils.get_file_paths(full_doc_dir)
    full_docs = []
    for fpath in fpaths:
        full_docs.extend(utils.load_json(fpath))

    map_product_id_full_info = {str(product["product_id"]): product for product in full_docs}
    # print(map_product_id_full_info.get("1901"))

    fpath = "./Data/output/Tag_Result/16.xml"
    raw_lines = utils.load_str(fpath).split("\n")
    num_docs = 0
    max_docs = 20
    num_sents = 0
    product_id = -1

    token_ids_of_doc = []
    ner_tags_of_doc = []
    map_token_id_tag = {}

    data = []
    product_id_pattern = re.compile("(?<=id=').*(?=' full_name)")
    end_sent_pattern = re.compile("[?!.]+")

    for line in raw_lines:
        if line.startswith("<Product id"):
            # print(line)
            product_id = product_id_pattern.findall(line)[0]
            # print(product_id)
            token_ids_of_doc = []
            ner_tags_of_doc = []
            map_token_id_tag = {}

        elif line.startswith("</Product>"):

            if len(map_token_id_tag) == 0:
                continue

            num_docs += 1
            doc = map_product_id_full_info[product_id]["doc"]

            tokens_of_doc, post_tags_of_doc = ViPosTagger.postagging(doc)

            full_ner_tags_of_doc = [map_token_id_tag.get(str(token_id), "O")
                                    for token_id in range(len(tokens_of_doc))]

            # Assign sent id
            sent_ids = []
            is_end_sent = True
            for token in tokens_of_doc:
                if end_sent_pattern.search(token):
                    is_end_sent = True
                else:
                    if is_end_sent:
                        num_sents += 1
                    is_end_sent = False
                sent_ids.append(num_sents)
            # print(tokens_of_doc)
            # print(map_token_id_tag)
            # exit()

            for token_id in range(len(tokens_of_doc)):
                sent_id, token, post_tag, ner_tag = sent_ids[token_id], tokens_of_doc[token_id], \
                                                    post_tags_of_doc[token_id], full_ner_tags_of_doc[token_id]
                # ner_tag = map_token_id_tag.get(str(token_id), "O")

                if ner_tag != "O":
                    if token_id > 0 and full_ner_tags_of_doc[token_id-1] == ner_tag \
                            and sent_ids[token_id-1] == sent_ids[token_id]:
                        prefix = "I-"
                    else:
                        prefix = "B-"
                    map_convert_tag = {"Category": "Cat",
                                       "Model": "Mod",
                                       "Info": "Inf",
                                       "Brand": "Bra",
                                       "O": "O"}
                    ner_tag = map_convert_tag[ner_tag]
                    ner_tag = prefix + ner_tag
                data.append((num_docs, sent_id, token, post_tag, ner_tag))

            product_id = -1

            # if num_docs >= max_docs:
            #     break

        elif line.startswith("<Tokens>") or line.startswith("</Tokens>") or \
                line.startswith("<Products>") or line.startswith("</Products>") or \
                line.startswith("<Ner_Tag>") or line.startswith("<Brand>"):
            continue
        else:
            arr = line[71:].split(",")
            token_id = arr[0].strip()
            ner_tag = arr[-1].strip()
            if len(token_id) == 0 or len(ner_tag) == 0:
                continue
            token_ids_of_doc.append(token_id)
            # ner_tag = ner_tag if len(ner_tag) > 0 else "O"
            ner_tags_of_doc.append(ner_tag)
            map_token_id_tag[token_id] = ner_tag

    data = pd.DataFrame(data, columns=["Doc_Id", "Sent_Id", "Token", "Pos_Tag", "Ner_Tag"])
    print(data.head(10))
    print("\n==================")
    print(data.tail(10))
    print(data.shape[0])

    save_path = "./Data/Ner_Dataset_2/ner.csv"
    utils.save_csv(data, save_path)


if __name__ == "__main__":
    # main()
    build_ner_dataset()
