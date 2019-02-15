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
    fpath = "./Data/output/Doc_by_lines/doc_1"
    raw_lines = utils.load_str(fpath).split("\n")
    num_docs = 0
    max_docs = 20
    num_sents = 0

    tokens_of_doc = []
    ner_tags_of_doc = []
    data = []
    p = re.compile("[?!.]+")
    for line in raw_lines:
        if line.startswith("<Product"):
            tokens_of_doc = []
            ner_tags_of_doc = []

        elif line.startswith("</Product>"):
            num_docs += 1
            doc = " ".join(tokens_of_doc)
            post_tags_of_doc = ViPosTagger.postagging(doc)[1]

            # Assign sent id
            sent_ids = []
            is_end_sent = True
            for token in tokens_of_doc:
                if p.search(token):
                    is_end_sent = True
                else:
                    if is_end_sent:
                        num_sents += 1
                    is_end_sent = False
                sent_ids.append(num_sents)

            for token_id in range(len(tokens_of_doc)):
                sent_id, token, ner_tag, post_tag = sent_ids[token_id], tokens_of_doc[token_id], \
                                                    ner_tags_of_doc[token_id], post_tags_of_doc[token_id]
                if ner_tag != "O":
                    if token_id > 0 and ner_tags_of_doc[token_id-1] == ner_tag \
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

            if num_docs >= max_docs:
                break

        else:
            idx = line.rfind(",")
            token = line[:idx]
            ner_tag = line[idx+1:]
            if token == "":
                continue
            tokens_of_doc.append(token)
            ner_tag = ner_tag if len(ner_tag) > 0 else "O"
            ner_tags_of_doc.append(ner_tag)

    data = pd.DataFrame(data, columns=["Doc_Id", "Sent_Id", "Token", "Pos_Tag", "Ner_Tag"])
    print(data.head(10))
    print("\n==================")
    print(data.tail(10))
    print(data.shape[0])

    save_path = "./Data/Ner_Dataset/ner.csv"
    utils.save_csv(data, save_path)


if __name__ == "__main__":
    main()
    # build_ner_dataset()
