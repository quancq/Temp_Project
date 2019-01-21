import os
import pandas as pd
import utils
from lxml import etree as ET
import time


def extract_token_tag(xml_path, sel_tag="Product"):
    tree = utils.load_xml(xml_path)
    # print(ET.tostring(tree.getroot(), pretty_print=True))

    text_with_nodes = tree.xpath("//TextWithNodes/Node")
    # print(ET.tostring(text_with_nodes))

    tags_with_offset = []
    annotations = tree.xpath("//AnnotationSet[@Name='Brand']/Annotation")
    map = {}
    for elm in annotations:
        # print("{} - {} - {}".format(elm.get("StartNode"), elm.get("EndNode"), elm.get("Type")))
        start = int(elm.get("StartNode"))
        end = int(elm.get("EndNode"))
        tag = elm.get("Type")
        if start < end and map.get((start, end, tag)) is None:
            map[(start, end, tag)] = 1
            tags_with_offset.append((start, end, tag))

    # print("File has {} tags".format(len(tags_with_offset)))

    tokens_with_offset = []
    tokens_with_tags = []
    is_begin = False
    for i in range(len(text_with_nodes) - 1):
        # print("{} - {} - {}".format(element.get("id"), element.text, element.tail))
        tokens = text_with_nodes[i].tail.strip().split(" ")
        for token in tokens:
            start_of_token = int(text_with_nodes[i].get("id"))
            end_of_token = int(text_with_nodes[i+1].get("id"))
            tokens_with_offset.append((start_of_token, end_of_token, token))

            # Get tag of token
            tags = []
            for start_of_tag, end_of_tag, tag in tags_with_offset:
                if start_of_tag <= start_of_token <= end_of_token <= end_of_tag and tag == sel_tag:
                    tags.append(tag)
                    break
            if len(tags) == 0:
                tags.append("O")
                is_begin = False
            else:
                if not is_begin:
                    tags = ["B-" + tags[0]]
                    is_begin = True
                else:
                    tags = ["I-" + tags[0]]
                    # is_begin = False

            tokens_with_tags.append((token, tags[0]))

    return tokens_with_tags


if __name__ == "__main__":
    start_time = time.time()
    input_dir = "./Data/Input"
    output_dir = "./Data/Output"
    # path = "./Data/Input/a.xml"
    num_tokens = 0
    for fname in utils.get_file_names(input_dir):
        fpath = os.path.join(input_dir, fname)
        tokens_with_tag = extract_token_tag(fpath, sel_tag="Product")
        num_tokens += len(tokens_with_tag)
        # for token, tags_of_token in tokens_with_tags:
        #     print("{} - {}".format(token, tags_of_token))

        save_path = os.path.join(output_dir, "{}.csv".format(fname[:fname.rfind(".")]))
        df = pd.DataFrame(tokens_with_tag, columns=["Word", "Ner_Tag"])
        utils.save_csv(df, save_path)

    exec_time = time.time() - start_time
    print("Extracted {} tokens. Time : {:.2f} seconds".format(num_tokens, exec_time))

