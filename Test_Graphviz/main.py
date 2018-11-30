import pygraphviz as PG
import pandas as pd
import pydot
import os
import utils
from subprocess import check_call


def load_cats():
    fpath = "./Dataset/productcat.xlsx"
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
        if parent_id != 0:
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


if __name__ == "__main__":
    map_root_pairs, map_id_name = load_cats()

    for root_id, pairs in map_root_pairs.items():
        root_name = map_id_name.get(root_id)
        print("\nRoot name : ", root_name)

        A = PG.AGraph(directed=True, strict=True)

        for parent_id, child_id in pairs:
            parent_name, child_name = map_id_name.get(parent_id), map_id_name.get(child_id)
            print("Parent name : {} - Child name : {}".format(parent_name, child_name))

            A.add_edge(parent_name, child_name)

        # # save the graph in dot format
        dot_dir = "./Visualize/dot"
        utils.mkdir(dot_dir)
        save_dot_path = os.path.join(dot_dir, root_name)
        A.write(save_dot_path)

        # pygraphviz renders graphs in neato by default,
        # so you need to specify dot as the layout engine
        A.layout(prog='dot')
        print("Save file {} done".format(save_dot_path))

        jpg_dir = "./Visualize/jpg"
        utils.mkdir(jpg_dir)
        (graph,) = pydot.graph_from_dot_file(save_dot_path)

        save_jpg_path = os.path.join(jpg_dir, "{}.jpg".format(root_name))
        graph.write_jpg(save_jpg_path)

        # jpg_dir = "./Visualize/jpg"
        # save_path = os.path.join(jpg_dir, "{}.jpg".format(root_name))
        # A.draw(save_path, format="jpg", prog="nop")

        check_call(['dot', '-Tjpg', save_dot_path, '-o', save_jpg_path, "-Grankdir=LR"])

