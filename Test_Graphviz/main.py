import pygraphviz as PG
import pandas as pd


def load_cats():
    fpath = "./Dataset/productcat.xlsx"
    df = pd.read_excel(fpath)

    print(df.head())

    map_id_name = {}
    parent_childs = []
    for i, row in df.iterrows():
        cat_id = row["cat_id"]
        cat_name = row["cat_name"]
        parent_id = row["parent_id"]
        map_id_name.update({cat_id: cat_name})
        parent_childs.append((parent_id, cat_id))

    # Convert id to name
    parent_childs = [(map_id_name.get(pid), map_id_name.get(cid)) for pid, cid in parent_childs if pid > 0]
    print(parent_childs[:5])
    print(len(parent_childs))

    return parent_childs


if __name__ == "__main__":
    A = PG.AGraph(directed=True, strict=True)

    for pname, cname in load_cats():
        A.add_edge(pname, cname)

    # save the graph in dot format
    A.write('./Visualize/ademo.dot')

    # pygraphviz renders graphs in neato by default,
    # so you need to specify dot as the layout engine
    A.layout(prog='dot')

