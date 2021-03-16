# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import pandas
import rmm
import cudf
import cugraph
import numpy as np
from math import ceil

default_palette = [
    # https://colorbrewer2.org/#type=diverging&scheme=Spectral&n=11
    4288545090,
    4292165199,
    4294208835,
    4294815329,
    4294893707,
    4294967231,
    4293326232,
    4289453476,
    4284924581,
    4281501885,
    4284370850
]


def print_df(name, df):
    print(str(name) + " dtypes:\n" + str(df.dtypes))
    print(str(name) + ":\n" + str(df))


def category_to_color(categories, color_palette=None, cat_colors: dict = None):

    if (cat_colors is not None):
        # cats = cudf.Series.from_categorical(pandas.Categorical(categories, categories=list(cat_colors.keys()), ordered=True))
        # cats = cudf.Series(categories).astype("categories")

        # cats = cats.cat.set_categories(list(cat_colors.keys()))
        pass

    if color_palette is None:
        color_palette = default_palette
    color_indices = cudf.Series(categories)
    color_palette = cudf.Series(color_palette)

    # Check if we need to convert from string to indices
    if color_indices.dtype.type != np.uint32:
        if (cat_colors is not None):
            # Use the category names to convert to codes to ensure the proper indexes
            color_indices = cudf.Series(categories,
                                        dtype="category").cat.set_categories(list(cat_colors.keys()))

            assert not color_indices.isna().any()

            # Now convert to uint32 codes
            color_indices = color_indices.cat.codes.astype(np.uint32)
        else:
            # Auto factorize
            color_indices = cudf.Series(categories.factorize()[0]).astype(np.uint32)

    # Set the color_palette if we have a category dictionary
    if (cat_colors is not None):
        color_palette = cudf.Series(list(cat_colors.values()))

    color_palettes = []
    num_color_ids = color_indices.max() + 1
    for i in range(ceil(num_color_ids / len(color_palette))):
        color_palettes.append(color_palette)
    return cudf.Series(
        cudf.core.column.build_categorical_column(
            ordered=True,
            codes=color_indices._column,
            categories=cudf.concat(color_palettes)[:num_color_ids],
        ).as_numerical_column(dtype=np.uint32))


def compute_edge_bundles(edges, id_, src, dst):
    def drop_index(df):
        return df.reset_index(drop=True)

    def smoosh(df):
        size = sum([df[x].dtype.itemsize for x in df])
        data = drop_index(drop_index(df).stack()).data
        dtype = cudf.utils.dtypes.min_unsigned_type(0, size * 8)
        return cudf.core.column.NumericalColumn(data, dtype=dtype)

    edges = cudf.DataFrame({
        "eid": drop_index(edges[id_]),
        "src": drop_index(edges[src]),
        "dst": drop_index(edges[dst]),
    })
    # Create a duplicate table with:
    # * all the [src, dst] in the upper half
    # * all the [dst, src] pairs as the lower half, but flipped so dst->src, src->dst
    bundles = drop_index(
        cudf.DataFrame({
            "eid": cudf.concat([edges["eid"], edges["eid"]], ignore_index=True),  # concat [src, dst] into the "src" column
            "src": cudf.concat([edges["src"], edges["dst"]], ignore_index=True),  # concat [dst, src] into the "dst" column
            "dst": cudf.concat([edges["dst"], edges["src"]], ignore_index=True),
        }))

    # Group the duplicated edgelist by [src, dst] and get the min edge id.
    # Since all the [dst, src] pairs have been flipped to [src, dst], each
    # edge with the same [src, dst] or [dst, src] vertices will be assigned
    # the same bundle id
    bundles = bundles.groupby(["src", "dst"]).agg({"eid": "min"}).reset_index().rename(columns={"eid": "bid"}, copy=False)

    # Join the bundle ids into the edgelist
    edges = edges.merge(bundles, on=["src", "dst"], how="inner")

    # Determine each bundle"s size and relative offset
    lengths = edges["bid"].value_counts(sort=False).sort_index()
    bundles = lengths.index.to_series().unique()
    offsets = lengths.cumsum() - lengths

    # Join the bundle segment lengths + offsets into the edgelist
    edges = edges.merge(cudf.DataFrame({
        "bid": drop_index(bundles.astype(np.uint32)),
        "start": drop_index(offsets.astype(np.uint32)),
        "count": drop_index(lengths.astype(np.uint32)),
    }),
                        on="bid",
                        how="left")

    # Determine each edge's index relative to its bundle
    edges = drop_index(edges.sort_values(by="bid"))
    edges["index"] = edges.index.to_series() - edges["start"]
    edges["index"] = edges["index"].astype(np.uint32)

    # Re-sort the edgelist by edge id and cleanup
    edges = drop_index(edges.sort_values(by="eid"))
    edges = edges.rename(columns={"eid": "id"}, copy=False)
    edges = edges[["id", "src", "dst", "index", "count"]]

    return {
        "edge": smoosh(edges[["src", "dst"]]).astype(np.uint64),
        "bundle": smoosh(edges[["index", "count"]]).astype(np.uint64),
    }


def from_cudf_edgelist(df, source="src", target="dst"):
    """
    Construct an enhanced graph from a cuDF edgelist that doesn't collapse
    duplicate edges and includes columns for node degree and edge bundle.
    """
    def drop_index(df):
        return df.reset_index(drop=True)

    def smoosh(df):
        size = sum([df[x].dtype.itemsize for x in df])
        data = drop_index(drop_index(df).stack()).data
        dtype = cudf.utils.dtypes.min_unsigned_type(0, size * 8)
        return cudf.core.column.NumericalColumn(data, dtype=dtype)

    def make_nodes(df, src, dst):
        nodes = drop_index(df[src].append(df[dst], ignore_index=True).unique())
        ids = drop_index(cudf.Series(nodes.factorize()[0])).astype(np.uint32)
        return drop_index(cudf.DataFrame({"id": ids, "node": nodes}).sort_values(by="id"))

    def make_edges(df, src, dst, nodes):
        def join(edges, nodes, col):
            edges = edges.set_index(col, drop=True)
            nodes = nodes.set_index("node", drop=True)
            edges = edges.join(nodes).sort_values(by="eid")
            edges = edges.rename(columns={"id": col}, copy=False)
            return drop_index(edges)

        edges = df.reset_index().rename(columns={"index": "eid"}, copy=False)
        edges = join(join(edges.assign(src=df[src], dst=df[dst]), nodes, "src"), nodes, "dst")
        return drop_index(edges.rename(columns={"eid": "id"}, copy=False))

    df = drop_index(df)
    graph = cugraph.MultiDiGraph()
    nodes = make_nodes(df, source, target)
    edges = make_edges(df, source, target, nodes)
    graph.edgelist = cugraph.Graph.EdgeList(edges["src"], edges["dst"])
    nodes = nodes.set_index("id", drop=False).join(graph.degree().set_index("vertex"))
    return graph, drop_index(nodes.sort_index()), edges


def annotate_nodes(graph, nodes, edges):
    return nodes.assign(
        # add node names
        name=nodes["name"] if "name" in nodes else nodes["id"],
        # add node sizes
        size=(nodes["degree"].scale() * (50 - 2) + 2).astype(np.uint8),
        # add node colors
        color=category_to_color(
            cugraph.spectralBalancedCutClustering(graph, min(9, graph.number_of_nodes() -
                                                             1)).sort_values(by="vertex").reset_index(drop=True)["cluster"],
            color_palette=[
                # Make all nodes white
                4294967295
                #                 # https://colorbrewer2.org/#type=diverging&scheme=Spectral&n=9
                #                 4292165199, 4294208835,
                #                 4294815329, 4294893707,
                #                 4294967231, 4293326232,
                #                 4289453476, 4284924581,
                #                 4281501885
            ]))


def annotate_edges(graph, nodes, edges):
    def drop_index(df):
        return df.reset_index(drop=True)

    def smoosh(df):
        size = sum([df[x].dtype.itemsize for x in df])
        data = drop_index(drop_index(df).stack()).data
        dtype = cudf.utils.dtypes.min_unsigned_type(0, size * 8)
        return cudf.core.column.NumericalColumn(data, dtype=dtype)

    def edge_colors(nodes, edges, col):
        edges = edges[["id", col]].set_index(col, drop=True)
        nodes = nodes[["id", "color"]].set_index("id", drop=True)
        return drop_index(edges.join(nodes).sort_values(by="id")["color"])

    return edges.assign(
        # add edge names
        name=edges["name"] if "name" in edges else edges["id"],
        # add edge colors
        color=smoosh(cudf.DataFrame({
            "src": edge_colors(nodes, edges, "src"),
            "dst": edge_colors(nodes, edges, "dst"),
        })))


def make_capwin_graph(df):
    def drop_index(df):
        return df.reset_index(drop=True)

    def smoosh(df):
        size = sum([df[x].dtype.itemsize for x in df])
        data = drop_index(drop_index(df).stack()).data
        dtype = cudf.utils.dtypes.min_unsigned_type(0, size * 8)
        return cudf.core.column.NumericalColumn(data, dtype=dtype)

    def add_edge_colors(edges, category):
        colors = drop_index(
            category_to_color(
                edges[category],
                color_palette=[
                    #     FALSE,      TRUE
                    # 268435455, 268369920

                    #    FALSE,       TRUE
                    # 33554431, 4293138972

                    #    ADDRESS   AUTH KEYS CREDENTIALS       EMAIL      FALSE
                    # 4294967091, 4294410687, 4293138972, 4281827000,  33554431
                    0x0fffeda0,  # address 268430752
                    0x0ffed976,  # bank_acct 268360054
                    0x0ffeb24c,  # email
                    0x0ffd8d3c,  # govt_id
                    0x0ffc4e2a,  # name
                    0x01ffffff,  # none 33554431
                    0x0fe31a1c,  # phone_num
                    0x0fbd0026,  # secret_keys
                    0x0f800026,  # user
                ],
                cat_colors={
                    "address": 0x0fffffbf,
                    "bank_acct": 0x0fffffbf,
                    "credit_card": 0x0fffffbf,
                    "email": 0x0fffffbf,
                    "govt_id": 0x0fffffbf,
                    "name": 0x0fffffbf,
                    "none": 0x02ffffff,
                    "phone_num": 0x0fffffbf,
                    "secret_keys": 0x80ff0000,
                    "user": 0x0fffffbf,
                }).astype(np.uint32))
        return edges.assign(color=smoosh(cudf.DataFrame({
            "src": drop_index(colors), "dst": drop_index(colors)
        })).astype(np.uint64),
                            src_color=colors)

    # Create graph
    graph, nodes, edges = from_cudf_edgelist(df, "src_ip", "dest_ip")
    # Add vis components
    nodes = nodes.rename(columns={"node": "name"}, copy=False)
    nodes = annotate_nodes(graph, nodes, edges)
    # add edge colors
    edges = add_edge_colors(edges, "pii")
    # add edge names
    edges["name"] = edges["src_ip"] + " -> " + edges["dest_ip"] + ("\nPII: " + edges["pii"]).replace("\nPII: FALSE", "")
    return graph, nodes, edges


def make_capwin_dataset(start, end, src_path, dst_path):
    def arange(size, dtype="uint32"):
        return cudf.core.index.RangeIndex(0, size).to_series().astype(dtype)

    def relabel_nodes(nodesA, nodesB):
        nodesA = nodesA.rename(columns={"id": "lhs_id"}, copy=False).set_index("name", drop=True)
        nodesB = nodesB.rename(columns={"id": "rhs_id"}, copy=False).set_index("name", drop=False)
        nodes = nodesA.join(nodesB, how="outer", sort=True).sort_values(by="lhs_id").reset_index(drop=True)
        nodes = nodes.reset_index().rename(columns={"index": "id"}, copy=False)
        nodes = nodes.drop(columns=["lhs_id"]).rename(columns={"rhs_id": "remap"}, copy=False)
        return nodes.reset_index(drop=True).sort_values(by="id").reset_index(drop=True).astype({"id": np.uint32})

    def relabel_edges(edgesA, edgesB, nodes):
        def remap(edges, nodes, col):
            edges = edges.set_index(col)
            nodes = nodes.rename(columns={"id": col}, copy=False)
            nodes = nodes.set_index("remap")
            return edges.join(nodes, sort=True).reset_index(drop=True)

        return remap(remap(edgesB, nodes, "src"), nodes,
                     "dst").drop(columns=["id"]).sort_values(by=["src", "dst"]).reset_index(drop=True).reset_index().rename(
                         columns={"index": "id"}, copy=False)

    df, nodes, edges = (cudf.DataFrame(), None, None)

    for i in range(start, end):
        # print(f"reading {src_path}/{i}.0.csv")
        try:
            df2 = cudf.read_csv(
                f"{src_path}/{i}.0.csv",
                header=0,
                parse_dates=[1],
                usecols=[2, 3, 6],
                dtype=[
                    "int32",  # index
                    "datetime64[ms]",  # timestamp
                    "str",  # src_ip
                    "str",  # dest_ip
                    "int32",  # src_port
                    "int32",  # dest_port
                    "str"  # pii
                ]).reset_index(drop=True)
        except Exception:
            # print(f"missing {src_path}/{i}.0.csv")
            continue

        # print(f"read {src_path}/{i}.0.csv")

        if "si" in df2:
            df2 = df2.rename(columns={"si": "pii"}, copy=False)

        df = cudf.concat([df.reset_index(drop=True), df2],
                         ignore_index=True).reset_index(drop=True).sort_values(by=["src_ip", "dest_ip"]).reset_index(drop=True)

        results = make_capwin_graph(df[["src_ip", "dest_ip", "pii"]])

        if i == 0:
            nodes = results[1][["id", "name", "degree", "size", "color"]]
            edges = results[2][["id", "name", "src", "dst", "color"]]
        else:
            nodes = relabel_nodes(nodes[["id", "name"]], results[1][["id", "name", "degree", "size", "color"]])
            edges = relabel_edges(edges[["id", "name", "src", "dst", "color"]],
                                  results[2][["id", "name", "src", "dst", "color"]],
                                  nodes[["id", "remap"]])

        edges = edges.assign(**compute_edge_bundles(edges, "id", "src", "dst"))

        nodes_out = nodes[["name", "id", "color", "size"]]
        edges_out = edges[["name", "src", "dst", "edge", "color", "bundle"]]

        nodes_out.to_csv(f"{dst_path}/{i}.0.nodes.csv", index=False)
        edges_out.to_csv(f"{dst_path}/{i}.0.edges.csv", index=False)

    # Print all the PII values at the end so we can map the edge colors
    # print(df["pii"].unique())


# %%
import shutil
import os

output_dir = "/home/mdemoret/Repos/rapids/rapids-js-dev/modules/demo/graph/data/network_graph_viz_frames_multi_label"
shutil.rmtree(output_dir)

os.makedirs(output_dir, exist_ok=True)

make_capwin_dataset(0,
                    200,
                    "viz_frames",
                    "/home/mdemoret/Repos/rapids/rapids-js-dev/modules/demo/graph/data/network_graph_viz_frames_multi_label")

# %%
