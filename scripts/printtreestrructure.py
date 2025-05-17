
# %%
def print_tree_structure(graph):
    for node in graph.nodes:
        parent = list(graph.predecessors(node))
        children = list(graph.successors(node))
        print(f"Node {node}:")
        print(f"  text: {graph.nodes[node]['text'].lower()}")
        print(f"  deprel: {graph.nodes[node]['deprel']}")
        print(f"  Parent: {parent}")
        print(f"  Children: {children}\n")


import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
from networkx.readwrite import json_graph
import json
import networkx as nx

def load_trees_from_file(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    trees = []
    for tree_data in data:
        tree = json_graph.node_link_graph(tree_data)
        trees.append(tree)
    return trees

# Load the trees from the file
H = load_trees_from_file('pattern_trees.json')

for h in H:
    print_tree_structure(h)

