import networkx as nx
import json
from collections import defaultdict
from networkx.readwrite import json_graph

with open('train.json', 'r') as file:
   train = json.load(file)

trees = []
for sent, words in enumerate(train):
    def search(graph, property):
       for node , data in graph.nodes(data = True):
          if data.get('word_id') == property:
             return node
       
    G1 = nx.DiGraph()
    for i, word in enumerate(words,1):
        G1.add_node(i, sentence_id=word['sentence_id'] , word_id=word['word_id'], text= word['text'].lower(), head = word['head'], deprel = word['deprel'], probability = word['probability'], children = word['children'], skill_tag = word['skill_tag'], skill_headword= word['skill_headword'])
    for node, data in G1.nodes(data=True):
        if data.get('head') > 0:
            parent_node = search(G1, data.get('head'))
            G1.add_edge(parent_node, node, weight = data.get('probability'))
    trees.append(G1)

headword_trees = []
for tree in trees:
    for node in tree.nodes(data = True):  
      if node[1].get('skill_headword') == True:
            parent = list(tree.predecessors(node[0]))
            if parent == []:
                headword_trees.append(tree)
                continue
            temp = tree.copy()
            temp.remove_edge(parent[0], node[0])
            components = list(nx.weakly_connected_components(temp))
            subgraphs = [temp.subgraph(component).copy() for component in components]
            headword_trees.append(subgraphs[1])

# merging headword trees
def save_trees_to_file(trees, filename):
    data = []
    for tree in trees:
        tree_data = json_graph.node_link_data(tree)
        data.append(tree_data)
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def find_root(graph):
    for node in graph.nodes:
        if graph.in_degree(node) == 0:
            return node
    return None

def relabeling(graph1,graph2):
    offset = max(graph1.nodes) + 1
    graph2 = nx.relabel_nodes(graph2, lambda x: x + offset)
    return graph2

def merge_graphs(graph_list):
    if not graph_list:
        return None
    merged_graph = nx.DiGraph()
    new = []
    stack = []
    new.append(graph_list[0])

    for i in graph_list:
        if not graph_list.index(i)+1 == len(graph_list):
            if stack == []:
                x = relabeling(i, graph_list[graph_list.index(i)+1])
                graph_list[graph_list.index(i)] = x
                new.append(x)
                stack.append(x)
            else:
                x = relabeling(stack.pop(), graph_list[graph_list.index(i)+1])
                new.append(x)
                stack.append(x)

    for graph in new:
        merged_graph = nx.compose(merged_graph, graph)
    
    roots = []
    for node in merged_graph.nodes:
        parent = list(merged_graph.predecessors(node))
        if parent == []:
            roots.append(node)
    children = []
    for root in roots[1:]:
        children.extend(list(merged_graph.successors(root)))

    for child in children:
        weight = merged_graph.nodes[child].get('probability') 
        merged_graph.add_edge(roots[0],child ,weight = weight)
    for root in roots[1:]:
        merged_graph.remove_node(root)
    

    return merged_graph

# Group trees by their root 'text' property
trees_by_root_text = defaultdict(list)

for tree in headword_trees:
    root = find_root(tree)
    if root is not None:
        root_text = tree.nodes[root]['text'].lower()
        trees_by_root_text[root_text].append(tree)

# Merge trees with the same root 'text'
pattern_trees = []

for root_text, tree_group in trees_by_root_text.items():
    merged_tree = merge_graphs(tree_group)
    if merged_tree is not None:
        pattern_trees.append(merged_tree)

headwortrees = save_trees_to_file(pattern_trees,'pattern_trees_ff1.json')