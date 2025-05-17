# %%
import stanza

# stanza.download('en', download_method=None)

import json
with open('test.json', 'r') as file:
    data = json.load(file)

# Extract the 'tokens' values from each entry in the JSON data
tokens_list = [entry['tokens'] for entry in data]
print(len(tokens_list))
tag_skill_list = [entry['tags_skill'] for entry in data]
tag_knowledge_list = [entry['tags_knowledge'] for entry in data]

stanza_input = []
stanza_skill_tags = []
# stanza_knowledge_tags =[]
for i in range(0,3569):
  if "B" in tag_skill_list[i]:# or "B" in tag_knowledge_list[i]:
    stanza_input.append(tokens_list[i])
    stanza_skill_tags.append(tag_skill_list[i])
    # stanza_knowledge_tags.append(tag_knowledge_list[i])
print(len(stanza_input))
print(len(stanza_skill_tags))

# Define a custom Word class with additional custom properties
class MyWord:
    def __init__(self, text, lemma, pos, custom_property=None):
        self.text = text
        self.lemma = lemma
        self.pos = pos
        self.skill_tag = custom_property
        self.knowledge_tag = custom_property
        self.children = []
    
    def add_child(self, child_node):
        self.children.append(child_node)


# Override the word_class argument to use your custom Word class
nlp = stanza.Pipeline('en', tokenize_pretokenized=True, word_class=MyWord, processors='tokenize,mwt,pos,lemma,depparse', download_method=None)

# Process a sentence
doc = nlp(stanza_input)

print(len(doc.sentences))
# Set a custom property for each word
for j, sent in enumerate(doc.sentences):
  for i, word in enumerate(doc.sentences[j].words):
    word.skill_tag = stanza_skill_tags[j][i]

print(doc.sentences[0].words)

head_words = []
head_word_object = []
for j, sent in enumerate(doc.sentences):
  for i, word in enumerate(doc.sentences[j].words):
    word.sentence_id = j
    word.skill_headword = False
    if word.skill_tag == "B" and word.head != 0 :
      x= doc.sentences[j].words[word.head - 1]
      if x.skill_tag == "O":
        head_words.append({'word':x.text, 'sentence id': j, 'word id': x.id})
        head_word_object.append(x)
        x.skill_headword = True

      else:
        while x.skill_tag != "O" and x.head != 0 :
          x = doc.sentences[j].words[x.head - 1]
        if x.skill_tag == "O":
          head_words.append({'word':x.text, 'sentence id': j, 'word id': x.id})
          head_word_object.append(x)
          x.skill_headword = True

# %%
# add children
roots =[]

for j, sent in enumerate(doc.sentences):
  # for i, word1 in enumerate(doc.sentences[j].words):
  #   for x, word2 in enumerate(doc.sentences[j].words):
  #      if word1.id == word2.head:
  #       print(word1.skill_tag)          
        
  #       word1.children.append(word2.id)
  # Create a dictionary to store word ids and their children
    children_dict = {word.id: [] for word in sent.words}
    children_dict_word = {word.id: [] for word in sent.words}


    # Populate the children_dict
    for word in sent.words:
        head_id = word.head
        if head_id == 0:
           roots.append(word.id)
        if head_id > 0:  # Head id of 0 means it's the root
            children_dict_word[head_id].append(word)
            children_dict[head_id].append(word.id)


    # Add the children property to each word
    for word in sent.words:
        word.children_word = children_dict_word[word.id]
        word.children = children_dict[word.id]


    # Print out the word details with the custom property
    for word in sent.words:
        print(f"Word: {word.text}, ID: {word.id}, Head: {word.head}, Children: {word.children}")

# add freq.

# formula for freq = number of all nodes with B or I tags in subtree of an edge / number of all nodes with B or I tages in e level higher + 1
# the denominator + 1 is for avoiding freq. to become 1


for j, sent in enumerate(doc.sentences):
  

  def dfs_postorder(root):
    if root:
        for child in root.children_word:
            # print(child)
            dfs_postorder(child)
        print(str(root.id) + "->", end='')
        if not root.children_word: 
          root.freq = (root.skill_tag == "B" or root.skill_tag == "I") + 0
        else:
           freq1 = 0
           for child in root.children_word:
              freq1 += child.freq
           freq1 += (root.skill_tag == "B" or root.skill_tag == "I")
           root.freq = freq1
     
  def probabilityCount(root):
     if root.head == 0:
        root.prob = 0
     else:
        parent = sent.words[root.head-1]
        p_freq = parent.freq - (parent.skill_tag == "B" or parent.skill_tag == "I")
        root.prob = root.freq / (p_freq + 1)
     for child in root.children_word:
        probabilityCount(child)
        
     
       
       

  root = sent.words[roots[j]-1]
  dfs_postorder(root)
  probabilityCount(root)
# for word in doc.sentences[0].words:
  #  print(word.id, word.freq, word.prob, word.children)

combined_data = []
for j, sent in enumerate(doc.sentences):
   for word in sent.words:
                info = {
                    'sentence_id': j,
                    'word_id': word.id,
                    'text': word.text.lower(),
                    'lemma': word.lemma,
                    'upos': word.upos,
                    'xpos': word.xpos,
                    'head': word.head,
                    'deprel': word.deprel,
                    'probability': word.prob,
                    'children': word.children,
                    'skill_tag': word.skill_tag,
                    'skill_headword': word.skill_headword

                }
                combined_data.append(info)
   
with open('labeled_dataset_token_info_test.json', 'w') as f:
    json.dump(combined_data, f, indent=4)
# %%
import networkx as nx


trees = []
# for word, i in enumerate(data, 1):
    # G1.add_node(i, **word)
for j, sent in enumerate(doc.sentences):
    def search(graph, property):
       for node , data in graph.nodes(data = True):
          if data.get('word_id') == property:
             return node
       
    G1 = nx.DiGraph()
    for i, word in enumerate(sent.words,1):
        G1.add_node(i, sentence_id=j , word_id=word.id, text= word.text.lower(), head = word.head, deprel = word.deprel, probability = word.prob, children = word.children, skill_tag = word.skill_tag, skill_headword= word.skill_headword)
    for node, data in G1.nodes(data=True):
        if data.get('head') > 0:
            parent_node = search(G1, data.get('head'))
            G1.add_edge(parent_node, node, weight = data.get('probability'))
    trees.append(G1)

def print_tree_structure(graph):
    for node in graph.nodes:
        parent = list(graph.predecessors(node))
        children = list(graph.successors(node))
        print(f"Node {node}:")
        print(f"  text: {graph.nodes[node]['text'].lower()}")
        print(f"  deprel: {graph.nodes[node]['deprel']}")
        print(f"  Parent: {parent}")
        print(f"  Children: {children}\n")


# next step: cut the trees and keep the ones with headwords
# join the trees for the same headword

# %%
print_tree_structure(trees[0])

# %%
# cut the trees: remove edge between headwords and their parents   
headword_trees = []
for tree in trees:
    for node in tree.nodes(data = True):
        
      if node[1].get('skill_headword') == True:
            parent = list(tree.predecessors(node[0]))
            # print(node)
            # print(parent)
            if parent == []:
                headword_trees.append(tree)
                # print("______*_______")
                continue

            temp = tree.copy()
            temp.remove_edge(parent[0], node[0])
            components = list(nx.weakly_connected_components(temp))
            subgraphs = [temp.subgraph(component).copy() for component in components]
            # print_tree_structure(subgraphs[0])
            # print("_______&________")
            # print_tree_structure(subgraphs[1])
            # print("_______&________")
            # print(list(tree.predecessors(node[0])))


            headword_trees.append(subgraphs[1])      

print_tree_structure(headword_trees[0])
print(len(trees))
print(len(headword_trees))

# %%
from networkx.readwrite import json_graph


def save_trees_to_file(trees, filename):
    data = []
    for tree in trees:
        tree_data = json_graph.node_link_data(tree)
        data.append(tree_data)
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

headwortrees = save_trees_to_file(headword_trees,'headword_trees.json')


# %%
# visualisation
# print(head_word_object[:50])
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

# Function to plot a tree graphically
def plot_tree(tree):
    pos = graphviz_layout(tree, prog="dot")
    labels = nx.get_node_attributes(tree, 'word_id')
    
    plt.figure(figsize=(12, 8))
    nx.draw(tree, pos, labels=labels, with_labels=True, node_size=5000, node_color='skyblue', font_size=10, font_weight='bold', edge_color='gray')
    plt.title("Tree Visualization")
    plt.show()

# Plot the tree
plot_tree(trees[0])
plot_tree(headword_trees[0])



# %%
# merging headword trees
from collections import defaultdict
from networkx.readwrite import json_graph

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
    print("Nodes of merged tree:", graph2.nodes(data=True))
    print("Edges of merged tree:", graph2.edges(data=True))
    print("________________________")
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
    print(roots)
    print(children)
    for child in children:
        weight = merged_graph.nodes[child].get('probability') 
        merged_graph.add_edge(roots[0],child ,weight = weight)
    for root in roots[1:]:
        merged_graph.remove_node(root)
    

    return merged_graph

def load_trees_from_file(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    trees = []
    for tree_data in data:
        tree = json_graph.node_link_graph(tree_data)
        trees.append(tree)
    return trees

# Load the trees from the file
headword_trees = load_trees_from_file('headword_trees.json')

# Group trees by their root 'text' property
trees_by_root_text = defaultdict(list)

for tree in headword_trees:
    root = find_root(tree)
    if root is not None:
        root_text = tree.nodes[root]['text'].lower()
        print(root_text, headword_trees.index(tree))
        trees_by_root_text[root_text].append(tree)

# Merge trees with the same root 'text'
pattern_trees = []

for root_text, tree_group in trees_by_root_text.items():
    merged_tree = merge_graphs(tree_group)
    if merged_tree is not None:
        pattern_trees.append(merged_tree)


headwortrees = save_trees_to_file(pattern_trees,'pattern_trees.json')
# for i in pattern_trees:
#     print_tree_structure(i)

print(len(pattern_trees))

# %%
print(len(headword_trees))
# %%
print(len(trees))
# %%
# algorithm 1

from networkx.readwrite import json_graph
import json
import networkx as nx
import gc

def find_root(graph):
    for node in graph.nodes:
        if graph.in_degree(node) == 0:
            return node
    return None

# Function to load a list of trees from a file
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

with open('index_v2.json', 'r') as file:
   I = json.load(file)

T = {}
for term, value in I.items():
   T[term] = 0

def TreeSearch(i, D, id, t):
   for sent , tup_list in zip(i[0],i[1]):
      for tup in tup_list:
        t_prim = tup[0]
        d = tup[1]
        for m in list(D.successors(id)):
           if D.nodes[m].get('deprel') == d:
              w = D.get_edge_data(id, m)['weight']
              T[t_prim] += w
              TreeSearch(I.get(t_prim) ,D ,m, t_prim)



for h in H[0:6]:
   identifier = find_root(h)
   word = h.nodes[identifier].get('text')
   res = I.get(word)
#    if res == None:
#       continue
#    else:
#       sents = res[0]
#       tup = res[1]
#       TreeSearch(res, h ,identifier, word)
   if res:
      TreeSearch(res, h, identifier, word)

# for T v2
# for t, score in T.items():
#     len_t = len(I.get(t)[0])
#     score = round(score / len_t, 4)
#     T[t] = score

# with open('T_v2.json', 'w') as f:
#     json.dump(T, f, indent=4)

# with open('W.json', 'w') as f:
#     json.dump(W, f, indent=4)
gc.collect()

# %%
from networkx.readwrite import json_graph
import json
import networkx as nx

def write_all_trees_to_file(trees, file_path):
    with open(file_path, 'w') as file:
        for i, graph in enumerate(trees):
            file.write(f"Tree {i+1}:\n")
            for node in graph.nodes:
                parent = list(graph.predecessors(node))
                children = list(graph.successors(node))
                file.write(f"Node {node}:\n")
                file.write(f"  text: {graph.nodes[node]['text']}\n")
                file.write(f"  deprel: {graph.nodes[node]['deprel']}\n")
                file.write(f"  Parent: {parent}\n")
                file.write(f"  Children: {children}\n\n")
            file.write("\n\n")  # Separate trees with blank lines

def load_trees_from_file(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    trees = []
    for tree_data in data:
        tree = json_graph.node_link_graph(tree_data)
        trees.append(tree)
    return trees
# Example usage:
# write_tree_structure_to_file(graph, 'tree_structure.txt')
pattern_trees = load_trees_from_file('pattern_trees500.json')
write_all_trees_to_file(pattern_trees, 'pattern_tree_structure.txt')


# %%
dict1 = {}
for h in H:
    identifier = find_root(h)
    word = h.nodes[identifier].get('text')
    lll = list(h.successors(identifier))
    hs = []
    for s in lll:
        f = h.nodes[s].get('sentence_id')
        if f not in hs:
            hs.append(f)
    dict1[word]=hs

with open('sentence_number_for_each_headword.json', 'w') as f:
    json.dump(dict1, f, indent=4)

# %%
print(len(dict1))

# %%
trees_with_more_than_one_headword = {}
for i,tree in enumerate(trees,0):
    temp = []
    for node in tree.nodes(data=True):
        if node[1].get('skill_headword') == True:
            temp.append(node[0])
    if len(temp) > 1:
        trees_with_more_than_one_headword[str(i)] = temp
    
with open('trees_with_more_than_one_headword.json', 'w') as f:
    json.dump(trees_with_more_than_one_headword, f, indent=4)


# %%
counter = 0
numbers = list(range(1001))
for h in headword_trees:
    sd = h.nodes(data =True)
    b = ''
    # for i in sd:
    #     b = i[1].get('sentence_id')
    #     break
    ss = False
    for i in sd:
        ss= i[1].get('skill_headword')
        if ss == True:
            b = i[1].get('sentence_id')
            break

    if b in numbers:
        numbers.remove(b)
    if b == 322:
        counter +=1
print(counter)
print(len(numbers))
print("***************")
print(numbers)
# %%
print(len(trees_with_more_than_one_headword))
# %%

headwortrees = save_trees_to_file(trees,'trees.json')
# %%
# validation
from collections import defaultdict
import json
with open('labeled_dataset_token_info.json', 'r') as file:
   dataset = json.load(file)
validation_data = defaultdict(list)
numbers = [1, 2, 3, 10, 13, 18, 19, 20, 21, 22, 33, 37, 38, 39, 41, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 64, 65, 66, 67, 68, 69, 70, 71, 73, 74, 77, 78, 79, 80, 81, 89, 91, 94, 95, 96, 101, 103, 104, 107, 108, 110, 111, 112, 114, 115, 116, 117, 118, 119, 120, 121, 122, 125, 132, 133, 134, 135, 138, 139, 140, 141, 142, 143, 144, 148, 149, 153, 156, 157, 161, 162, 163, 164, 171, 173, 175, 176, 186, 187, 188, 190, 191, 196, 197, 199, 200, 201, 203, 204, 205, 206, 207, 210, 211, 212, 213, 215, 216, 217, 219, 220, 223, 224, 225, 227, 229, 230, 231, 234, 236, 237, 243, 246, 247, 248, 249, 251, 252, 261, 262, 263, 265, 266, 271, 276, 278, 279, 280, 281, 283, 284, 285, 286, 287, 288, 289, 292, 294, 295, 296, 297, 298, 300, 305, 325, 328, 340, 343, 345, 351, 352, 353, 354, 357, 361, 370, 375, 376, 380, 381, 386, 387, 388, 418, 420, 421, 422, 423, 424, 425, 429, 430, 431, 439, 444, 445, 447, 452, 454, 457, 458, 462, 472, 473, 474, 478, 481, 482, 485, 487, 491, 497, 502, 508, 510, 512, 518, 522, 524, 526, 527, 528, 529, 533, 542, 543, 546, 549, 550, 555, 559, 560, 565, 566, 571, 572, 573, 574, 575, 576, 577, 578, 579, 582, 587, 597, 600, 603, 606, 607, 608, 610, 612, 614, 616, 619, 620, 621, 622, 625, 628, 632, 637, 638, 639, 641, 647, 650, 651, 652, 658, 664, 665, 670, 671, 675, 676, 677, 679, 680, 681, 683, 685, 695, 699, 703, 706, 711, 712, 713, 714, 715, 716, 717, 721, 723, 724, 726, 732, 735, 744, 746, 747, 752, 754, 759, 768, 780, 782, 785, 790, 793, 794, 799, 804, 805, 806, 814, 815, 833, 840, 847, 850, 851, 852, 854, 855, 856, 857, 861, 862, 863, 865, 871, 874, 882, 885, 887, 888, 891, 895, 897, 899, 901, 909, 910, 914, 917, 920, 923, 932, 936, 948, 949, 951, 953, 957, 964, 965, 966, 967, 968, 977, 979, 988, 993, 998]
for entry in dataset:
    if entry['sentence_id'] in numbers:
        temp = []
        temp.append(entry['sentence_id'])
        temp.append(entry['word_id'])
        temp.append(entry['text'])
        temp.append(entry['skill_tag'])
        validation_data[str(entry['sentence_id'])].append(temp)

print(len(validation_data))
print(validation_data['10'])


# %%
import json
with open('T.json', 'r') as file:
   T = json.load(file)
for entry,items in validation_data.items():
    for item in items:
        if item[2] in T.keys():
            score = T[item[2]]
            item.append(score)

with open('validation_data.json', 'w') as f:
    json.dump(validation_data, f, indent=4)



# %%
# validation v2
from collections import defaultdict
import json
with open('labeled_dataset_token_info.json', 'r') as file:
   dataset = json.load(file)
validation_data = defaultdict(list)
numbers = [1, 2, 3, 10, 13, 18, 19, 20, 21, 22, 33, 37, 38, 39, 41, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 64, 65, 66, 67, 68, 69, 70, 71, 73, 74, 77, 78, 79, 80, 81, 89, 91, 94, 95, 96, 101, 103, 104, 107, 108, 110, 111, 112, 114, 115, 116, 117, 118, 119, 120, 121, 122, 125, 132, 133, 134, 135, 138, 139, 140, 141, 142, 143, 144, 148, 149, 153, 156, 157, 161, 162, 163, 164, 171, 173, 175, 176, 186, 187, 188, 190, 191, 196, 197, 199, 200, 201, 203, 204, 205, 206, 207, 210, 211, 212, 213, 215, 216, 217, 219, 220, 223, 224, 225, 227, 229, 230, 231, 234, 236, 237, 243, 246, 247, 248, 249, 251, 252, 261, 262, 263, 265, 266, 271, 276, 278, 279, 280, 281, 283, 284, 285, 286, 287, 288, 289, 292, 294, 295, 296, 297, 298, 300, 305, 325, 328, 340, 343, 345, 351, 352, 353, 354, 357, 361, 370, 375, 376, 380, 381, 386, 387, 388, 418, 420, 421, 422, 423, 424, 425, 429, 430, 431, 439, 444, 445, 447, 452, 454, 457, 458, 462, 472, 473, 474, 478, 481, 482, 485, 487, 491, 497, 502, 508, 510, 512, 518, 522, 524, 526, 527, 528, 529, 533, 542, 543, 546, 549, 550, 555, 559, 560, 565, 566, 571, 572, 573, 574, 575, 576, 577, 578, 579, 582, 587, 597, 600, 603, 606, 607, 608, 610, 612, 614, 616, 619, 620, 621, 622, 625, 628, 632, 637, 638, 639, 641, 647, 650, 651, 652, 658, 664, 665, 670, 671, 675, 676, 677, 679, 680, 681, 683, 685, 695, 699, 703, 706, 711, 712, 713, 714, 715, 716, 717, 721, 723, 724, 726, 732, 735, 744, 746, 747, 752, 754, 759, 768, 780, 782, 785, 790, 793, 794, 799, 804, 805, 806, 814, 815, 833, 840, 847, 850, 851, 852, 854, 855, 856, 857, 861, 862, 863, 865, 871, 874, 882, 885, 887, 888, 891, 895, 897, 899, 901, 909, 910, 914, 917, 920, 923, 932, 936, 948, 949, 951, 953, 957, 964, 965, 966, 967, 968, 977, 979, 988, 993, 998]
for entry in dataset:
    if entry['sentence_id'] in numbers:
        temp = {}
        temp['sentence_id'] = entry['sentence_id']
        temp['word_id'] = entry['word_id']
        temp['word'] = entry['text']
        temp['skill_tag'] = entry['skill_tag']
        temp['score'] = 0
        validation_data[str(entry['sentence_id'])].append(temp)

with open('T_v2_2.json', 'r') as file:
   T = json.load(file)

words_not_in_T = 0
words_in_T = 0
for entry,items in validation_data.items():
    for item in items:
        if item['word'] in T.keys():
            score = T[item['word']]
            item['score'] = score
            words_in_T +=1
        else:
            words_not_in_T +=1

print(words_in_T)
print(words_not_in_T)
print(len(T))

with open('validation_data_v2_2.json', 'w') as f:
    json.dump(validation_data, f, indent=4)
# %%
with open('validation_data_v2_2.json', 'r') as file:
   validation_data = json.load(file)

true_positive =0
true_negative =0
false_positive = 0
false_negative =0
# max_score = 54.7748
max_score = 128.1132
# threshold = round(max_score/2, 2)
threshold = 5
print(str(threshold))

for entry,items in validation_data.items():
    for item in items:
        # if item['score'] > max_score:
        #     max_score = item['score']
        tag = item['skill_tag']
        score = item['score']
        if ((tag == 'I' or tag == 'B') and score >= threshold) :
            true_positive +=1
        if ((tag == 'I' or tag == 'B') and score < threshold) :
            false_negative +=1
        if ((tag == 'O') and score < threshold) :
            true_negative +=1
        if ((tag == 'O') and score >= threshold) :
            false_positive +=1

precision = round( true_positive/(true_positive+false_positive),4) # 0.6087
recall = round(true_positive/(true_positive+false_negative),4) # 0.0054

print(str(precision))
print(str(recall))
# print(max_score)

# %%
print(str(true_positive))
print(str(false_positive))
print(str(false_negative))
print(str(true_negative))

# %%
with open('index_v2.json', 'r') as file:
   I = json.load(file)

print(len(I))
