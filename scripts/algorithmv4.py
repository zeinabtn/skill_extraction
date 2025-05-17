# %%
import json
import networkx as nx
from networkx.readwrite import json_graph
from collections import defaultdict


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

# Function to load a list of trees from a file
def load_trees_from_file(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    trees = []
    for tree_data in data:
        tree = json_graph.node_link_graph(tree_data)
        trees.append(tree)
    return trees

headwords = load_trees_from_file('pattern_trees.json')

# %%

P_trees = []
for tree in headwords:

    G1 = nx.DiGraph()
    def make_pattern_tree_root(node,nodeid):
        temp = defaultdict(lambda: [0, 0])
        
        children = list(tree.successors(node))
        print(len(children))
        for child in children:
                dep = tree.nodes(data = True)[child]['deprel']
                tag = tree.nodes(data = True)[child]['skill_tag']
                temp[dep][0] += 1
                if tag == 'B' or tag == 'I':
                    temp[dep][1]+=1
        final = {}
        for depen, tup in temp.items():
                    if tup[1] != 0:
                        final[depen] = tup[1]/tup[0]
        root_node = nodeid 
        for dep, weight in final.items():
            G1.add_node(nodeid+1, name= dep, freq = weight)
            G1.add_edge(root_node,nodeid+1)
            nodeid+=1
        # for child in children:
            # make_pattern_tree(child, nodeid)
        return nodeid



    root = find_root(tree)
    name1 = tree.nodes(data = True)[root]['text']

    G1.add_node(1, name = name1)

    next_node = make_pattern_tree_root(root,1) + 1
    
    leaf_nodes = [n for n, d in tree.out_degree() if d == 0]
    all_paths = []
    for leaf in leaf_nodes:
        paths = list(nx.all_simple_paths(tree, root, leaf))
        all_paths.extend(paths)
    all_paths_names = [[tree.nodes[node]['deprel'] for node in path] for path in all_paths]
    for path in all_paths_names:
         path[0] = 'root'

    def similar_lists(list1,list2,level):
        for i in range(level):
            if list1[i] != list2[i]:
                return False
    
        return True
    stat = True
    children = ['a']
    leaf_nodess = []
    while stat:
        if leaf_nodess == []:
            leaf_nodess = [n for n, d in G1.out_degree() if d == 0]
        else:
            if leaf_nodess == [n for n, d in G1.out_degree() if d == 0]:
                stat = False
            else:
                leaf_nodess = [n for n, d in G1.out_degree() if d == 0]
        all_pathss = []
        for leaf in leaf_nodess:
            paths = list(nx.all_simple_paths(G1, 1, leaf))
            all_pathss.extend(paths)
        # all_paths_namess = [[G1.nodes[node]['name'] for node in path] for path in all_pathss]
        all_paths_namess = []
        for p in all_pathss:
            temp = []
            for i in p:
                temp.append(G1.nodes[i]['name'])
            all_paths_namess.append(temp)
        # print('*'+ str(len(all_paths_namess)))
        for path in all_paths_namess:
            path[0] = 'root'
        for path in all_paths_namess:
            level = len(path) - 1
            pathsx = nx.single_source_shortest_path_length(tree, root)
            nodes_at_level = [node for node, distance in pathsx.items() if distance == level]
            children = []
            for node in nodes_at_level:
                 children.extend(list(tree.successors(node)))
            if children == []:
                stat = False
            # print('**'+str(len(nodes_at_level)))
            # print(children)
            children_dep = []
            for child in children:
                 dep_name = tree.nodes[child]['deprel']
                 if dep_name not in children_dep:
                      children_dep.append(dep_name)

            test_paths = []
            for i in range(len(children_dep)):
                 test_paths.append( path.copy() + [children_dep[i]])
            compare_res = defaultdict(lambda: [0, 0])
            # print(test_paths)
            for tree_path in all_paths_names:
                if len(tree_path) >= level+2:
                     for test_path in test_paths:
                        res = similar_lists(tree_path,test_path,level+2)
                        if res:
                            compare_res[test_path[level+1]][0] += 1
                            x = all_paths[all_paths_names.index(tree_path)][level+1]
                            tag = tree.nodes[x]['skill_tag']
                            if tag== 'B'or tag == 'I':
                                compare_res[test_path[level+1]][1] += 1
            # print(compare_res)
            # making the tree
            parent = all_pathss[all_paths_namess.index(path)][-1]
            for dep, tup in compare_res.items():
                if tup[1] != 0:
                    weight = tup[1]/tup[0]
                    G1.add_node(next_node, name= dep, freq = weight)
                    G1.add_edge(parent,next_node)
                    next_node +=1
        # break
        # print(G1.nodes(data = True))
        # print(G1.edges())


            
                     
                     


    P_trees.append(G1)
    print(G1.nodes(data = True))
    print(G1.edges())
save_trees_to_file(P_trees,'P_trees.json')
# %%
# algorithm 1
H = load_trees_from_file('P_trees.json')

with open('index_v2.json', 'r') as file:
   I = json.load(file)

T = {}
for term, value in I.items():
   T[term] = 0

def TreeSearch(i, tup_list, D, id, t):
      for tup in tup_list:
        t_prim = tup[0]
        d = tup[1]

        for m in list(D.successors(id)):
           if D.nodes[m].get('name') == d:
              w = D.nodes[m].get('freq')
              T[t_prim] += w
              children = I.get(t_prim)[0].index(i)
              TreeSearch(i ,I.get(t_prim)[1][children] ,D ,m, t_prim)



for h in H:
   identifier = find_root(h)
   word = h.nodes[identifier].get('name')
   res = I.get(word)
   if res:
    for i,dependency in zip(res[0],res[1]):
        TreeSearch(i,dependency, h, identifier, word)

# for T v2
for t, score in T.items():
    len_t = len(I.get(t)[0])
    score = round(score / len_t, 4)
    T[t] = score

with open('T_v4.json', 'w') as f:
    json.dump(T, f, indent=4)

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

with open('T_v4.json', 'r') as file:
   T = json.load(file)

words_not_in_T = 0
words_in_T = 0
validation_data_v2_3 = defaultdict(list) # only validation words that are in T

for entry,items in validation_data.items():
    for item in items:
        if item['word'] in T.keys():
            score = T[item['word']]
            item['score'] = score
            validation_data_v2_3[entry].append(item)
            words_in_T +=1
        else:
            words_not_in_T +=1

print(words_in_T) # 5860
print(words_not_in_T) # 466
print(len(T)) # 28530

with open('validation_data_v4.json', 'w') as f:
    json.dump(validation_data_v2_3, f, indent=4)
print(len(validation_data_v2_3))


# %%
import json
with open('validation_data_v4.json', 'r') as file:
   validation_data = json.load(file)

true_positive =0
true_negative =0
false_positive = 0
false_negative =0
# max_score = 54.7748
max_score = 128.1132
# threshold = round(max_score/2, 2)
threshold = 0.1
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

precision = round( true_positive/(true_positive+false_positive),4)
recall = round(true_positive/(true_positive+false_negative),4)

print(str(precision))
print(str(recall))

# %%
print(str(true_positive)) # 154
print(str(false_positive)) # 58
print(str(false_negative)) # 2421
print(str(true_negative)) # 3693
# %%
'''
 threshold | precision | recall | true_positive | false_positive | false_negative | true_negative
    0.5         0.60     0.0528       129             84              2312            3335
    0.1         0.60     0.5621       1372            901             1069            2518
    0.09       0.5655    0.6575
'''
# %%
# recall 10%
relevant = 0
for entry,items in validation_data.items():
    for item in items:
        tag = item['skill_tag']
        if tag == 'B' or tag == 'I':
            relevant +=1

print(relevant)
print(relevant*0.1) # 244
# %%
import json
with open('validation_data_v4.json', 'r') as file:
   validation_data = json.load(file)

new = []
for i,j in validation_data.items():
    for x in j:
        new.append(x)

sorted_list_desc = sorted(new, key=lambda x: x['score'], reverse=True)
retrieved = 0
relevant_retrieved = 0
for x in sorted_list_desc:
    if x['skill_tag'] == 'B' or x['skill_tag'] == 'I':
        relevant_retrieved +=1
    if relevant_retrieved == 244:
        retrieved = sorted_list_desc.index(x)+1
        break

precisionAt10percentRecall = 244/retrieved
print(precisionAt10percentRecall) # 0.6815642458100558
    
with open('validation_data_v4sorted.json', 'w') as f:
    json.dump(sorted_list_desc, f, indent=4)
# %%
