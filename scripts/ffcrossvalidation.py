# %%
# 4 for making pattern trees, 1 for validation
import json
import numpy as np
from collections import defaultdict
import networkx as nx
from networkx.readwrite import json_graph
with open('labeled_dataset_token_info.json', 'r') as file:
   data = json.load(file)

new = defaultdict(list)
for d in data:
   new[d['sentence_id']].append(d)
new_list = []

for n, lst in new.items():
    new_list.append(lst)
train = new_list 
with open('labeled_dataset_token_info_dev.json', 'r') as file:
   data = json.load(file)

new = defaultdict(list)
for d in data:
   new[d['sentence_id']].append(d)
new_list = []

for n, lst in new.items():
    new_list.append(lst)  
validation = new_list
length_of_each_part = len(new_list) // 5
# fold 1
# train = new_list[:length_of_each_part*4]
# validation = new_list[length_of_each_part*4:]
# fold 2
# train = new_list[:length_of_each_part*3] 
# validation = new_list[length_of_each_part*3:length_of_each_part*4]
# train.extend(new_list[length_of_each_part*4:])
# fold 3
# train = new_list[:length_of_each_part*2]
# validation = new_list[length_of_each_part*2:length_of_each_part*3]
# train.extend(new_list[length_of_each_part*3:])
# fold 4
# train = new_list[:length_of_each_part]
# validation = new_list[length_of_each_part:length_of_each_part*2]
# train.extend(new_list[length_of_each_part*2:])
# fold 5
# validation = new_list[:length_of_each_part]
# train = new_list[length_of_each_part:]

def make_pattern_tree(train):
    import networkx as nx
    import json
    from collections import defaultdict
    from networkx.readwrite import json_graph

    trees = []
    for words in train:
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

    return(pattern_trees)

pattern_trees = make_pattern_tree(train)

# def make_Ptree(pattern_trees):
#     def find_root(graph):
#         for node in graph.nodes:
#             if graph.in_degree(node) == 0:
#                 return node
#         return None
#     P_trees = []
#     for tree in pattern_trees:

#         G1 = nx.DiGraph()
#         def make_pattern_tree_root(node,nodeid):
#             temp = defaultdict(lambda: [0, 0])
        
#             children = list(tree.successors(node))
#             # print(len(children))
#             for child in children:
#                 dep = tree.nodes(data = True)[child]['deprel']
#                 tag = tree.nodes(data = True)[child]['skill_tag']
#                 temp[dep][0] += 1
#                 if tag == 'B' or tag == 'I':
#                     temp[dep][1]+=1
#             final = {}
#             for depen, tup in temp.items():
#                     if tup[1] != 0:
#                         final[depen] = tup[1]/tup[0]
#             root_node = nodeid 
#             for dep, weight in final.items():
#                 G1.add_node(nodeid+1, name= dep, freq = weight)
#                 G1.add_edge(root_node,nodeid+1)
#                 nodeid+=1
#             # for child in children:
#                 # make_pattern_tree(child, nodeid)
#             return nodeid



#         root = find_root(tree)
#         name1 = tree.nodes(data = True)[root]['text']

#         G1.add_node(1, name = name1)

#         next_node = make_pattern_tree_root(root,1) + 1
    
#         leaf_nodes = [n for n, d in tree.out_degree() if d == 0]
#         all_paths = []
#         for leaf in leaf_nodes:
#             paths = list(nx.all_simple_paths(tree, root, leaf))
#             all_paths.extend(paths)
#         all_paths_names = [[tree.nodes[node]['deprel'] for node in path] for path in all_paths]
#         for path in all_paths_names:
#             path[0] = 'root'

#         def similar_lists(list1,list2,level):
#             for i in range(level):
#                 if list1[i] != list2[i]:
#                     return False
    
#             return True
#         stat = True
#         children = ['a']
#         leaf_nodess = []
#         while stat:
#             if leaf_nodess == []:
#                 leaf_nodess = [n for n, d in G1.out_degree() if d == 0]
#             else:
#                 if leaf_nodess == [n for n, d in G1.out_degree() if d == 0]:
#                     stat = False
#                 else:
#                     leaf_nodess = [n for n, d in G1.out_degree() if d == 0]
#             all_pathss = []
#             for leaf in leaf_nodess:
#                 paths = list(nx.all_simple_paths(G1, 1, leaf))
#                 all_pathss.extend(paths)
#             # all_paths_namess = [[G1.nodes[node]['name'] for node in path] for path in all_pathss]
#             all_paths_namess = []
#             for p in all_pathss:
#                 temp = []
#                 for i in p:
#                     temp.append(G1.nodes[i]['name'])
#                 all_paths_namess.append(temp)
#             # print('*'+ str(len(all_paths_namess)))
#             for path in all_paths_namess:
#                 path[0] = 'root'
#             for path in all_paths_namess:
#                 level = len(path) - 1
#                 pathsx = nx.single_source_shortest_path_length(tree, root)
#                 nodes_at_level = [node for node, distance in pathsx.items() if distance == level]
#                 children = []
#                 for node in nodes_at_level:
#                     children.extend(list(tree.successors(node)))
#                 if children == []:
#                     stat = False
#                 # print('**'+str(len(nodes_at_level)))
#                 # print(children)
#                 children_dep = []
#                 for child in children:
#                     dep_name = tree.nodes[child]['deprel']
#                     if dep_name not in children_dep:
#                         children_dep.append(dep_name)

#                 test_paths = []
#                 for i in range(len(children_dep)):
#                     test_paths.append( path.copy() + [children_dep[i]])
#                 compare_res = defaultdict(lambda: [0, 0])
#                 # print(test_paths)
#                 for tree_path in all_paths_names:
#                     if len(tree_path) >= level+2:
#                         for test_path in test_paths:
#                             res = similar_lists(tree_path,test_path,level+2)
#                             if res:
#                                 compare_res[test_path[level+1]][0] += 1
#                                 x = all_paths[all_paths_names.index(tree_path)][level+1]
#                                 tag = tree.nodes[x]['skill_tag']
#                                 if tag== 'B'or tag == 'I':
#                                     compare_res[test_path[level+1]][1] += 1
#                 # print(compare_res)
#                 # making the tree
#                 parent = all_pathss[all_paths_namess.index(path)][-1]
#                 for dep, tup in compare_res.items():
#                     if tup[1] != 0:
#                         weight = tup[1]/tup[0]
#                         G1.add_node(next_node, name= dep, freq = weight)
#                         G1.add_edge(parent,next_node)
#                         next_node +=1

            
                     
                     


#         P_trees.append(G1)
#     return P_trees

# P_trees = make_Ptree(pattern_trees)

def algorithm1(H):
    def find_root(graph):
        for node in graph.nodes:
            if graph.in_degree(node) == 0:
                return node
        return None
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
                if D.nodes[m].get('deprel') == d:
                    w = D.get_edge_data(id, m)['weight']
                    T[t_prim] += w
                    tag = D.nodes[m].get('skill_tag')
                    children = I.get(t_prim)[0].index(i)
                    TreeSearch(i ,I.get(t_prim)[1][children] ,D ,m, t_prim)



    for h in H:
        identifier = find_root(h)
        word = h.nodes[identifier].get('text')
        res = I.get(word)
        if res:
            for i,dependency in zip(res[0],res[1]):
                TreeSearch(i,dependency, h, identifier, word)

    # for T v2
    for t, score in T.items():
        len_t = len(I.get(t)[0])
        score = round(score / len_t, 4)
        T[t] = score

    return T

T = algorithm1(pattern_trees)

# def algorithm1Ptree(H):
#     def find_root(graph):
#         for node in graph.nodes:
#             if graph.in_degree(node) == 0:
#                 return node
#         return None
#     with open('index_v2.json', 'r') as file:
#         I = json.load(file)

#     T = {}
#     for term, value in I.items():
#         T[term] = 0

#     def TreeSearch(i, tup_list, D, id, t):
#       for tup in tup_list:
#         t_prim = tup[0]
#         d = tup[1]

#         for m in list(D.successors(id)):
#            if D.nodes[m].get('name') == d:
#               w = D.nodes[m].get('freq')
#               T[t_prim] += w
#               children = I.get(t_prim)[0].index(i)
#               TreeSearch(i ,I.get(t_prim)[1][children] ,D ,m, t_prim)



#     for h in H:
#         identifier = find_root(h)
#         word = h.nodes[identifier].get('name')
#         res = I.get(word)
#         if res:
#             for i,dependency in zip(res[0],res[1]):
#                 TreeSearch(i,dependency, h, identifier, word)

#     # for T v2
#     for t, score in T.items():
#         len_t = len(I.get(t)[0])
#         score = round(score / len_t, 4)
#         T[t] = score
#     return T

# T = algorithm1Ptree(P_trees)

def make_new_validation(validation):
    words_not_in_T = 0
    words_in_T = 0
    new_validation= defaultdict(list)

    for items in validation:
        for item in items:
            if item['text'] in T.keys():
                score = T[item['text']]
                item['score'] = score
                new_validation[str(item['sentence_id'])].append(item)
                words_in_T +=1
            else:
                words_not_in_T +=1

    print(words_in_T) # 6435
    print(words_not_in_T) # 470 
    return new_validation

validation = make_new_validation(validation)

# %%
import json
from collections import defaultdict


def group_terms(terms, window_size):
    grouped_terms = []
    for i, term in enumerate(terms):
        if term['text'] == '.':
            continue

        temp = []
        tags = []
        scores = 0
        temp.append(terms[i]['text'])
        tags.append(terms[i]['skill_tag'])
        scores += terms[i]['score']
        for j in range(1, window_size):
            if i + j < len(terms):
                if terms[i + j]['text']== '.':
                    continue
                temp.append(terms[i + j]['text'])
                tags.append(terms[i+j]['skill_tag'])
                scores += terms[i+j]['score']
        if len(temp) == window_size:
            grouped_terms.append({'phrase': temp, 'tags': tags, 'score': scores/len(temp)})
           
    return grouped_terms

x = defaultdict(list)
for sent,words in validation.items():
   
   for i in range(2,6):
      grouped_terms = group_terms(words, i)
      x[sent].extend(grouped_terms)

# print(x['1'])

# validation test 2

new = []

for i,j in x.items():
    for b in j:
        if not ('O'in b['tags'] and 'B' in b['tags']):
            if not ('O' in b['tags'] and 'I' in b['tags']):
                # print(b)
                new.append(b)
max_f1 = 0
phrase_max_score = 0
sorted_list_desc = sorted(new, key=lambda a: a['score'], reverse=True)
sum_skill = 0
number_skill = 0
sum_non_skill = 0
number_non_skill = 0

for i in sorted_list_desc:
    if 'O' in i['tags']:
        sum_non_skill += i['score']
        number_non_skill += 1
    else:
        sum_skill += i['score']
        number_skill +=1
        
skill_word_average_score = sum_skill/number_skill
non_skill_word_average_score = sum_non_skill/number_non_skill
print(skill_word_average_score) # 0.2036031954117182
print(non_skill_word_average_score) # 0.10644162035683126
var_d_p = 0
var_d_n = 0
for i in sorted_list_desc:
    if 'O' in i['tags']:
        var_d_n += (i['score'] - non_skill_word_average_score)*(i['score'] - non_skill_word_average_score)   
    else:
        var_d_p += (i['score'] - skill_word_average_score)*(i['score'] - skill_word_average_score)
var_p = var_d_p/sum_skill
var_n = var_d_n/sum_non_skill
print(var_p)
print(var_n)
print((skill_word_average_score-non_skill_word_average_score)*(skill_word_average_score-non_skill_word_average_score)/(var_n+var_p))
relevant = 0
for entry in sorted_list_desc:
        if 'I'in entry['tags'] or 'B'in entry['tags']:
            relevant += 1
retrieved = 0
relevant_retrieved = 0
# for x in sorted_list_desc:
#     if not('O' in x['tags']):
#         relevant_retrieved +=1
#     if relevant_retrieved == relevant//10*1 :
#         retrieved = sorted_list_desc.index(x)+1
#         print('10')
#         print(relevant*0.1/retrieved)
#         print(retrieved)
#         print(relevant//10*1)
#     if relevant_retrieved == relevant//10*2 :
#         retrieved = sorted_list_desc.index(x)+1
#         print('20')
#         print(relevant*0.2/retrieved)
#         print(retrieved)
#     if relevant_retrieved == relevant//10*3 :
#         retrieved = sorted_list_desc.index(x)+1
#         print('30')
#         print(relevant*0.3/retrieved)
#         print(retrieved)
#     if relevant_retrieved == relevant//10*4 :
#         retrieved = sorted_list_desc.index(x)+1
#         print('40')
#         print(relevant*0.4/retrieved)
#     if relevant_retrieved == relevant//10*5 :
#         retrieved = sorted_list_desc.index(x)+1
#         print('50')
#         print(relevant*0.5/retrieved)
#     if relevant_retrieved == relevant//10*6 :
#         retrieved = sorted_list_desc.index(x)+1
#         print('60')
#         print(relevant*0.6/retrieved)
#     if relevant_retrieved == relevant//10*7 :
#         retrieved = sorted_list_desc.index(x)+1
#         print('70')
#         print(relevant*0.7/retrieved)
#     if relevant_retrieved == relevant//10*8 :
#         retrieved = sorted_list_desc.index(x)+1
#         print('80')
#         print(relevant*0.8/retrieved)
#     if relevant_retrieved == relevant//10*9 :
#         retrieved = sorted_list_desc.index(x)+1
#         print('90')
#         print(relevant*0.9/retrieved)
#     if relevant_retrieved == relevant//10*10 :
#         retrieved = sorted_list_desc.index(x)+1
#         print('100')
#         print(relevant*1/retrieved)
#         print(retrieved)
# for th in [0.1,0.05,0.2,2,0.07,0.08,0.09]:
#     true_positive =0
#     true_positive_ls = []
#     true_negative =0
#     false_positive = 0
#     false_negative =0
#     false_negative_ls = []
#     for ph in sorted_list_desc:
#         if ph['score'] > phrase_max_score:
#             phrase_max_score = ph['score']
#         if not('O' in ph['tags']):
#             if ph['score'] >= th:
#                 true_positive +=1
#             if ph['score'] < th:
#                 false_negative +=1
#         if 'O' in ph['tags']:
#             if ph['score'] >= th:
#                 false_positive +=1
#             if  ph['score'] < th:
#                 true_negative +=1
#     precision = round( true_positive/(true_positive+false_positive),4)
#     recall = round(true_positive/(true_positive+false_negative),4)
#     f1 = 2*(precision*recall)/(precision+recall)
#     print(th)
#     print(f1)
#     print(precision)
#     print(recall)
#     print()
# with open('validation_data_phrases_v4_fold5.json', 'w') as f:
#     json.dump(sorted_list_desc, f, indent=4)

print(phrase_max_score)
# %%
max_f1 = 0
max_th = 0
start = 0
end = 11.5
step = 0.1

current = start
for num in np.arange(0, 2.5, 0.01):
    true_positive =0
    true_negative =0
    false_positive = 0
    false_negative =0
    for ph in sorted_list_desc:
        if not('O' in ph['tags']):
            if ph['score'] >= num:
                true_positive +=1
            if ph['score'] < num:
                false_negative +=1
        if 'O' in ph['tags']:
            if ph['score'] >= num:
                false_positive +=1
            if  ph['score'] < num:
                true_negative +=1
    if true_positive == 0 and false_positive == 0:
        continue
    if false_negative == 0 and true_positive== 0 :
        continue
    precision = round( true_positive/(true_positive+false_positive),4)
    recall = round(true_positive/(true_positive+false_negative),4)
    if precision==0 or recall==0:
        continue
    f1 = 2*(precision*recall)/(precision+recall)
     
    if f1 > max_f1:
        max_f1 = f1
        max_th = num
    # current += step
    if num == 0.04:
        print('0.04 '+str(f1))
    if num == 0.24:
        print('0.24 '+str(f1))
    if num == 0.36:
        print('0.36 '+ str(f1))
    if num == 0.19:
        print('0.19 '+ str(f1))
    if num == 0.17:
        print('0.17 '+ str(f1))
print('max f1 '+ str(max_f1))
print('max th '+ str(max_th))
# %%
true_positive =0
true_positive_ls = []
true_negative =0
false_positive = 0
false_negative =0
false_negative_ls = []
max_score = 0
# threshold = round(max_score/2, 2)
threshold = 0.8
print(str(threshold))
sinle_max_f1 = 0
single_max_th = 0
for th in np.arange(0, 2.0, 0.01):

    for entry,items in validation.items():
        for item in items:
            if item['score'] > max_score:
                max_score = item['score']
            tag = item['skill_tag']
            score = item['score']
            if ((tag == 'I' or tag == 'B') and score >= th) :
                true_positive +=1
                true_positive_ls.append(item)
            if ((tag == 'I' or tag == 'B') and score < th) :
                false_negative +=1
                false_negative_ls.append(item)
            if ((tag == 'O') and score < th) :
                true_negative +=1
            if ((tag == 'O') and score >= th) :
                false_positive +=1

    precision = round( true_positive/(true_positive+false_positive),4)
    recall = round(true_positive/(true_positive+false_negative),4)
    f1 = 2*(precision*recall)/(precision+recall)
    if f1 > sinle_max_f1:
        sinle_max_f1 = f1
        single_max_th = th
    if th == 0.28 :
        print('0.28 '+str(f1))
    if th == 0.3:
        print('0.3 '+str(f1))
    if th == 0.38:
        print('0.38 '+str(f1))
    if th == 0.17:
        print('0.17 '+str(f1))
    # if th == 0.44:
    #     print('0.44 '+str(f1))
    # print(th)
    # print(f1)
print('f1 '+ str(sinle_max_f1))
print('th '+str(single_max_th))
print()
# print(str(precision))
# print(str(recall))
# print(str(true_positive))
# print(str(false_positive))
# print(str(false_negative))
# print(str(true_negative))
# print(str(max_score))
# print(true_positive_ls)
# print(false_negative_ls)

relevant = 0
for entry,items in validation.items():
    for item in items:
        tag = item['skill_tag']
        if tag == 'B' or tag == 'I':
            relevant +=1
print(relevant)
print(relevant*0.1) # 244

new = []
for i,j in validation.items():
    for x in j:
        new.append(x)
sorted_list_desc = sorted(new, key=lambda x: x['score'], reverse=True)
retrieved = 0
relevant_retrieved = 0
# for x in sorted_list_desc:
#     if x['skill_tag'] == 'B' or x['skill_tag'] == 'I':
#         relevant_retrieved +=1
#     if relevant_retrieved == relevant//10*1 :
#         retrieved = sorted_list_desc.index(x)+1
#         print('10')
#         print(relevant*0.1/retrieved)
#         print(retrieved)
#         print(relevant//10*1)
#     if relevant_retrieved == relevant//10*2 :
#         retrieved = sorted_list_desc.index(x)+1
#         print('20')
#         print(relevant*0.2/retrieved)
#         print(retrieved)
#     if relevant_retrieved == relevant//10*3 :
#         retrieved = sorted_list_desc.index(x)+1
#         print('30')
#         print(relevant*0.3/retrieved)
#         print(retrieved)
#     if relevant_retrieved == relevant//10*4 :
#         retrieved = sorted_list_desc.index(x)+1
#         print('40')
#         print(relevant*0.4/retrieved)
#     if relevant_retrieved == relevant//10*5 :
#         retrieved = sorted_list_desc.index(x)+1
#         print('50')
#         print(relevant*0.5/retrieved)
#     if relevant_retrieved == relevant//10*6 :
#         retrieved = sorted_list_desc.index(x)+1
#         print('60')
#         print(relevant*0.6/retrieved)
#     if relevant_retrieved == relevant//10*7 :
#         retrieved = sorted_list_desc.index(x)+1
#         print('70')
#         print(relevant*0.7/retrieved)
#     if relevant_retrieved == relevant//10*8 :
#         retrieved = sorted_list_desc.index(x)+1
#         print('80')
#         print(relevant*0.8/retrieved)
#     if relevant_retrieved == relevant//10*9 :
#         retrieved = sorted_list_desc.index(x)+1
#         print('90')
#         print(relevant*0.9/retrieved)
#     if relevant_retrieved == relevant//10*10 :
#         retrieved = sorted_list_desc.index(x)+1
#         print('100')
#         print(relevant*1/retrieved)
#         print(retrieved)


# precisionAt10percentRecall = relevant*0.1/retrieved
# print(precisionAt10percentRecall) # 0.746177370030581
# print(retrieved)

sum_skill = 0
number_skill = 0
sum_non_skill = 0
number_non_skill = 0
for i in sorted_list_desc:
    if i['skill_tag'] == 'I' or i['skill_tag'] == 'B':
        sum_skill += i['score']
        number_skill +=1
    else:
        sum_non_skill += i['score']
        number_non_skill += 1

skill_word_average_score = sum_skill/number_skill
non_skill_word_average_score = sum_non_skill/number_non_skill
print(skill_word_average_score) # 0.2036031954117182
print(non_skill_word_average_score) # 0.10644162035683126
var_d_p = 0
var_d_n = 0
for i in sorted_list_desc:
    if i['skill_tag'] == 'I' or i['skill_tag'] == 'B':
        var_d_p += (i['score'] - skill_word_average_score)*(i['score'] - skill_word_average_score)
    else:
        var_d_n += (i['score'] - non_skill_word_average_score)*(i['score'] - non_skill_word_average_score)   
var_p = var_d_p/sum_skill
var_n = var_d_n/sum_non_skill
print(var_p)
print(var_n)
print((skill_word_average_score-non_skill_word_average_score)*(skill_word_average_score-non_skill_word_average_score)/(var_n+var_p))
'''
V 2-3
        | threshold | precision | recall |  TP  | FP  | FN  | TN   | total  | max score
fold 1      0.3        0.4709      0.5458  1132  1272   942   3089   6435      20.5464
            0.35       0.5166      0.4354  903   845    1171  3516
            0.4        0.5218      0.3987  827   758    1247  3603
            0.45       0.5257      0.3843  797   719    1277  3642
            0.5        0.5158      0.3462  718   674    1356  3687
fold 2      0.3        0.4522      0.532   965   1169   849   2437   5420      127.2825
            0.35       0.4954      0.4405  799   814    1015  2792
            0.4        0.499       0.4019  729   732    1085  2874
            0.45       0.4978      0.3798  689   695    1125  2911
            0.5        0.4984      0.3352  608   612    1206  2994
fold 3      0.3        0.3592      0.4429  1023  1825   1287  5876   10011     31.8331
            0.35       0.3674      0.4065  939   1617   1371  6084
            0.4        0.3599      0.3697  854   1519   1456  6182
            0.45       0.356       0.3394  784   1418   1526  6283
            0.5        0.3498      0.3061  707   1314   1603  6387 
fold 4      0.3        0.4142      0.5448  1082  1530   904   3092   6608      127.8609
            0.35       0.4461      0.4502  879   1110   1092  3512
            0.4        0.4498      0.4149  824   1008   1162  3614
            0.45       0.454       0.3877  770   926    1216  3696
            0.5        0.4567      0.3691  733   872    1253  3750
fold 5      0.3        0.578       0.5918  793   579    547   1112   3031       127.6956
            0.35       0.6071      0.4694  629   407    711   1284
            0.4        0.6091      0.4396  589   378    751   1313
            0.45       0.621       0.4157  557   340    783   1351
            0.5        0.618       0.3888  521   322    819   1369
'''

'''
V 4
        | threshold | precision | recall | TP   |  FP  |  FN   |  TN  |  total   |  max score | relevant  | p10%r
fold 1      0.1         0.5159    0.5251  1089    1022   985     3339    6435        15.3906     207        0.5575268817204301
            0.09        0.4795    0.6143  1274    1383   800     2978
            0.12        0.5158    0.4571  948     890    1126    3471
            0.11        0.5131    0.4913  1019    967    1055    3394
fold 2      0.1         0.4972    0.5419  983     994    831     2612    5420        15.3796
            0.09        0.4784    0.5617  1019    1111   795     2495
            0.12        0.5049    0.4873  884     867    930     2739
            0.11        0.5013    0.5204  944     939    870     2667
fold 3      0.1         0.3547    0.5316  1228    2234   1082    5467    10011       1.8167
            0.09        0.3515    0.5619  1298    2395   1012    5306
            0.12        0.3671    0.4874  1126    1941   1184    5760
            0.11        0.3644    0.5126  1184    2065   1126    5636
fold 4      0.1         0.4286    0.6193  1230    1640   756     2982    6608        15.3969
            0.09        0.4151    0.6324  1256    1770   730     2852
            0.12        0.4543    0.5176  1028    1235   958     3387
            0.11        0.4471    0.5317  1056    1306   930     3316
fold 5      0.1         0.6235    0.597   800     483    540     1208    3031        11.7131     134       0.7570621468926554
            0.09        0.5862    0.6851  918     648    422     1043
            0.12        0.6268    0.5478  734     437    606     1254
            0.11        0.6299    0.5754  771     453    569     1238

'''
'''
fold 5
recall |  relevant* recall  |   retrived | precision 
 10            134                 177     0.7570621468926554
 20                                        0.6261682242990654
 30                                        0.6452648475120385
 40                                        0.6434573829531812
 50                                        0.61865189289012
 60                                        0.6208494208494209
 70                                        0.5811648079306071
 80                                        0.5447154471544715
 90                                        0.4878640776699029
 100                                       0.44224422442244227
fold 1
 10                                        0.5575268817204301
 20                                        0.543643512450852
 30                                        0.5377700950734658
 40                                        0.5267301587301587
 50                                        0.5159203980099503
 60                                        0.48799999999999993
 70                                        0.4552524302289119
 80                                        0.4221882951653944
 90                                        0.3723518850987433
 100                              6428     0.3226509023024269
fold 2
 10        181.4                           0.5496969696969697
 20                                        0.5160739687055477               
 30                                        0.48982898289828974
 40                                        0.5021453287197232
 50                                        0.5058561070831009
 60                                        0.47198612315698174
 70                                        0.4375603032391454
 80                                        0.3946695675822682
 90                                        0.37376373626373627
 100                                       0.3371747211895911
'''

# %%
