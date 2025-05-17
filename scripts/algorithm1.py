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
H = load_trees_from_file('patterntrees_without_O.json')

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

with open('T_v3.json', 'w') as f:
    json.dump(T, f, indent=4)

# with open('W.json', 'w') as f:
#     json.dump(W, f, indent=4)
gc.collect()
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

with open('T_v3.json', 'r') as file:
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

with open('validation_data_v3.json', 'w') as f:
    json.dump(validation_data_v2_3, f, indent=4)
print(len(validation_data_v2_3))

# %%
import json
with open('validation_data_v3.json', 'r') as file:
   validation_data = json.load(file)

true_positive =0
true_negative =0
false_positive = 0
false_negative =0
# max_score = 54.7748
max_score = 128.1132
# threshold = round(max_score/2, 2)
threshold = 0.5
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

precision = round( true_positive/(true_positive+false_positive),4) # 0.7264
recall = round(true_positive/(true_positive+false_negative),4) # 0.0598

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
    0.5         0.61     0.4031       984             629              1457            2790
    0.4         0.60     0.4556       
    0.3       0.5572     0.5887
    0.5 v3    0.6236     0.3781       923             557              1518            2862
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
with open('validation_data_v2_3.json', 'r') as file:
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
print(precisionAt10percentRecall) # 0.746177370030581
    


# %%
with open('validation_data_v2_3sorted.json', 'w') as f:
    json.dump(sorted_list_desc, f, indent=4)
# %%
import json
with open('validation_data_v2_3sorted.json', 'r') as file:
   validation_data = json.load(file)

sum_skill = 0
number_skill = 0
sum_non_skill = 0
number_non_skill = 0
for i in validation_data:
    if i['skill_tag'] == 'I' or i['skill_tag'] == 'B':
        sum_skill += i['score']
        number_skill +=1
    else:
        sum_non_skill += i['score']
        number_non_skill += 1

skill_word_average_score = sum_skill/number_skill
non_skill_word_average_score = sum_non_skill/number_non_skill
print(skill_word_average_score)
print(non_skill_word_average_score)

# %%
