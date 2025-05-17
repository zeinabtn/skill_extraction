# %%
import json
from collections import defaultdict


with open("validation_data_v4.json", 'r') as file:
    validation = json.load(file)


def group_terms(terms, window_size):
    grouped_terms = []
    for i, term in enumerate(terms):
        if term['word'] == '.':
            continue

        temp = []
        tags = []
        scores = 0
        temp.append(terms[i]['word'])
        tags.append(terms[i]['skill_tag'])
        scores += terms[i]['score']
        for j in range(1, window_size):
            if i + j < len(terms):
                if terms[i + j]['word']== '.':
                    continue
                temp.append(terms[i + j]['word'])
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

relevant = 18828
all = 0
for entry,items in x.items():
    for phrase in items:
        if 'O'in phrase['tags']:
            relevant -= 1
    # all += len(items)
# print(all)
print(relevant)
print(relevant*0.1) # 534

new = []

for i,j in x.items():
    for b in j:
        if not ('O'in b['tags'] and 'B' in b['tags']):
            if not ('O' in b['tags'] and 'I' in b['tags']):
                # print(b)
                new.append(b)
# %%
sorted_list_desc = sorted(new, key=lambda a: a['score'], reverse=True)
for th in [0.05,0.06,0.07,0.08,0.09]:
    true_positive =0
    true_positive_ls = []
    true_negative =0
    false_positive = 0
    false_negative =0
    false_negative_ls = []
    for ph in sorted_list_desc:
        if not('O' in ph['tags']):
            if ph['score'] >= th:
                true_positive +=1
            if ph['score'] < th:
                false_negative +=1
        if 'O' in ph['tags']:
            if ph['score'] >= th:
                false_positive +=1
            if  ph['score'] < th:
                true_negative +=1
    precision = round( true_positive/(true_positive+false_positive),4)
    recall = round(true_positive/(true_positive+false_negative),4)
    f1 = 2*(precision*recall)/(precision+recall)
    print(th)
    print(f1)
    print()
    # print(true_positive)
    # print(true_negative)
    # print(false_negative)
    # print(false_positive)


# retrieved = 0
# relevant_retrieved = 0
# # print(sorted_list_desc[0])
# for z in sorted_list_desc:
#     if 'O' not in z['tags']:
#         relevant_retrieved +=1
#     # if relevant_retrieved == 534:
#     #     retrieved = sorted_list_desc.index(z)+1
#     #     break
#     if relevant_retrieved == relevant//10*1 :
#         retrieved = sorted_list_desc.index(z)+1
#         print('10')
#         print(relevant*0.1/retrieved)
#         print(retrieved)
#         print(relevant//10*1)
#     if relevant_retrieved == relevant//10*2 :
#         retrieved = sorted_list_desc.index(z)+1
#         print('20')
#         print(relevant*0.2/retrieved)
#         print(retrieved)
#     if relevant_retrieved == relevant//10*3 :
#         retrieved = sorted_list_desc.index(z)+1
#         print('30')
#         print(relevant*0.3/retrieved)
#         print(retrieved)
#     if relevant_retrieved == relevant//10*4 :
#         retrieved = sorted_list_desc.index(z)+1
#         print('40')
#         print(relevant*0.4/retrieved)
#     if relevant_retrieved == relevant//10*5 :
#         retrieved = sorted_list_desc.index(z)+1
#         print('50')
#         print(relevant*0.5/retrieved)
#     if relevant_retrieved == relevant//10*6 :
#         retrieved = sorted_list_desc.index(z)+1
#         print('60')
#         print(relevant*0.6/retrieved)
#     if relevant_retrieved == relevant//10*7 :
#         retrieved = sorted_list_desc.index(z)+1
#         print('70')
#         print(relevant*0.7/retrieved)
#     if relevant_retrieved == relevant//10*8 :
#         retrieved = sorted_list_desc.index(z)+1
#         print('80')
#         print(relevant*0.8/retrieved)
#     if relevant_retrieved == relevant//10*9 :
#         retrieved = sorted_list_desc.index(z)+1
#         print('90')
#         print(relevant*0.9/retrieved)
#     if relevant_retrieved == relevant//10*10 :
#         retrieved = sorted_list_desc.index(z)+1
#         print('100')
#         print(relevant*1/retrieved)
#         print(retrieved)


# precisionAt10percentRecall = 534/retrieved
# print(precisionAt10percentRecall) # 0.355525965379494
# just o phrases and skill phrases: 0.5313432835820896 
# print(retrieved) # 1005

# with open('validation_data_phrases_v2_3.json', 'w') as f:
#     json.dump(sorted_list_desc, f, indent=4)

'''
        precision                precision(with mix B I O phrases)
v 2-3    0.6020293122886133        0.3912087912087912
v 4      0.5313432835820896        0.355525965379494
'''

# %%
