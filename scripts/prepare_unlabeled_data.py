import stanza
# stanza.download('en')
import json
data = []
for i in range(0,1000):
    with open('unlabeled/result/output_{}.json'.format(i), 'r', encoding='utf-8') as f:
        data.append(json.load(f))

text = []
for i in range(0,1000):
    text.append(data[i]["text"])
    # print(text[i])
    with open('unlabeled/final/output_{}.json'.format(i), 'w', encoding='utf-8') as f:
        json.dump({str(i) : text[i]}, f, ensure_ascii=False, indent=4)

nlp = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma,depparse', download_method=None)
docs = []

for i in text:
    docs.append(nlp(i))

# 


# Function to extract information and add sentence IDs
def extract_info_with_sentence_id(docs):
    combined_data = []
    sentence_id = 0
    
    for doc in docs:
        for sentence in doc.sentences:
            for word in sentence.words:
                info = {
                    'sentence_id': sentence_id,
                    'word_id': word.id,
                    'text': word.text,
                    'lemma': word.lemma,
                    'upos': word.upos,
                    'xpos': word.xpos,
                    'head': word.head,
                    'deprel': word.deprel
                }
                combined_data.append(info)
            sentence_id += 1
    
    return combined_data

# Extract information with sentence IDs
# data_with_ids = extract_info_with_sentence_id(docs)

# Print extracted data with sentence IDs
# print(json.dumps(data_with_ids, indent=4))

# File path
# file_path = 'parsed_output_with_ids.json'

# Save data to JSON file
# with open(file_path, 'w') as f:
    # json.dump(data_with_ids, f, indent=4)

# print(f"Data saved to {file_path}")

# 
# print(docs[0].sentences[0].words)
# roots =[]
# for doc in docs:
#     temp_roots = []
#     for j, sent in enumerate(doc.sentences):
#         children_dict = {word.id: [] for word in sent.words}

#         for word in sent.words:
#             head_id = word.head
#             if head_id == 0:
#                 temp_roots.append(word.id)
#             if head_id > 0: 
#                 children_dict[head_id].append(word)

#         for word in sent.words:
#             word.children = children_dict[word.id]

#     roots.append(temp_roots)


# for doc,rootss in zip(docs,roots):
#     for j, sent in enumerate(doc.sentences):

#         def dfs_postorder(root):
#             if root:
#                 for child in root.children:
#                     # print(child)
#                     dfs_postorder(child)
#                 print(str(root.id) + "->", end='')
#                 if not root.children: 
#                     root.freq = (root.skill_tag == "B" or root.skill_tag == "I") + 0
#                 else:
#                     freq1 = 0
#                     for child in root.children:
#                         freq1 += child.freq
#                     freq1 += (root.skill_tag == "B" or root.skill_tag == "I")
#                     root.freq = freq1
     
#         def probabilityCount(root):
#             if root.head == 0:
#                 root.prob = 0
#             else:
#                 parent = sent.words[root.head-1]
#                 p_freq = parent.freq - (parent.skill_tag == "B" or parent.skill_tag == "I")
#                 root.prob = root.freq / (p_freq + 1)
#             for child in root.children:
#                 probabilityCount(child)
    
#         root = sent.words[rootss[j]-1]
#         dfs_postorder(root)
#         probabilityCount(root)

# class WordinTree:
#     def __init__(self,word,id,prob,children,head,dependency) -> None:
#         self.word = word
#         self.id = id
#         self.prob = prob
#         self.children = children
#         self.head = head
#         self.dependency = dependency

# final_data = []
# count = 0
# for doc,rootss,i in zip(docs,roots,range(0,len(doc))):
#     for sent in doc.sentences:
#         nodes = []
#         for word in sent.words:
#             node = WordinTree(word.text ,word.id ,word.prob, word.children ,word.head ,word.deprel)
#             nodes.append(node)
#         final_data.append((count, nodes))
#         count +=1
        
# print(final_data)

# dependency_index = {}
# counter = 0

# for doc in docs:
#     for sent in doc.sentences:
#         for word in sent.words:
#             if word.text in dependency_index.keys():
#                 dependency_index[word.text] 




from collections import defaultdict

class JobAdIndex:
    def __init__(self):
        self.index = defaultdict(lambda: ([], []))

    # def add_term(self, term, sentence_id=None):
    #     if sentence_id is not None:
    #         self.index[term][0].append(sentence_id)
    #         self.index[term][1].append([])  # Initialize an empty list for children

    def add_dependency(self, term, sentence_id, dependency):
        try:
            index = self.index[term][0].index(sentence_id)
            self.index[term][1][index].append(dependency)
        except ValueError:
            raise ValueError(f"Sentence ID {sentence_id} not found for term '{term}'")
    
    def add_term(self, term, sentence_id=None):
        if sentence_id is not None:
            self.index[term][0].append(sentence_id)
            self.index[term][1].append([])

    # def add_dependency(self, term, dependency):
    #     self.index[term][1].append(dependency)
    
    def add_sentence(self, sentence_id, term, dependency):
        self.index[term][0].append(sentence_id)
        self.index[term][1].append(dependency)

    def lookup_dependencies(self, term):
        return self.index[term][1]
    
    def search_term(self, term):
        return term in self.index
    
    def search_sentence(self, term, sentence_id):
        return sentence_id in self.index[term][0]
    
    def write_index_to_file(self, file_name, format='json'):
        if format == 'json':
            with open(file_name, 'w') as file:
                json.dump(dict(self.index), file, indent=4)
        elif format == 'txt':
            with open(file_name, 'w') as file:
                for term, (sentence_ids, dependencies) in self.index.items():
                    file.write(f'Term: {term}, Sentence IDs: {sentence_ids}, Dependencies: {dependencies}\n')
        else:
            print("Unsupported file format. Please choose 'json' or 'txt'.")

# Example usage
dependency_index = JobAdIndex()
# job_index.add_sentence(1, ["job", "advertisement"], [("job", "NN"), ("advertisement", "NN")])
# job_index.add_sentence(2, ["term", "dependency"], [("term", "NN"), ("dependency", "NN")])

# print(job_index.lookup_dependencies("job"))
# print(job_index.lookup_dependencies("dependency")) 

counter = 0 

for doc in docs:
    for sent in doc.sentences:
        children_dict = {word.id: [] for word in sent.words}
        for word in sent.words:
            # if dependency_index.search_term(word.text):
                # head = ''
                # depen =''
                # for word2 in sent.words:
                #     if word.head == word2.id:
                #         head = word2.text
                #         depen = word.deprel
                #         break
                # dependency_index.add_sentence(counter, word.text, (head, depen))
            # else:
                # pass
                # make the dict item for a new word and update the children for other words
                if not dependency_index.search_term(word.text.lower()):
                    dependency_index.add_term(word.text.lower(), sentence_id= counter)
                else:
                    # check if the current counter is saved in first list for this word
                    if dependency_index.search_sentence(word.text.lower(), counter):
                        pass
                    else:
                      dependency_index.add_term(word.text.lower(), sentence_id= counter)  
                if word.head > 0 :
                    parent =  sent.words[word.head - 1].text.lower()
                    if not dependency_index.search_term(parent):
                        dependency_index.add_term(parent, sentence_id= counter)
                        dependency_index.add_dependency(parent, counter, (word.text.lower(), word.deprel))
                    else:
                        if dependency_index.search_sentence(parent, counter):
                            dependency_index.add_dependency(parent, counter, (word.text.lower(), word.deprel))
                        else:
                            dependency_index.add_term(parent, sentence_id= counter)
        

        counter +=1

dependency_index.write_index_to_file('index_v2.json', format='json')
