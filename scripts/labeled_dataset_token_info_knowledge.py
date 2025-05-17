# %%
import stanza
import json
with open('dataset labeled/train1.json', 'r') as file:
    data = json.load(file)

tokens_list = [entry['tokens'] for entry in data]
print(len(tokens_list))
tag_knowledge_list = [entry['tags_knowledge'] for entry in data]

stanza_input = []
stanza_knowledge_tags =[]
for i in range(0,4800):
  if "B" in tag_knowledge_list[i]:
    stanza_input.append(tokens_list[i])
    stanza_knowledge_tags.append(tag_knowledge_list[i])
print(len(stanza_input))
print(len(stanza_knowledge_tags))

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
    word.knowledge_tag = stanza_knowledge_tags[j][i]

print(doc.sentences[0].words)

head_words = []
head_word_object = []
for j, sent in enumerate(doc.sentences):
  for i, word in enumerate(doc.sentences[j].words):
    word.sentence_id = j
    word.knowledge_headword = False
    if word.knowledge_tag == "B" and word.head != 0 :
      x= doc.sentences[j].words[word.head - 1]
      if x.knowledge_tag == "O":
        head_words.append({'word':x.text, 'sentence id': j, 'word id': x.id})
        head_word_object.append(x)
        x.knowledge_headword = True

      else:
        while x.knowledge_tag != "O" and x.head != 0 :
          x = doc.sentences[j].words[x.head - 1]
        if x.knowledge_tag == "O":
          head_words.append({'word':x.text, 'sentence id': j, 'word id': x.id})
          head_word_object.append(x)
          x.knowledge_headword = True

# %%
# add children
roots =[]

for j, sent in enumerate(doc.sentences):
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
          root.freq = (root.knowledge_tag == "B" or root.knowledge_tag == "I") + 0
        else:
           freq1 = 0
           for child in root.children_word:
              freq1 += child.freq
           freq1 += (root.knowledge_tag == "B" or root.knowledge_tag == "I")
           root.freq = freq1
     
  def probabilityCount(root):
     if root.head == 0:
        root.prob = 0
     else:
        parent = sent.words[root.head-1]
        p_freq = parent.freq - (parent.knowledge_tag == "B" or parent.knowledge_tag == "I")
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
                    'skill_tag': word.knowledge_tag,
                    'skill_headword': word.knowledge_headword

                }
                combined_data.append(info)
   
with open('labeled_dataset_token_info_knowledge.json', 'w') as f:
    json.dump(combined_data, f, indent=4)

# %%
