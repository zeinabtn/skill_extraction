# import json
# with open('validation_data_v2_3sorted.json', 'r') as file:
#    validation_data = json.load(file)

# sum_skill = 0
# number_skill = 0
# sum_non_skill = 0
# number_non_skill = 0
# for i in validation_data:
#     if i['skill_tag'] == 'I' or i['skill_tag'] == 'B':
#         sum_skill += i['score']
#         number_skill +=1
#     else:
#         sum_non_skill += i['score']
#         number_non_skill += 1

# skill_word_average_score = sum_skill/number_skill
# non_skill_word_average_score = sum_non_skill/number_non_skill
# print(skill_word_average_score) # 1.2690183531339685
# print(non_skill_word_average_score) # 0.47489403334308566

import json
with open('validation_data_v4sorted.json', 'r') as file:
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
print(skill_word_average_score) # 0.2036031954117182
print(non_skill_word_average_score) # 0.10644162035683126

'''
average score:
                | skill terms  |  non-skill terms
pattern tree v1 |  1.2690      |     0.4748
pattern tree v2 |  0.20360     |     0.10644

'''


