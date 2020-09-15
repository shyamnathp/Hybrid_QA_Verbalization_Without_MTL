words = []

with open('/data/premnadh/Hybrid-QASystem/sentence_transformers/Vocab/testVocab.txt', 'r') as file1:
    contents = file1.read()
    with open('/data/premnadh/Hybrid-QASystem/sentence_transformers/Vocab/newVocab.txt', 'r') as file2:  
        for line in file2:     
            for word in line.split():      
                if word not in contents:
                    words.append(word)

newVocab = open('/data/premnadh/Hybrid-QASystem/sentence_transformers/Vocab/extra_vocab.txt','w') 
for word in words:
    print(f'{word}')
    newVocab.write(f'{word}\n')

unique_words = set()
# vocabs = ['/data/premnadh/VQUANDA-Baseline-Models-1/src_vocab.txt','/data/premnadh/VQUANDA-Baseline-Models-1/src_vocab_query.txt','/data/premnadh/VQUANDA-Baseline-Models-1/trg_vocab.txt']

# for voc in vocabs:
#     with open(voc, 'r') as file1:
#         #contents = file1.read()
#         for line in file1:
#             for word in line.split():
#                 unique_words.add(word)

# newVocab = open('/data/premnadh/Hybrid-QASystem/sentence_transformers/Vocab/VQUANDA_Vocab.txt','w') 
# for word in unique_words:
#     print(f'{word}')
#     newVocab.write(f'{word}\n')

# with open('/data/premnadh/Hybrid-QASystem/sentence_transformers/Vocab/Vquanda_vocab_nochars.txt', 'r') as file1:
#     for line in file1:
#         for word in line.split():
#             unique_words.add(word)

# newVocab = open('/data/premnadh/Hybrid-QASystem/sentence_transformers/Vocab/vquanda_vocab_unique.txt','w') 
# for word in unique_words:
#     print(f'{word}')
#     newVocab.write(f'{word}\n')
                    
#Unused Functions
#  
#getting entry in LCQuad
# lcTrain = [val for val in dataLCQTrain if val['_id'] == uid]
# lcTest = [val for val in dataLCQTest if val['_id'] == uid]
# if(lcTrain):
#     unique_template.add((lcTrain[0])['sparql_template_id'])
# elif(lcTest):
#     unique_template.add((lcTest[0])['sparql_template_id'])


# def _template_to_vquanda(self, data):
#     ansTempDic = {}
#     for val1 in tqdm.tqdm(self.templateJson):
#         tempId = val1['id']
#         lcTrainIds = [val['_id'] for val in self.dataLCQTrain if val['sparql_template_id'] == tempId]
#         lcTestIds = [val['_id'] for val in self.dataLCQTest if val['sparql_template_id'] == tempId]
#         lcIds = lcTrainIds + lcTestIds
#         if(lcIds):
#             answers = [val for val in data if val['uid'] in lcIds]
#             index = random.randrange(len(answers))
#             #for index in range(len(answers)):
#             answer_template = self._cover_answers((answers[index])['verbalized_answer'])
#             question, answer_template = self._cover_entities((answers[index])['uid'], "test", answer_template)
#             answer_template = self._cover_nouns_and_verbs(answer_template)
#             #answer_template = self._cover_predicates(answer_template)
#             ansTempDic[tempId] = answer_template
#         test = 0

# def _cover_predicates(self, text):

#     for word in self.predicates:
#         word = (' ' + word + ' ')
#         if word in text:
#             predicate = ' pred '
#             text = text.replace(word, predicate)

#     j = 1
#     textList = text.split()
#     for i, word in enumerate(textList):
#         if word == 'pred':
#             predicate = 'pred' + str(j)
#             textList[i] = predicate
#             #word = predicate
#             j += 1
#     text = ' '.join(textList)
#     return text

# def _cover_nouns_and_verbs(self, text):
#     doc = self.nlp(text)
#     possible_ent = ['NOUN', 'PROPN']
#     skip_values = ['ans']
#     predicate = "pred"
#     entity = "ent"

#     res = text.rindex("ent")
#     num = text[res + 3]
#     j = int(num)

#     textList = text.split()
#     for i, token in enumerate(doc):
#         if(token.text == 'ans'):
#             continue
#         elif((token.text).startswith('ent')):
#             j += 1
#             continue

#         if(token.pos_ == 'VERB'):
#             textList[i] = predicate + str(j)
#         elif(token.pos_ in possible_ent):
#             textList[i] = entity + str(j)
#         #else Do Nothing
#         j += 1
#     text =' '.join(textList)

# def _answer_templates(self, data):
#     ansTempDic = {}
#     for example in tqdm.tqdm(data):
#         tempId = self._vquanda_to_template(uid=example['uid'])
#         if tempId not in ansTempDic:
#             answers = [val['verbalized_answer'] for val in data if val['uid']=]
#             index = random.randrange(len(answers))
#             answer_template = self._cover_answers(answers[index])
#             question, answer_template = self._cover_entities(example['uid'], "test", answer_template)
#             ansTempDic[tempId] = answer_template
#     return ansTempDic

# def _read_predicates(self):
#     pred_list = []
#     with open('/data/premnadh/Hybrid-QASystem/Utils/predicates.txt', 'r') as file1:
#         contents = file1.readlines()
#         nContents = [con.replace(',\n', '') for con in contents]
#         for pred in nContents:
#             if 'dbpedia.org/ontology' in pred or 'dbpedia.org/property' in pred:
#                 pred = pred.replace('<','').replace('>','')
#                 pred = pred.rsplit('/', 1)[-1].lstrip('<').rstrip('>')
#                 pred = filter(None, re.split("([A-Z][^A-Z]*)", pred))
#                 pred = ' '.join(pred)
#                 pred_list.append(pred)
#     return pred_list
