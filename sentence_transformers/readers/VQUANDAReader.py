"""
This file is responsible for:
1. Preprocess, cover entities, cover answers
2. Create InputExample objects.
3. Use the query to form the logical forms.
4. self.similary stores the most similar 2 logical forms, to the particular sparql_id
"""
import os
import re
import json
import tqdm
import random
import operator

import spacy
from spacy.symbols import NOUN, PROPN, VERB
from pathlib import Path
from sklearn.model_selection import train_test_split
from torchtext.data import Field, Example, Dataset
from . import InputExample
from Utils.constants import (
    ANSWER_TOKEN, ENTITY_TOKEN, SOS_TOKEN, EOS_TOKEN,
    SRC_NAME, TRG_NAME, TRAIN_PATH, TEST_PATH
)

class VQUANDAReader(object):
    TOKENIZE_SEQ = lambda self, x: x.replace("?", " ?").\
                                    replace(".", " .").\
                                    replace(",", " ,").\
                                    replace("'", " '").\
                                    replace("(", "( ").replace(")", " )").replace(":", " : ").\
                                    split()
    ANSWER_REGEX = r'\[.*?\]'
    QUERY_DICT = {
        'x': 'varx',
        'uri': 'varuri',
        '{': 'brackopen',
        '}': 'brackclose',
        '.': 'sepdot',
        'COUNT(uri)': 'counturi'
    }
    ROOT_PATH = Path(os.path.dirname(__file__))

    def __init__(self, dataset_folder):
        self.data_path = str(dataset_folder)
        self.train_path = self.data_path + '/train.json'
        self.test_path = self.data_path + '/test.json'
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        self.src_field = None
        self.trg_field = None
        self.nlp = spacy.load('en_core_web_md')
        self.earl_entities = self._read_earl_entites(str('/data/premnadh/Hybrid-QASystem/Utils/earl_entities.json'))
        #self.predicates = self._read_predicates()
        with open('/data/premnadh/Hybrid-QASystem/LCQuad/train-data.json') as json_fileTwo:
            self.dataLCQTrain = json.load(json_fileTwo)
        with open('/data/premnadh/Hybrid-QASystem/LCQuad/test-data.json') as json_fileThree:
            self.dataLCQTest = json.load(json_fileThree)
        self.templateJson = json.load(open('/data/premnadh/Hybrid-QASystem-New/Hybrid-QASystem/Utils/templates.json'))
        self.similarityDict = self._similarity()

    def _read_earl_entites(self, path):
        entities = []
        with open(path) as json_file:
            entities = json.load(json_file)
        keys = [ent['uid'] for ent in entities]
        entities = dict(zip(keys, entities))
        return entities
    
    def _cover_answers(self, text):
        """
        Cover answers on text using an answer token
        """
        return re.sub(self.ANSWER_REGEX, ANSWER_TOKEN, text)
    
    def _cover_entities(self, uid, question, answer):
        """
        Cover entities on a given text.
        Since we use external entity recognizer it might
        miss some entities or cover wrong ones.
        A better approach will be to annotate all data
        and cover all entities. This should improve model
        performance.
        """
        # Try EARL for covering entities
        # EARL results are serialized
        data = self.earl_entities[uid]
        question_entities = data['question_entities']
        answer_entities = data['answer_entities']
        # we cover all recognized entitries by the same token
        # this has to be improved based on the number of entities.
        # For example if we have 2 entities we create 2 tokens e.g. <ent1> <ent2>
        # In this way we know the position of each entity in the translated output
        i = j = 1
        for ent in question_entities: 
            question = question.replace(ent, ENTITY_TOKEN + str(i))
            i += 1
        for ent in answer_entities: 
            answer = answer.replace(ent, ENTITY_TOKEN + str(j))
            j += 1

        return question, answer

    def _spacy_similarity(self, x, y):
        return x.similarity(y)

    def _similarity(self):
        """
        find similarity of each sparql_template_id to the 2 most similary logical forms
        """
        simDic = {}
        for val1 in tqdm.tqdm(self.templateJson):
            id1 = val1['id']
            logFormEmbOne = self.nlp(val1['logical_form'])
            similarityDictCommon = {val['id']: self._spacy_similarity(logFormEmbOne, self.nlp(val['logical_form'])) for val in self.templateJson if val['id'] != id1}
            similarityDict = {k: v for k, v in sorted(similarityDictCommon.items(), key=lambda item: item[1], reverse=True) if v != 1.0}
            maxIds = list(similarityDict.keys())
            simDic[id1] = maxIds[:2]
        return simDic
            
    def create_negative_forms(self, predicates, typeClass, originalId, typeOriginal):
        """
        Create negative logical forms based on similarity - fill predicates, class
        """
        negative_forms = []
        #create two negative examples
        for k in range(2):
            #id = random.randrange(len(template_ids))
            id = (self.similarityDict[originalId])[k]
            tempEntry = [val for val in self.templateJson if val['id'] == id]
            logical_form = (tempEntry[0])['logical_form']
            
            #basic formatting
            logical_form = logical_form.replace('(', ' ( ')\
                                   .replace(')', ' ) ')\
                                   .replace(',', ' ,').strip()

            #replace class
            if typeClass is not None:
                logical_form = logical_form.replace('class', typeClass)

            #replace predicates
            j = 1
            for originalPredicate in predicates:
                predicate = 'pred' + str(j)
                logical_form = logical_form.replace(predicate, originalPredicate)
                j += 1

            negative_forms.append(logical_form)

        return negative_forms

    def _vquanda_to_template(self, uid):
        """
        Convert vquanda id to sparql_template_id
        """
        lcTrain = [val for val in self.dataLCQTrain if val['_id'] == uid]
        lcTest = [val for val in self.dataLCQTest if val['_id'] == uid]
        tempId = None
        if(lcTrain):
            tempId = (lcTrain[0])['sparql_template_id']
        elif(lcTest):
            tempId = (lcTest[0])['sparql_template_id']
        return tempId

    def _get_negative_answer(self, predicates, typeClass, templateId):
        """
        Get the top 2 negative answer templates based on most 2 similar logical forms
        """
        negative_answers = []
        for k in range(2):
            id = (self.similarityDict[templateId])[k]
            tempEntry = [val for val in self.templateJson if val['id'] == id]
            answer_template = (tempEntry[0])['answer_template']
            idx = random.randrange(len(answer_template))
            #change answer_template to not list in templates.json
            answer = answer_template[idx]
            if typeClass is not None:
                answer = answer.replace('class', typeClass)

            #replace predicates
            j = 1
            for originalPredicate in predicates:
                predicate = 'pred' + str(j)
                answer = answer.replace(predicate, originalPredicate)
                j += 1

            negative_answers.append(answer)
        return negative_answers
            


    def _prepare_query(self, query, uid, cover_entities=True):
        """
        trasnform query from this:
        SELECT DISTINCT COUNT(?uri) WHERE { ?x <http://dbpedia.org/ontology/commander> <http://dbpedia.org/resource/Andrew_Jackson> . ?uri <http://dbpedia.org/ontology/knownFor> ?x  . }
        To this:
        select distinct count_uri where brack_open var_x commander Andrew Jackson sep_dot var_uri known for var_x sep_dot brack_close
        """
        tempId = self._vquanda_to_template(uid)

        tempEntry = [val for val in self.templateJson if val['id'] == tempId]
        logical_form = (tempEntry[0])['logical_form']
        typeOfQuery = (tempEntry[0])['type']

        logical_form = logical_form.replace('(', ' ( ')\
                                   .replace(')', ' ) ')\
                                   .replace(',', ' ,').strip()

        query = query.replace('\n', ' ')\
                     .replace('\t', '')\
                     .replace('?', '')\
                     .replace('{?', '{ ?')\
                     .replace('>}', '> }')\
                     .replace('{uri', '{ uri')\
                     .replace('uri}', 'uri }').strip()
        query = query.split()
        new_query = []
        predicateWords = []
        typeOfClass: str = None
        i = j = 1
        classType = 0
        for q in query:
            if q in self.QUERY_DICT:
                q = self.QUERY_DICT[q]
            if 'http' in q:
                if 'dbpedia.org/ontology' in q or 'dbpedia.org/property' in q:
                    original_q = q
                    original_q = original_q.replace('<','').replace('>','')
                    q = q.rsplit('/', 1)[-1].lstrip('<').rstrip('>')
                    q = filter(None, re.split("([A-Z][^A-Z]*)", q))
                    q = ' '.join(q)

                    with open('/data/premnadh/Hybrid-QASystem/Utils/predicates.txt', 'r') as file1:
                        contents = file1.readlines()
                        nContents = [con.replace(',\n', '') for con in contents]
                        if original_q in nContents:
                            predicate = 'pred'+str(j)
                            logical_form = logical_form.replace(predicate, q.lower())
                            predicateWords.append(q.lower())
                            j += 1

                    if(classType == 1):
                        logical_form = logical_form.replace('class', q.lower())
                        typeOfClass = q.lower()
                        classType = 0
                elif 'www.w3.org/1999/02/22-rdf-syntax-ns#type' in q:
                    q = 'type'
                    classType = 1
                elif cover_entities:
                    q = ENTITY_TOKEN + str(i)
                    i += 1
                else:
                    q = q.rsplit('/', 1)[-1].lstrip('<').rstrip('>').replace('_', ' ')
            new_query.append(q.lower())

        assert new_query[-1] == 'brackclose', 'Query not ending with a bracket.'
        return ' '.join(new_query), logical_form, predicateWords, typeOfClass, tempId, typeOfQuery

    def get_examples_train(self, filename, max_examples=0):
        """
        filename specified which data split to use (train.csv, dev.csv, test.csv).
        """
        examples = []
        data = []
        # read data
        with open(self.data_path + filename) as json_file:
            data = json.load(json_file)

        # cover answers
        for example in data:
            example.update((k, self._cover_answers(v)) for k, v in example.items() if k == "verbalized_answer")
        
        unique_template = set()
        for example in data:
            uid = example['uid']
            question = example['question']
            query = example['query']
            query, logical_form, predicates, typeOfClass, templateId, typeOfQuery = self._prepare_query(query, uid)
            negative_forms = self.create_negative_forms(predicates, typeOfClass, templateId, typeOfQuery)
            negative_answer = self._get_negative_answer(predicates, typeOfClass, templateId)

            answer = example['verbalized_answer']

            question, answer = self._cover_entities(uid, question, answer)
            #two negative forms added to InputExample
            examples.append(InputExample(guid=uid, texts=[question, logical_form], label=answer, target=1))
            #examples.append(InputExample(guid=uid, texts=[question, negative_forms[0]], label=negative_answer[0], target=-1))

            for i, neg_form in enumerate(negative_forms):
                examples.append(InputExample(guid=uid, texts=[question, neg_form], label=negative_answer[i], target=-1))

        return examples

    def get_examples_test(self, filename, max_examples=0):
        """
        filename specified which data split to use (train.csv, dev.csv, test.csv).
        """
        examples = []
        data = []
        # read data
        with open(self.data_path + filename) as json_file:
            data = json.load(json_file)

        # cover answers
        for example in data:
            example.update((k, self._cover_answers(v)) for k, v in example.items() if k == "verbalized_answer")
        
        unique_template = set()
        for example in data:
            uid = example['uid']
            question = example['question']
            query = example['query']
            query, logical_form, predicates, typeOfClass, templateId, typeOfQuery = self._prepare_query(query, uid)
            #negative_forms = self.create_negative_forms(predicates, typeOfClass, templateId, typeOfQuery)
            answer = example['verbalized_answer']
            
            question, answer = self._cover_entities(uid, question, answer)
            #two negative forms added to InputExample
            examples.append(InputExample(guid=uid, texts=[question, logical_form], label=answer, target=1))

        return examples

