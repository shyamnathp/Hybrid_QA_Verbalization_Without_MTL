"""VerbalDataset"""
import os
import re
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from torchtext.data import Field, Example, Dataset

from sentence_transformers.qa_system.constants import (
    ANSWER_TOKEN, ENTITY_TOKEN, SOS_TOKEN, EOS_TOKEN,
    SRC_NAME, TRG_NAME, TRAIN_PATH, TEST_PATH
)
class VerbalDataset(object):
    """VerbalDataset class"""
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
        self.earl_entities = self._read_earl_entites('/data/premnadh/Hybrid-QASystem-New/Hybrid-QASystem/Utils/earl_entities.json')
        self.templateJson = json.load(open('/data/premnadh/Hybrid-QASystem-New/Hybrid-QASystem/Utils/templates.json'))
        with open('/data/premnadh/Hybrid-QASystem/LCQuad/train-data.json') as json_fileTwo:
            self.dataLCQTrain = json.load(json_fileTwo)
        with open('/data/premnadh/Hybrid-QASystem/LCQuad/test-data.json') as json_fileThree:
            self.dataLCQTest = json.load(json_fileThree)

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
        # we cover all recognized entitries by the same token
        # this has to be improved based on the number of entities.
        # For example if we have 2 entities we create 2 tokens e.g. <ent1> <ent2>
        # In this way we know the position of each entity in the translated output
        i = j = 1
        for ent in question_entities: 
            question = question.replace(ent, ENTITY_TOKEN + str(i))
            i += 1

        return question, answer

    def _vquanda_to_template(self, uid):
        lcTrain = [val for val in self.dataLCQTrain if val['_id'] == uid]
        lcTest = [val for val in self.dataLCQTest if val['_id'] == uid]
        tempId = None
        if(lcTrain):
            tempId = (lcTrain[0])['sparql_template_id']
        elif(lcTest):
            tempId = (lcTest[0])['sparql_template_id']
        return tempId

    def _prepare_query_hybrid(self, query, uid, cover_entities=True):
        """
        trasnform query from this:
        SELECT DISTINCT COUNT(?uri) WHERE { ?x <http://dbpedia.org/ontology/commander> <http://dbpedia.org/resource/Andrew_Jackson> . ?uri <http://dbpedia.org/ontology/knownFor> ?x  . }
        To this:
        select distinct count_uri where brack_open var_x commander Andrew Jackson sep_dot var_uri known for var_x sep_dot brack_close
        """
        # lcTrain = [val for val in self.dataLCQTrain if val['_id'] == uid]
        # lcTest = [val for val in self.dataLCQTest if val['_id'] == uid]
        # tempId = None
        # if(lcTrain):
        #     tempId = (lcTrain[0])['sparql_template_id']
        # elif(lcTest):
        #     tempId = (lcTest[0])['sparql_template_id']
        tempId = self._vquanda_to_template(uid)

        tempEntry = [val for val in self.templateJson if val['id'] == tempId]
        #if(len(tempEntry) != 0):
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

                    with open('/data/premnadh/Hybrid-QASystem-New/Hybrid-QASystem/Utils/predicates.txt', 'r') as file1:
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

    def _prepare_query(self, query, cover_entities):
        """
        trasnform query from this:
        SELECT DISTINCT COUNT(?uri) WHERE { ?x <http://dbpedia.org/ontology/commander> <http://dbpedia.org/resource/Andrew_Jackson> . ?uri <http://dbpedia.org/ontology/knownFor> ?x  . }
        To this:
        select distinct count_uri where brack_open var_x commander Andrew Jackson sep_dot var_uri known for var_x sep_dot brack_close
        """
        query = query.replace('\n', ' ')\
                     .replace('\t', '')\
                     .replace('?', '')\
                     .replace('{?', '{ ?')\
                     .replace('>}', '> }')\
                     .replace('uri}', 'uri }').strip()
        query = query.split()
        new_query = []
        for q in query:
            if q in self.QUERY_DICT:
                q = self.QUERY_DICT[q]
            if 'http' in q:
                if 'dbpedia.org/ontology' in q or 'dbpedia.org/property' in q:
                    q = q.rsplit('/', 1)[-1].lstrip('<').rstrip('>')
                    q = filter(None, re.split("([A-Z][^A-Z]*)", q))
                    q = ' '.join(q)
                elif 'www.w3.org/1999/02/22-rdf-syntax-ns#type' in q:
                    q = 'type'
                elif cover_entities:
                    q = ENTITY_TOKEN
                else:
                    q = q.rsplit('/', 1)[-1].lstrip('<').rstrip('>').replace('_', ' ')
            new_query.append(q.lower())

        assert new_query[-1] == 'brackclose', 'Query not ending with a bracket.'
        return ' '.join(new_query)

    def _extract_question_answer(self, train, test):
        return [[data['question'], data['verbalized_answer']] for data in train], \
                [[data['question'], data['verbalized_answer']] for data in test]

    def _extract_query_answer(self, train, test):
        return [[data['query'], data['verbalized_answer']] for data in train], \
                [[data['query'], data['verbalized_answer']] for data in test]

    def _extract_question_query(self, train, test):
        return [[data['question'], data['query']] for data in train], \
                [[data['question'], data['query']] for data in test]

    def _make_torchtext_dataset(self, data, fields):
        examples = [Example.fromlist(i, fields) for i in data]
        return Dataset(examples, fields)

    def load_data_and_fields(self, cover_entities=True, query_as_input=True):
        """
        Load verbalization data
        Create source and target fields
        """
        train, test, val = [], [], []
        # read data
        with open(self.train_path) as json_file:
            train = json.load(json_file)

        with open(self.test_path) as json_file:
            test = json.load(json_file)

        # cover answers
        #for data in train: data.update((k, self._cover_answers(v)) for k, v in data.items() if k == "verbalized_answer")
        #for data in test: data.update((k, self._cover_answers(v)) for k, v in data.items() if k == "verbalized_answer")

        for data in [train, test]:
            for example in data:
                uid = example['uid']
                question = example['question']
                answer = example['verbalized_answer']
                query = example['query']
                query, logical_form, predicates, typeOfClass, templateId, typeOfQuery = self._prepare_query_hybrid(query, uid)
                question, answer = self._cover_entities(uid, question, answer)
                example.update(question=question, verbalized_answer=answer, query=logical_form)


        # extract question-answer or query-answer pairs
        # if query_as_input:
        #     train, test = self._extract_query_answer(train, test)
        # else:
        #     train, test = self._extract_question_answer(train, test)

        #extract question-query pair
        train, test = self._extract_question_query(train, test)

        # split test data to val-test
        test, val = train_test_split(test, test_size=0.5, shuffle=False)

        # create fields
        self.src_field = Field(tokenize=self.TOKENIZE_SEQ,
                               init_token=SOS_TOKEN,
                               eos_token=EOS_TOKEN,
                               lower=True,
                               include_lengths=True,
                               batch_first=True)
        self.trg_field = Field(tokenize=self.TOKENIZE_SEQ,
                               init_token=SOS_TOKEN,
                               eos_token=EOS_TOKEN,
                               lower=True,
                               batch_first=True)

        fields_tuple = [(SRC_NAME, self.src_field), (TRG_NAME, self.trg_field)]

        # create toechtext datasets
        self.train_data = self._make_torchtext_dataset(train, fields_tuple)
        self.valid_data = self._make_torchtext_dataset(val, fields_tuple)
        self.test_data = self._make_torchtext_dataset(test, fields_tuple)

        # build vocabularies
        self.src_field.build_vocab(self.train_data, min_freq=2)
        self.trg_field.build_vocab(self.train_data, min_freq=2)

    def get_data(self):
        """Return train, validation and test data objects"""
        return self.train_data, self.valid_data, self.test_data

    def get_fields(self):
        """Return source and target field objects"""
        return self.src_field, self.trg_field

    def get_vocabs(self):
        """Return source and target vocabularies"""
        return self.src_field.vocab, self.trg_field.vocab
