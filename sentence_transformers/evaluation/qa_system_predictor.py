from . import SentenceEvaluator, SimilarityFunction
from torch.utils.data import DataLoader
import torch
import logging
from tqdm import tqdm
from ..util import batch_to_device
import os
import csv
import nltk
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import pearsonr, spearmanr
import numpy as np
import statistics
from Utils.constants import CUDA
from sentence_transformers.qa_system.transformer import Encoder, Decoder, NoamOpt
from sentence_transformers.qa_system.seq2seq import Seq2Seq
from sentence_transformers.qa_system.verbaldataset import VerbalDataset
from sentence_transformers.qa_system.predictor import Predictor

class QA_Hybrid_System(object):

    def __init__(self, dataloader: DataLoader, decoder, criterion, main_similarity: SimilarityFunction = None, name: str = '', show_progress_bar: bool = None):
        """
        Constructs an evaluator based for the dataset

        The labels need to indicate the similarity between the sentences.

        :param dataloader:
            the data for the evaluation
        :param main_similarity:
            the similarity metric that will be used for the returned score
        """
        self.dataloader = dataloader
        self.decoder = decoder
        self.criterion = criterion
        self.main_similarity = main_similarity
        self.name = name
        if name:
            name = "_"+name

        if show_progress_bar is None:
            show_progress_bar = (logging.getLogger().getEffectiveLevel() == logging.INFO or logging.getLogger().getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.device = torch.device(CUDA if torch.cuda.is_available() else "cpu")
        self.score = 0
        self.instances = 0
        self.additional_tokens = ['[CLS]', '[SEP]', '[PAD]']
        self.dataset = VerbalDataset('/data/premnadh/VQUANDA/dataset')
        self.dataset.load_data_and_fields(cover_entities=True, query_as_input=True)
        self.src_vocab, self.trg_vocab = self.dataset.get_vocabs()
        self.qa_system_model = self.load('/data/premnadh/Question-Answering-System/saved_models/')

    def load(self, model_path):
        """Load model using name"""
        name = 'transformer.pt'
        encoder = Encoder(self.src_vocab, self.device)
        decoder = Decoder(self.trg_vocab, self.device)
        model = Seq2Seq(encoder, decoder, name=None).to(self.device)
        model.load_state_dict(torch.load(model_path + name))
        return model
    
    def example_score(self, reference, hypothesis):
        """Calculate blue score for one example"""
        return nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)

    def _get_logical_form(self, question, predictor):
        hypothesis = predictor.predict(question)
        return hypothesis
    
    def _process_logical_form(self, sentence_features):
        feature_lists = {}
        for feature_name in sentence_features:
            if feature_name not in feature_lists:
                feature_lists[feature_name] = []
            feature_lists[feature_name].append(sentence_features[feature_name])

        for feature_name in feature_lists:
            feature_lists[feature_name] = torch.tensor(np.asarray(feature_lists[feature_name]))
        
        return feature_lists

    def __call__(self, model: 'SequentialSentenceEmbedder', output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        model.eval()
        embeddings1 = []
        embeddings2 = []
        labels = []
        results = []
        epoch_loss = 0

        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}:"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps:"
        else:
            out_txt = ":"

        logging.info("Evaluation the model on "+self.name+" dataset"+out_txt)

        self.dataloader.collate_fn = model.smart_batching_collate

        iterator = self.dataloader
        if self.show_progress_bar:
            iterator = tqdm(iterator, desc="Convert Evaluating")

        for step, batch in enumerate(iterator, start=0):
            sts = 0
            features, targets, cosineTargets = batch_to_device(batch, sts, self.device)
            for idx in range(targets.size()[0]):
    
                sentence_features = []

                target = targets[idx].tolist()
                for i,_ in reversed(list(enumerate(target))):
                    if(target[i]==0):
                        del target[i]
                    else:
                        break
                target = target[1:-1]
                reference = model._first_module().detokenize(target)

                #for the question
                #for paired_sentence_idx in range(len(features)):
                input_ids = features[0]['input_ids'][idx].unsqueeze(0)
                token_type_ids = features[0]['token_type_ids'][idx].unsqueeze(0)
                input_mask = features[0]['input_mask'][idx].unsqueeze(0)
                sentence_length = features[0]['sentence_lengths'][idx].unsqueeze(0)
                sentence_features.append({'input_ids': input_ids, 'token_type_ids': token_type_ids, 'input_mask': input_mask, 'sentence_lengths': sentence_length})
                
                input_ids_original = features[0]['input_ids'][idx]
                detokenize_question = model._first_module().detokenize(input_ids_original)
                question = [x for x in detokenize_question if x not in self.additional_tokens]

                predictor = Predictor(self.qa_system_model, self.src_vocab, self.trg_vocab, self.device)
                hypo_log_form_text = self._get_logical_form(question, predictor)
                hypo_log_form = model.tokenize(''.join(hypo_log_form_text))
                log_form_features = model.get_sentence_features(hypo_log_form, len(hypo_log_form_text))
                #convert to tensor and then to batch
                feature_list_logform = self._process_logical_form(log_form_features)
                #to device
                for feature_name in feature_list_logform:
                    feature_list_logform[feature_name] = feature_list_logform[feature_name].to(self.device)

                input_ids_logForm = feature_list_logform['input_ids'][0].unsqueeze(0)
                token_type_ids_logForm = feature_list_logform['token_type_ids'][0].unsqueeze(0)
                input_mask_logForm = feature_list_logform['input_mask'][0].unsqueeze(0)
                sentence_length_logForm = feature_list_logform['sentence_lengths'][0].unsqueeze(0)
                sentence_features.append({'input_ids': input_ids_logForm, 'token_type_ids': token_type_ids_logForm, 'input_mask': input_mask_logForm, 'sentence_lengths': sentence_length_logForm})

                with torch.no_grad():
                    emb = [model(sent_features) for sent_features in sentence_features]
                    tokens = [sentence_feature['input_ids'] for sentence_feature in sentence_features]
                    src_tokens = tokens[0]
                    src_query_tokens = tokens[1]
                
                reps_a, encoder_a = emb[0]
                emb1 = reps_a['sentence_embedding'].to("cpu").numpy()
                reps_b, encoder_b = emb[1]
                emb2 = reps_b['sentence_embedding'].to("cpu").numpy()

                try:
                    cosine_scores = (1 - (paired_cosine_distances(emb1, emb2)))
                    euclidean_distances = paired_euclidean_distances(emb1, emb2)
                except Exception as e:
                    raise(e)

                with torch.no_grad():
                    cosinescore_tensor = torch.tensor(cosine_scores).to(self.device)
                    threshhold_tensor = torch.sigmoid(model.fc(cosinescore_tensor))
                    threshhold_tensor = torch.round(threshhold_tensor)
                highSimilarity = threshhold_tensor.item()

                if(highSimilarity):
                    encoder_out = torch.cat((encoder_a, encoder_b), dim=1)
                else:
                    extra = torch.zeros(encoder_a.size()).to(CUDA)
                    encoder_out = torch.cat((extra, encoder_b), dim=1)

                outputs = [model._first_module().cls_token_id]

                for _ in range(self.decoder.max_positions-2):
                    trg_tensor = torch.LongTensor(outputs).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        output = self.decoder(trg_tensor, encoder_out,
                                        src_tokens=src_tokens, src_query_tokens=src_query_tokens)

                    prediction = output.argmax(2)[:, -1].item()

                    if prediction == model._first_module().sep_token_id:
                        break
                    outputs.append(prediction)
                
                translation = model._first_module().detokenize(outputs)
                hypothesis = translation[1:]

                blue_score = self.example_score(reference, hypothesis)
                results.append({
                    'reference': reference,
                    'hypothesis': hypothesis
                })

                embeddings1.extend(emb1)
                embeddings2.extend(emb2)
            
                self.score += blue_score
                self.instances += 1

        for result in results[350:370]:
            print("reference ", result['reference'])
            print("hypothesis", result['hypothesis'])
        print("Bleu Score: ", self.score / self.instances)
        return self.score / self.instances

