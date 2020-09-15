"""
This file is reponsible for:
1. the prediction evaluation on the unseen test set.
2. diplsay the attention as a part(with respect to the prediction)
sentence_features = {question, query}
"""
from . import SentenceEvaluator, SimilarityFunction
from torch.utils.data import DataLoader
import torch
import logging
from tqdm import tqdm
from ..util import batch_to_device
import os
import csv
import nltk
from nltk.translate.bleu_score import SmoothingFunction
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import pearsonr, spearmanr
import numpy as np
import statistics
from Utils.constants import CUDA
from sentence_transformers.evaluation.display_attention import display_attention

class Predictor(object):

    def __init__(self, modelSimilarity, dataloader: DataLoader, decoder, criterion, main_similarity: SimilarityFunction = None, name: str = '', show_progress_bar: bool = None):
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
        self.modelSimilarity = modelSimilarity
        if name:
            name = "_"+name

        if show_progress_bar is None:
            show_progress_bar = (logging.getLogger().getEffectiveLevel() == logging.INFO or logging.getLogger().getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.additional_tokens = ['[PAD]']
        self.device = torch.device(CUDA if torch.cuda.is_available() else "cpu")
        self.score = 0
        self.instances = 0

    def example_score(self, reference, hypothesis):
        """Calculate blue score for one example"""
        #smoothing = SmoothingFunction.method4()
        return nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)

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
                
                original_q = []
                for paired_sentence_idx in range(len(features)):
                    #for feature_name in features[paired_sentence_idx]:
                    input_ids_original = features[paired_sentence_idx]['input_ids'][idx]
                    input_ids = features[paired_sentence_idx]['input_ids'][idx].unsqueeze(0)
                    detokenize_sentence = model._first_module().detokenize(input_ids_original)
                    #original_q.append([x for x in detokenize_sentence if x not in self.additional_tokens])
                    original_q.append(detokenize_sentence)
                    token_type_ids = features[paired_sentence_idx]['token_type_ids'][idx].unsqueeze(0)
                    input_mask = features[paired_sentence_idx]['input_mask'][idx].unsqueeze(0)
                    sentence_length = features[paired_sentence_idx]['sentence_lengths'][idx].unsqueeze(0)
                    sentence_features.append({'input_ids': input_ids, 'token_type_ids': token_type_ids, 'input_mask': input_mask, 'sentence_lengths': sentence_length})
                
                with torch.no_grad():
                    emb = [model(sent_features) for sent_features in sentence_features]
                    embSimilarity = [self.modelSimilarity(sent_features) for sent_features in sentence_features]
                    tokens = [sentence_feature['input_ids'] for sentence_feature in sentence_features]
                    src_tokens = tokens[0]
                    src_query_tokens = tokens[1]
                
                reps_a, encoder_a = emb[0]
                emb1 = reps_a['sentence_embedding'].to("cpu").numpy()
                reps_b, encoder_b = emb[1]
                emb2 = reps_b['sentence_embedding'].to("cpu").numpy()

                reps_a_sim, encoder_a_sim = embSimilarity[0]
                emb1_sim = reps_a_sim['sentence_embedding'].to("cpu").numpy()
                reps_b_sim, encoder_b_sim = embSimilarity[1]
                emb2_sim = reps_b_sim['sentence_embedding'].to("cpu").numpy()

                try:
                    cosine_scores = (1 - (paired_cosine_distances(emb1_sim, emb2_sim)))
                except Exception as e:
                    raise(e)

                #pass through trained perceptron to see if it has a good similarity or not
                with torch.no_grad():
                    cosinescore_tensor = torch.tensor(cosine_scores).to(self.device)
                    threshhold_tensor = torch.sigmoid(self.modelSimilarity.fc(cosinescore_tensor))
                    threshhold_tensor = torch.round(threshhold_tensor)
                highSimilarity = threshhold_tensor.item()

                #if goodsimilarty : use both question and query
                #else: set questions to 0's and append to query. Thus prediction is only based on query.

                if(highSimilarity):
                    encoder_out = torch.cat((encoder_a, encoder_b), dim=1)
                else:
                    extra = torch.zeros(encoder_a.size()).to(CUDA)
                    encoder_out = torch.cat((extra, encoder_b), dim=1)

                outputs = [model._first_module().cls_token_id]

                for _ in range(self.decoder.max_positions-2):
                    trg_tensor = torch.LongTensor(outputs).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        output, attention = self.decoder(trg_tensor, encoder_out,
                                                src_tokens=src_tokens, src_query_tokens=src_query_tokens)

                    prediction = output.argmax(2)[:, -1].item()

                    if prediction == model._first_module().sep_token_id:
                        break
                    outputs.append(prediction)
                
                translation = model._first_module().detokenize(outputs)
                hypothesis = translation[1:]

                #display
                # if(self.instances in range(250, 265)):
                #     display_attention(original_q[0]+original_q[1], hypothesis, attention, "test"+str(self.instances))

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
