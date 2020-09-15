from . import SentenceEvaluator, SimilarityFunction
from torch.utils.data import DataLoader

import torch
import logging
from tqdm import tqdm
from ..util import batch_to_device
import os
import csv
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import pearsonr, spearmanr
import numpy as np
import statistics
from Utils.constants import CUDA

class EmbeddingSimilarityEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on the similarity of the embeddings by calculating the Spearman and Pearson rank correlation
    in comparison to the gold standard labels.
    The metrics are the cosine similarity as well as euclidean and Manhattan distance
    The returned score is the Spearman correlation with a specified metric.

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """


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
        self.csv_file = "similarity_evaluation"+name+"_results.csv"
        self.csv_headers = ["epoch", "steps", "cosine_pearson", "cosine_spearman", "euclidean_pearson", "euclidean_spearman", "manhattan_pearson", "manhattan_spearman", "dot_pearson", "dot_spearman"]

    def __call__(self, model: 'SequentialSentenceEmbedder', output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        model.eval()
        embeddings1 = []
        embeddings2 = []
        labels = []
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

        for step, batch in enumerate(iterator):
            sts=0
            features, label_ids, targets = batch_to_device(batch, sts, self.device)
            with torch.no_grad():
                emb = [model(sent_features) for sent_features in features]
                tokens = [sentence_feature['input_ids'] for sentence_feature in features]
                src_tokens = tokens[0]
                src_query_tokens = tokens[1]
            
            reps_a, encoder_a = emb[0]
            emb1 = reps_a['sentence_embedding']
            reps_b, encoder_b = emb[1]
            emb2 = reps_b['sentence_embedding']

            input_trg = label_ids[:, :-1]

            batch_size = src_tokens.size()[0]

            encoder_out = torch.empty([1, encoder_a.size()[1] + encoder_b.size()[1], 768]).to(CUDA)

            for idx in range(batch_size):
                emb_a = (emb1[idx].unsqueeze(0)).to("cpu").detach().numpy()
                emb_b = (emb2[idx].unsqueeze(0)).to("cpu").detach().numpy()
                encoder_question = encoder_a[idx].unsqueeze(0)
                encoder_query = encoder_b[idx].unsqueeze(0)

                try:
                    cosine_scores = (1 - (paired_cosine_distances(emb_a, emb_b)))
                    euclidean_distances = paired_euclidean_distances(emb_a, emb_b)
                except Exception as e:
                    raise(e)     

                with torch.no_grad():
                    cosinescore_tensor = torch.tensor(cosine_scores).to(self.device)
                    threshhold_tensor = torch.sigmoid(model.fc(cosinescore_tensor))
                    threshhold_tensor = torch.round(threshhold_tensor)
                highSimilarity = threshhold_tensor.item()
                
                #refactor the code
                #if(highSimilarity):
                if(idx == 0):
                    encoder_out = torch.cat((encoder_question, encoder_query), dim=1)
                else:
                    encoder_sample = torch.cat((encoder_question, encoder_query), dim=1)
                    encoder_out = torch.cat((encoder_out, encoder_sample), dim=0)
                # else:
                #     extra = encoder_question.size()[1]
                #     temp = torch.zeros(1, extra, 768).to(CUDA)
                #     if(idx == 0):
                #         encoder_out = torch.cat((temp, encoder_query), dim=1)
                #     else:
                #         encoder_sample_query = torch.cat((temp, encoder_query), dim=1) 
                #         encoder_out = torch.cat((encoder_out, encoder_sample_query), dim=0)

            decoder_out, attention = self.decoder(input_trg, encoder_out,
                                        src_tokens=src_tokens, src_query_tokens=src_query_tokens,
                                        teacher_forcing_ratio=1.0)

            trg = label_ids[:, 1:]
            output = decoder_out.contiguous().view(-1, decoder_out.shape[-1])
            trg = trg.contiguous().view(-1)
            loss = self.criterion(output, trg)
            epoch_loss += loss.item()
            embeddings1.extend(emb1)
            embeddings2.extend(emb2)
        
        eval_loss = epoch_loss / len(iterator)
        return eval_loss
        