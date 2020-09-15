"""
This file is reponsible for:
1. Computing the loss. We have two kinds of losses:
    a. The cross entropy loss from the decoder (for verbalization).
    b. CosineEmbedding loss from the similary of question and logical form. This loss gives the right
       logical form a high cosine similarity.
2. Trains a perceptron to find the threshold for the cosine similarity.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import statistics
import numpy as np
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict
from ..SentenceTransformer import SentenceTransformer
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances
from Utils.constants import CUDA

class MultipleNegativesRankingLoss(nn.Module):
    def __init__(self, model: SentenceTransformer, decoder, criterion):
        super(MultipleNegativesRankingLoss, self).__init__()
        self.model = model
        self.decoder = decoder
        self.threshold_criterion = nn.BCEWithLogitsLoss()
        self.criterion = criterion
        self.device = torch.device(CUDA if torch.cuda.is_available() else "cpu")
        self.cosineCriterion = nn.CosineEmbeddingLoss(margin=0.1).to(self.device)
        self.distance_metric = lambda x, y: F.pairwise_distance(x, y, p=2)
        self.triplet_margin = 1

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor, targets: Tensor, epoch):
        loss = 0
        reps = [self.model(sentence_feature) for sentence_feature in sentence_features]
        tokens = [sentence_feature['input_ids'] for sentence_feature in sentence_features]
        src_tokens = tokens[0]
        src_query_tokens = tokens[1]
        
        reps_a, encoder_a = reps[0]
        reps_a = reps_a['sentence_embedding']
        reps_b, encoder_b = reps[1]
        reps_b = reps_b['sentence_embedding']
        
        input_trg = labels[:, :-1]

        batch_size = src_tokens.size()[0]

        encoder_out = torch.zeros([1, encoder_a.size()[1] + encoder_b.size()[1], 768]).to(self.device)
        
        checkIndex = 0
        count = 0
        threshhold_loss = 0
        cosinescores_batch = []
        for idx in range(batch_size):
            emb_a = (reps_a[idx].unsqueeze(0)).to("cpu").detach().numpy()
            emb_b = (reps_b[idx].unsqueeze(0)).to("cpu").detach().numpy()
            encoder_question = encoder_a[idx].unsqueeze(0)
            encoder_query = encoder_b[idx].unsqueeze(0)
        
            try:
                cosine_scores = 1 - (paired_cosine_distances(emb_a, emb_b))
            except Exception as e:
                raise(e) 

            cosinescores_batch.append(cosine_scores)

            target = targets[idx]

            if(target == 1.0):
                #if(checkIndex == 0):
                if(idx == 0):
                    encoder_out = torch.cat((encoder_question, encoder_query), dim=1)
                    #checkIndex = 1
                else:
                    encoder_sample = torch.cat((encoder_question, encoder_query), dim=1)
                    encoder_out = torch.cat((encoder_out, encoder_sample), dim=0)
            else:
                # i = idx - count
                # input_tr1 = input_trg[:i, :]
                # input_tr2 = input_trg[i+1:, :]
                # input_trg = torch.cat([input_trg[:i, :], input_trg[i+1:, :]])
                # labels = torch.cat([labels[:i, :], labels[i+1:, :]])
                # src_tokens = torch.cat([src_tokens[:i, :], src_tokens[i+1:, :]])
                # src_query_tokens = torch.cat([src_query_tokens[:i, :], src_query_tokens[i+1:, :]])
                # count += 1

                #Masking on src_tokens does not work due to error in gradient computation
                extra = encoder_question.size()[1] 
                temp = torch.zeros(1, extra, 768).to(self.device)
                if(idx == 0):
                    encoder_out = torch.cat((temp, encoder_query), dim=1)
                else:
                    encoder_sample_query = torch.cat((temp, encoder_query), dim=1) 
                    encoder_out = torch.cat((encoder_out, encoder_sample_query), dim=0)
        
        #training the threshold
        # cosinescore_tensor = torch.tensor(cosinescores_batch).to(self.device)
        # fc_output = self.model.fc(cosinescore_tensor)
        # threshold_targets = targets.clone()
        # threshold_targets[threshold_targets == -1] = 0
        # threshhold_loss = self.threshold_criterion(fc_output, threshold_targets.unsqueeze(1))

        #if(encoder_out.sum().item() != 0):
        decoder_out, attention = self.decoder(input_trg, encoder_out,
                                        src_tokens=src_tokens, src_query_tokens=src_query_tokens,
                                        teacher_forcing_ratio=1.0, onlyQuery=0)

        trg = labels[:, 1:]
        output = decoder_out.contiguous().view(-1, decoder_out.shape[-1])
        trg = trg.contiguous().view(-1)
        loss = self.criterion(output, trg)
        #cosine_loss = self.cosineCriterion(reps_a, reps_b, targets)
        return loss  #+ cosine_loss + threshhold_loss

    # Multiple Negatives Ranking Loss
    # Paper: https://arxiv.org/pdf/1705.00652.pdf
    #   Efficient Natural Language Response Suggestion for Smart Reply
    #   Section 4.4
    def multiple_negatives_ranking_loss(self, embeddings_a: Tensor, embeddings_b: Tensor):
        """
        Compute the loss over a batch with two embeddings per example.

        Each pair is a positive example. The negative examples are all other embeddings in embeddings_b with each embedding
        in embedding_a.

        See the paper for more information: https://arxiv.org/pdf/1705.00652.pdf
        (Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4)

        :param embeddings_a:
            Tensor of shape (batch_size, embedding_dim)
        :param embeddings_b:
            Tensor of shape (batch_size, embedding_dim)
        :return:
            The scalar loss
        """
        scores = torch.matmul(embeddings_a, embeddings_b.t())
        diagonal_mean = torch.mean(torch.diag(scores))
        mean_log_row_sum_exp = torch.mean(torch.logsumexp(scores, dim=1))
        return -diagonal_mean + mean_log_row_sum_exp

    def TripletLoss(self, rep_anchor, rep_pos, rep_neg1, rep_neg2):
        distance_pos = self.distance_metric(rep_anchor, rep_pos)
        distance_neg1 = self.distance_metric(rep_anchor, rep_neg1)
        distance_neg2 = self.distance_metric(rep_anchor, rep_neg2)

        #losses = F.relu(2*distance_pos - distance_neg1 - distance_neg2 + 2*self.triplet_margin)
        losses1 = F.relu(distance_pos - distance_neg1 + self.triplet_margin)
        losses2 = F.relu(distance_pos - distance_neg2 + self.triplet_margin)
        losses = torch.cat((losses1, losses2), dim=0)
        return losses.mean()

