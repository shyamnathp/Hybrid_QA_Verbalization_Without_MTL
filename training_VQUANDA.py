"""
This file is the main run file, responsible for creating the dataloaders, preprocessing the dataset, training
and evaluation
"""
from torch.utils.data import DataLoader
import math
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses, readers, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.evaluation.predictor import Predictor
from sentence_transformers.evaluation.qa_system_predictor import QA_Hybrid_System
from sentence_transformers.readers import VQUANDAReader
from sentence_transformers.models.transformer import NoamOpt, Decoder
from sentence_transformers.STS.datasets import SentencesDatasetSTS
from sentence_transformers.STS.STSDataReader import STSDataReader
from sentence_transformers.STS.EmbeddingSimilarityEvaluatorSTS import EmbeddingSimilarityEvaluatorSTS
from Utils.constants import CUDA

import logging
import random
from datetime import datetime
import numpy as np
import os

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

print(torch.cuda.is_available())

device = torch.device(CUDA if torch.cuda.is_available() else "cpu")
# Read the dataset
model_name = 'distilbert-base-uncased'
train_batch_size = 30
num_epochs = 20
model_save_path = 'output/training_VQUANDA_continue_training-' + model_name + '-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#model_save_path = '/data/premnadh/Hybrid-QASystem-WithoutMultitask/Hybrid-QASystem/saved_models/similarityModel_Batch-30_Epoch_20'

vquanda_reader = readers.VQUANDAReader.VQUANDAReader('/data/premnadh/VQUANDA/dataset')

# Use BERT for mapping tokens to embeddings
word_embedding_model = models.BERT(model_name)
decoder = Decoder(word_embedding_model.tokenizer.vocab, device)

for p in decoder.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)
# define criterion
criterion = nn.CrossEntropyLoss(ignore_index=0)
# Load a pre-trained sentence transformer model
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])#, decoder=decoder, criterion=criterion)

parameters_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'The loss model has {parameters_num:,} trainable parameters')
print('--------------------------------')

# #STS Training
# sts_batch_size = 16
# sts_reader = STSDataReader('/data/premnadh/Hybrid-QASystem-New/Hybrid-QASystem/sentence_transformers/stsbenchmark', normalize_scores=True)
# # Convert the dataset to a DataLoader ready for training
# logging.info("Read STSbenchmark train dataset")
# train_data_sts = SentencesDatasetSTS(sts_reader.get_examples('sts-train.csv'), model)
# train_dataloader_sts = DataLoader(train_data_sts, shuffle=True, batch_size=sts_batch_size)
# train_loss_sts = losses.CosineSimilarityLoss(model=model)

# logging.info("Read STSbenchmark dev dataset")
# dev_data_sts = SentencesDatasetSTS(examples=sts_reader.get_examples('sts-dev.csv'), model=model)
# dev_dataloader_sts = DataLoader(dev_data_sts, shuffle=False, batch_size=sts_batch_size)
# evaluator_sts = EmbeddingSimilarityEvaluatorSTS(dev_dataloader_sts)

# # Configure the training. We skip evaluation in this example
# warmup_steps = math.ceil(len(train_data_sts)*num_epochs/train_batch_size*0.1) #10% of train data for warm-up
# logging.info("Warmup-steps: {}".format(warmup_steps))

# model.fit(train_objectives=[(train_dataloader_sts, train_loss_sts)],
#           evaluator=evaluator_sts,
#           epochs=4,
#           evaluation_steps=1000,
#           warmup_steps=warmup_steps,
#           output_path=None, sts = 1)

# Convert the dataset to a DataLoader ready for training
logging.info("Read VQUANDA train dataset")
train_data = SentencesDataset(vquanda_reader.get_examples_train('/train.json'), model)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
train_loss = losses.MultipleNegativesRankingLoss(model=model, decoder=decoder, criterion=criterion)

logging.info("Read VQUANDA dev dataset")
dev_data = SentencesDataset(examples=vquanda_reader.get_examples_test('/test.json'), model=model)
test, val = train_test_split(dev_data, test_size=0.5, shuffle=False)
dev_dataloader = DataLoader(val, shuffle=False, batch_size=train_batch_size)
evaluator = EmbeddingSimilarityEvaluator(dev_dataloader, decoder=decoder, criterion=criterion)

# Configure the training. We skip evaluation in this example
warmup_steps = math.ceil(len(train_data)*num_epochs/train_batch_size*0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path, sts=0)

##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################
similarityPath = '/data/premnadh/Hybrid-QASystem-WithoutMultitask/Hybrid-QASystem/saved_models/similarityModel_Batch-30_Epoch_20'
modelSimilarity = SentenceTransformer(similarityPath)
model = SentenceTransformer(model_save_path)
#test_data = SentencesDataset(examples=vquanda_reader.get_examples("/test.json"), model=model)
test_dataloader = DataLoader(test, shuffle=False, batch_size=train_batch_size)
predictor = Predictor(modelSimilarity, test_dataloader, decoder=decoder, criterion=criterion)
#predictor = QA_Hybrid_System(test_dataloader, decoder=decoder, criterion=criterion)
model.evaluate(predictor)
