# coding: utf-8

# Loading the librairies used

import torch
from tqdm.notebook import tqdm
import random
import argparse
import pandas as pd
import numpy as np
import os

import sentencepiece
from transformers import XLNetTokenizer
from transformers import XLNetForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


# Reading the data and arguments

PATH = os.getcwd()
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str,
                    default=PATH+"/Data/")
parser.add_argument('--results_dir', type=str,
                    default=PATH+"/Results/")

args = parser.parse_args()

batch_size = 10
epochs = 2
seed_val = 17


DATA_PATH = args.data_dir
RESULTS_PATH = args.results_dir

X_train = pd.read_json(DATA_PATH+"train.json").set_index('Id')
X_test = pd.read_json(DATA_PATH+"test.json").set_index('Id')
Y_train = pd.read_csv(DATA_PATH+"train_label.csv").set_index('Id')


# Creation of the token lists and encoded data sets

tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')

encoded_data_train = tokenizer.batch_encode_plus(
    X_train.description.values, 
    add_special_tokens=True, 
    return_attention_mask=True, 
    padding=True,
    max_length=150, #max length of the description 
    return_tensors='pt',
    truncation=True
)


encoded_data_test = tokenizer.batch_encode_plus(
    X_test.description.values, 
    add_special_tokens=True, 
    return_attention_mask=True, 
    padding=True,
    max_length=150, #max length of the description 
    return_tensors='pt',
    truncation=True
)


input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(Y_train.Category.values)

input_ids_test = encoded_data_test['input_ids']
attention_masks_test = encoded_data_test['attention_mask']
labels_test = torch.tensor(np.array([0]*len(input_ids_test)), dtype=torch.long) #unimportant

dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)


# Creation of the model

model = XLNetForSequenceClassification.from_pretrained('xlnet-large-cased',
                                                       num_labels=28,
                                                       output_attentions=False,
                                                       output_hidden_states=False)



dataloader_train = DataLoader(dataset_train, 
                              sampler=RandomSampler(dataset_train), 
                              batch_size=batch_size)

dataloader_test = DataLoader(dataset_test, 
                             sampler=SequentialSampler(dataset_test), 
                             batch_size=batch_size)

optimizer = AdamW(model.parameters(),
                  lr=1e-5, 
                  eps=1e-8)

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train)*epochs)


random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


# GPU (highly recommended)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# Evaluation function

def evaluate(dataloader_val, model):

    model.eval()
    
    predictions = []
    
    for batch in dataloader_val:
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        logits = outputs[1]
        logits = logits.detach().cpu().numpy()
        
        predictions.append(logits)
    
    predictions = np.concatenate(predictions, axis=0)
    
    return predictions


# Training

for epoch in tqdm(range(1, epochs+1)):
    
    model.train()
    
    loss_train_total = 0

    progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
    for batch in progress_bar:

        model.zero_grad()
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }       

        outputs = model(**inputs)
        
        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
        
    tqdm.write(f'\nEpoch {epoch}')
    
    loss_train_avg = loss_train_total/len(dataloader_train)            
    tqdm.write(f'Training loss: {loss_train_avg}')


# Prediction

predictions = evaluate(dataloader_test, model)
arg_predictions = np.argmax(predictions, axis=1).flatten()

df_predictions = pd.DataFrame(arg_predictions, columns=['Category'])
df_predictions.to_csv(RESULTS_PATH+"predictions_xlnet_large.csv", index_label="Id")
    