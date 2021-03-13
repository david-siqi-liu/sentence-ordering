from src.config import *
from src.data import *
from src.graph import *

from transformers import AdamW, BertForSequenceClassification, BertTokenizer, get_linear_schedule_with_warmup
import torch
import time
import copy
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from scipy.stats import spearmanr


def load_model_fsent():
    return BertForSequenceClassification.from_pretrained(
        args['model_type'],
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False
    )


def load_model_pair():
    return BertForSequenceClassification.from_pretrained(
        args['model_type'],
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False
    )


def load_tokenizer():
    return BertTokenizer.from_pretrained(
        args['model_type'],
        do_lower_case=args['do_lower_case']
    )


def get_optimizer_fsent(model):
    return AdamW(
        model.parameters(),
        lr=args['lr_fsent'],
        eps=args['adam_eps_fsent']
    )


def get_optimizer_pair(model):
    return AdamW(
        model.parameters(),
        lr=args['lr_pair'],
        eps=args['adam_eps_pair']
    )


def get_scheduler_fsent(optimizer, num_training_steps):
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args['warmup_steps_fsent'],
        num_training_steps=num_training_steps
    )


def get_scheduler_pair(optimizer, num_training_steps):
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args['warmup_steps_pair'],
        num_training_steps=num_training_steps
    )


def train(model, model_name, dataloaders, dataset_sizes, optimizer, scheduler, num_epochs):
    # set seed
    set_seed()

    # starting time
    since = time.time()

    # store model that yields the highest accuracy in the validation set
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0

    # keep track of the statistics
    losses = {'train': [], 'val': []}
    accuracies = {'train': [], 'val': []}

    # iterate through each epoch
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # set model to training mode
            else:
                model.eval()   # set model to evaluate mode

            # loss and number of correct predictions
            running_loss = 0.0
            running_corrects = 0
            running_data = 0

            # iterate over each batch
            with tqdm(dataloaders[phase], unit="batch") as tepoch:
                for batch in tepoch:
                    # Unravel inputs
                    input_ids = batch['input_ids'].to(device)
                    if 'token_type_ids' in batch:
                        # pairwise prediction
                        token_type_ids = batch['token_type_ids'].to(device)
                    else:
                        # first sentence prediction
                        token_type_ids = None
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)

                    # reset the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(
                            input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        loss = outputs.loss
                        logits = outputs.logits
                        preds = torch.argmax(logits, axis=1)

                        # backward only if in training
                        if phase == 'train':
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(),
                                1.0
                            )
                            optimizer.step()
                            scheduler.step()

                    # add to statistics
                    running_loss += loss.item() * len(labels)
                    running_corrects += torch.sum(
                        preds == labels.data.flatten()
                    )
                    running_data += len(labels)

                    # update progress bar
                    tepoch.set_postfix(
                        loss=(running_loss / running_data),
                        accuracy=(running_corrects.item() / running_data)
                    )
                    time.sleep(0.1)

            # compute loss and accuracy at epoch level
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            losses[phase].append(epoch_loss)
            accuracies[phase].append(epoch_acc)

            print('{} Loss: {:.4f}; Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc
            ))

            # deep copy the model when epoch accuracy (on validation set) is the best so far
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                torch.save(model, f'{model_name}_model_checkpoint.pt')
                print('Best model so far! Saved checkpoint.')

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60
    ))
    print('Best Val Acc: {:4f} at Epoch: {:d}'.format(best_acc, best_epoch))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_epoch, losses, accuracies


def predict_submission(docs, device, model_fsent, model_pair, tokenizer):
    preds = []
    with tqdm(total=len(docs)) as pbar:
        for i, doc in enumerate(docs):
            pred_index = predict(doc, device, model_fsent,
                                 model_pair, tokenizer)
            d = {
                'ID': doc['ID'],
                'index1': pred_index[0],
                'index2': pred_index[1],
                'index3': pred_index[2],
                'index4': pred_index[3],
                'index5': pred_index[4],
                'index6': pred_index[5],
            }
            preds.append(d)
            pbar.update(1)
            time.sleep(0.1)
    return pd.DataFrame(preds)


def evaluate(docs, device, model_fsent, model_pair, tokenizer, num_samples=100):
    set_seed()
    sample_docs = random.sample(docs, num_samples)
    running_corr = 0
    with tqdm(total=num_samples) as pbar:
        for i, doc in enumerate(sample_docs):
            gold_index = doc['indexes']
            pred_index = predict(doc, device, model_fsent,
                                 model_pair, tokenizer)
            corr, _ = spearmanr(gold_index, pred_index)
            running_corr += corr
            pbar.update(1)
            pbar.set_postfix(
                spearman=(running_corr / (i + 1))
            )
            time.sleep(0.1)
    return running_corr / num_samples


def predict(doc, device, model_fsent, model_pair, tokenizer):
    # predict first sentence position
    fsent_pos = predict_fsent_pos(doc, device, model_fsent, tokenizer)
    # predict sentence pairs logits
    pair_logits = predict_pair_logits(doc, device, model_pair, tokenizer)
    # construct graph based on predictions
    num_vertices = len(doc['sentences'])
    graph = make_graph(num_vertices, pair_logits)
    # sort and get the order
    order, weight = graph.max_flow(fsent_pos)
    # get indices
    index = [-1] * num_vertices
    for p, o in enumerate(order):
        index[o] = p
    return index


def predict_fsent_pos(doc, device, model, tokenizer):
    fsent_pos, max_logit = None, float('-inf')
    for i, text in enumerate(doc['sentences']):
        cleaned_text = clean_text(text)
        encoding = tokenizer.encode_plus(
            cleaned_text,
            add_special_tokens=True,
            max_length=args['max_seq_length_fsent'],
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        outputs = model(
            input_ids=encoding.input_ids.to(device),
            token_type_ids=None,
            attention_mask=encoding.attention_mask.to(device)
        )
        logits = outputs.logits.detach().cpu().numpy().squeeze()
        pred = np.argmax(logits)
        if (pred == 1) and (logits[1] > max_logit):
            fsent_pos = i
            max_logit = logits[1]
    return fsent_pos


def predict_pair_logits(doc, device, model, tokenizer):
    pair_logits = []
    perms = list(permutations(enumerate(doc['sentences']), 2))
    for (a, text_a), (b, text_b) in perms:
        cleaned_text_a = clean_text(text_a)
        cleaned_text_b = clean_text(text_b)
        encoding = tokenizer.encode_plus(
            cleaned_text_a,
            cleaned_text_b,
            add_special_tokens=True,
            max_length=args['max_seq_length_pair'],
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        if len(encoding['input_ids']) > args['max_seq_length_pair']:
            cleaned_text_a, cleaned_text_b = truncate_texts(
                cleaned_text_a,
                cleaned_text_b,
                tokenizer,
                args['max_seq_length_pair']
            )
            encoding = tokenizer.encode_plus(
                cleaned_text_a,
                cleaned_text_b,
                add_special_tokens=True,
                max_length=args['max_seq_length_pair'],
                return_token_type_ids=True,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            assert len(encoding['input_ids']) <= args['max_seq_length_pair']
        outputs = model(
            input_ids=encoding.input_ids.to(device),
            token_type_ids=encoding.token_type_ids.to(device),
            attention_mask=encoding.attention_mask.to(device)
        )
        logits = outputs.logits.detach().cpu().numpy().squeeze()
        pair_logits.append((a, b, logits))
    return pair_logits


def make_graph(num_vertices, pair_logits):
    graph = Graph(num_vertices)
    # add edge weights
    for a, b, [logit0, logit1] in pair_logits:
        graph.addEdge(a, b, logit1)
    return graph
