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


def load_model():
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


def get_optimizer(model, lr, eps):
    return AdamW(
        model.parameters(),
        lr=lr,
        eps=eps
    )


def get_scheduler(optimizer, num_warmup_steps, num_training_steps):
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )


def train_fsent(model, device, dataloaders, dataset_sizes, optimizer, scheduler, num_epochs):
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
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
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
                    token_type_ids = None
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)

                    # reset the parameter gradients
                    optimizer.zero_grad()
                    model.zero_grad()

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
                best_epoch = epoch + 1
                torch.save(model, '{:s}fsent_model_checkpoint.pt'.format(
                    args['dir'] + args['model_output_dir']
                ))
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


def train_pair(model, model_fsent, tokenizer, device, val_set, train_dataloader, optimizer, scheduler, num_epochs):
    # set seed
    set_seed()

    # starting time
    since = time.time()

    # store model that yields the highest Spearman correlation in the validation set
    best_model_wts = copy.deepcopy(model.state_dict())
    best_spearman = float('-inf')
    best_epoch = 0

    # keep track of the statistics
    losses = {'train': []}
    accuracies = {'train': []}
    spearmans = {'val': []}

    # iterate through each epoch
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # each epoch has a training and validation phase
        for phase in ['train', 'val']:

            if phase == 'train':
                model.train()  # set model to training mode

                # loss and number of correct predictions
                running_loss = 0.0
                running_corrects = 0
                running_data = 0

                # iterate over each batch
                with tqdm(train_dataloader, unit="batch") as tepoch:
                    for batch in tepoch:
                        # Unravel inputs
                        input_ids = batch['input_ids'].to(device)
                        token_type_ids = batch['token_type_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['label'].to(device)

                        # number of data points
                        n_data = len(labels)

                        # reset the parameter gradients
                        optimizer.zero_grad()
                        model.zero_grad()

                        # forward
                        # track history only in train
                        with torch.set_grad_enabled(True):
                            outputs = model(
                                input_ids=input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask,
                                labels=labels
                            )
                            loss = outputs.loss
                            logits = outputs.logits
                            preds = torch.argmax(logits, axis=1)

                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(),
                                1.0
                            )
                            optimizer.step()
                            scheduler.step()

                        # add to statistics
                        running_loss += loss.item() * n_data
                        running_corrects += torch.sum(
                            preds == labels.data.flatten()
                        )
                        running_data += n_data

                        # update progress bar
                        tepoch.set_postfix(
                            loss=(running_loss / running_data),
                            accuracy=(running_corrects.item() / running_data)
                        )
                        time.sleep(0.1)

                # compute loss and accuracy at epoch level
                epoch_loss = running_loss / running_data
                epoch_acc = running_corrects.double() / running_data
                losses['train'].append(epoch_loss)
                accuracies['train'].append(epoch_acc)

                print('{} Loss: {:.4f}; Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc
                ))

            else:
                model.eval()   # set model to evaluate mode

                epoch_spearman, _ = evaluate(
                    val_set, device, model_fsent, model, tokenizer
                )
                spearmans['val'].append(epoch_spearman)

                # deep copy the model when epoch spearman correlation (on validation set) is the best so far
                if epoch_spearman > best_spearman:
                    best_spearman = epoch_spearman
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_epoch = epoch + 1
                    torch.save(model, '{:s}pair_model_checkpoint.pt'.format(
                        args['dir'] + args['model_output_dir']
                    ))
                    print('Best model so far! Saved checkpoint.')

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60
    ))
    print('Best Val Spearman: {:4f} at Epoch: {:d}'.format(
        best_spearman, best_epoch
    ))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_epoch, losses, accuracies, spearmans


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


def evaluate(docs, device, model_fsent, model_pair, tokenizer, num_samples=None):
    if num_samples:
        set_seed()
        docs = random.sample(docs, num_samples)
    running_spearman = 0
    worst_spearman_docs, worst_spearman = [], float('inf')
    with tqdm(total=len(docs)) as pbar:
        for i, doc in enumerate(docs):
            gold_index = doc['indexes']
            pred_index = predict(doc, device, model_fsent,
                                 model_pair, tokenizer)
            spearman, _ = spearmanr(gold_index, pred_index)
            running_spearman += spearman
            if spearman < worst_spearman:
                worst_spearman = spearman
                worst_spearman_docs = [doc]
            elif spearman == worst_spearman:
                worst_spearman_docs.append(doc)
            pbar.update(1)
            pbar.set_postfix(
                spearman=(running_spearman / (i + 1))
            )
            time.sleep(0.1)
    return running_spearman / len(docs), worst_spearman_docs


def predict(doc, device, model_fsent, model_pair, tokenizer):
    # predict first sentence position
    fsent_pos = predict_fsent_pos(doc, device, model_fsent, tokenizer)
    # predict sentence pairs logits
    pair_logits = predict_pair_logits(doc, device, model_pair, tokenizer)
    # construct graph based on predictions
    num_vertices = len(doc['sentences'])
    graph = make_graph(num_vertices, pair_logits)
    # sort and get the order
    if args['graph_method'] == 'max_flow':
        order, weight = graph.max_flow(fsent_pos)
    elif args['graph_method'] == 'greedy':
        order, weight = graph.greedy(fsent_pos)
    else:
        raise InvalidInputError()
    # get indices
    indexes = [-1] * num_vertices
    for p, o in enumerate(order):
        indexes[o] = p
    return indexes


def predict_fsent_pos(doc, device, model, tokenizer):
    model.eval()
    fsent_pos, max_logit = None, float('-inf')
    for i, text in enumerate(doc['sentences']):
        cleaned_text = clean_text(text)
        encoding = tokenizer.encode_plus(
            cleaned_text,
            add_special_tokens=True,
            max_length=args['model_fsent']['max_length'],
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
    model.eval()
    pair_logits = []
    perms = list(combinations(enumerate(doc['sentences']), 2))
    for (a, text_a), (b, text_b) in perms:
        cleaned_text_a = clean_text(text_a)
        cleaned_text_b = clean_text(text_b)

        cleaned_text_a, cleaned_text_b = truncate_texts(
            cleaned_text_a,
            cleaned_text_b,
            tokenizer,
            args['model_pair']['max_length']
        )

        encoding = tokenizer.encode_plus(
            cleaned_text_a,
            cleaned_text_b,
            add_special_tokens=True,
            max_length=args['model_pair']['max_length'],
            return_token_type_ids=True,
            padding='max_length',
            truncation=False,
            return_attention_mask=True,
            return_tensors='pt'
        )

        assert len(encoding['input_ids']) <= args['model_pair']['max_length']

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
        graph.add_edge(a, b, logit1)
        graph.add_edge(b, a, logit0)
    return graph
