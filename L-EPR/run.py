import os
import argparse
import numpy as np
import pandas as pd
import csv
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, RobertaTokenizer, AutoTokenizer

from bert import BERTModelConfig, BERTClassifier
from roberta import RoBERTaModelConfig, RoBERTaClassifier
from codebert import CodeBERTModelConfig, CodeBERTClassifier
from unixcoder import UniXcoderModelConfig, UniXcoderClassifier
from sptcode import SPTCodeModelConfig, SPTCodeClassifier
from graphcodebert import GraphCodeBERTModelConfig, GraphCodeBERTClassifier
from codet5p import CodeT5PModelConfig, CodeT5PClassifier

from nltk import word_tokenize
from data.vocab import Vocab, load_vocab
from utils import train_word2vec, DataProcess, NPRDataset, get_batch_inputs, data_transformer, Sample, TextDataset
from lstm import LSTMModelConfig, LSTMClassifier
from textcnn import TextCNNModelConfig, TextCNNClassifier
from train_eval import train_eval_model, train_eval_model_2, train_eval_model_dfg, train_eval_model_seq

parser = argparse.ArgumentParser(description='NPRs Classification')
parser.add_argument('--data', type=str, required=True, help='choose a type of dataset')
parser.add_argument('--model', type=str, required=True, help='choose a model')
args = parser.parse_args()

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    data_type = args.data
    train_data_path = ''
    test_data_path = ''
    if data_type == 'code':
        train_data_path = 'dataset/train1_code.csv'
        valid_data_path = 'dataset/valid1_code.csv'
        test_data_path = 'dataset/defects4j_code.csv'
        test_data_path_2 = 'dataset/test5000_code.csv'
        max_length = 512
    elif data_type == 'rename':
        train_data_path = 'dataset/train1_rename.csv'
        valid_data_path = 'dataset/valid1_rename.csv'
        test_data_path = 'dataset/test_code_rename.csv'
        test_data_path_2 = 'dataset/test_code_rename_5000.csv'
        max_length = 512
    elif data_type == 'dfg':
        train_data_path = 'dataset/train1_code.csv'
        valid_data_path = 'dataset/valid1_code.csv'
        test_data_path = 'dataset/defects4j_code.csv'
        test_data_path_2 = 'dataset/test5000_code.csv'
        # train_data_path = 'dataset/train1_rename.csv'
        # valid_data_path = 'dataset/valid1_rename.csv'
        # test_data_path = 'dataset/test_code_rename.csv'
        # test_data_path_2 = 'dataset/test_code_rename_5000.csv'
        max_length = 512
        dfg_max_length = 128

    train_data = pd.read_csv(train_data_path, delimiter='\t', header=0)
    valid_data = pd.read_csv(valid_data_path, delimiter='\t', header=0)
    test_data = pd.read_csv(test_data_path, delimiter='\t', header=0)
    test_data_2 = pd.read_csv(test_data_path_2, delimiter='\t', header=0)
    train_text = list(train_data['text'].values)
    valid_text = list(valid_data['text'].values)
    test_text = list(test_data['text'].values)
    test_text_2 = list(test_data_2['text'].values)
    train_labels = []
    with open(train_data_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        i = 0
        for row in reader:
            if i == 0:
                i = 1
                continue
            label = [int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5]), int(row[6])]
            train_labels.append(label)
    valid_labels = []
    with open(valid_data_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        i = 0
        for row in reader:
            if i == 0:
                i = 1
                continue
            label = [int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5]), int(row[6])]
            valid_labels.append(label)
    test_labels = []
    with open(test_data_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        i = 0
        for row in reader:
            if i == 0:
                i = 1
                continue
            label = [int(row[2]), int(row[3]), int(row[4]), int(row[5]), int(row[6]), int(row[7])]
            test_labels.append(label)
    test_labels_2 = []
    with open(test_data_path_2, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        i = 0
        for row in reader:
            if i == 0:
                i = 1
                continue
            label = [int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5]), int(row[6])]
            test_labels_2.append(label)

    model_name = args.model
    if model_name == 'bert':
        model_config = BERTModelConfig()
        if not os.path.exists(model_config.save_path):
            os.makedirs(model_config.save_path)

        model = BERTClassifier(model_config.bert_path, model_config.hidden_dim, model_config.output_size)

        tokenizer = BertTokenizer.from_pretrained(model_config.bert_path)
        train_text_id = tokenizer(train_text, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        valid_text_id = tokenizer(valid_text, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        test_text_id = tokenizer(test_text, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        test_text_id_2 = tokenizer(test_text_2, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        x_train = torch.stack([train_text_id['input_ids'], train_text_id['attention_mask']], 1)
        y_train = torch.from_numpy(np.array(train_labels)).float()
        x_valid = torch.stack([valid_text_id['input_ids'], valid_text_id['attention_mask']], 1)
        y_valid = torch.from_numpy(np.array(valid_labels)).float()
        x_test = torch.stack([test_text_id['input_ids'], test_text_id['attention_mask']], 1)
        y_test = torch.from_numpy(np.array(test_labels)).float()
        x_test_2 = torch.stack([test_text_id_2['input_ids'], test_text_id_2['attention_mask']], 1)
        y_test_2 = torch.from_numpy(np.array(test_labels_2)).float()
        train_data = TensorDataset(x_train, y_train)
        valid_data = TensorDataset(x_valid, y_valid)
        test_data = TensorDataset(x_test, y_test)
        test_data_2 = TensorDataset(x_test_2, y_test_2)
        train_loader = DataLoader(train_data, shuffle=True, batch_size=model_config.batch_size, drop_last=True)
        valid_loader = DataLoader(valid_data, shuffle=False, batch_size=1, drop_last=False)
        test_loader = DataLoader(test_data, shuffle=False, batch_size=1, drop_last=False)
        test_loader_2 = DataLoader(test_data_2, shuffle=False, batch_size=1, drop_last=False)

        if model_config.use_cuda:
            print('Run on GPU.')
        else:
            print('No GPU available, run on CPU.')

        train_eval_model(model_config, model, train_loader, valid_loader, test_loader, test_loader_2)
    elif model_name == 'roberta':
        model_config = RoBERTaModelConfig()
        if not os.path.exists(model_config.save_path):
            os.makedirs(model_config.save_path)

        model = RoBERTaClassifier(model_config.roberta_path, model_config.hidden_dim, model_config.output_size)

        tokenizer = RobertaTokenizer.from_pretrained(model_config.roberta_path)
        train_text_id = tokenizer(train_text, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        valid_text_id = tokenizer(valid_text, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        test_text_id = tokenizer(test_text, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        x_train = torch.stack([train_text_id['input_ids'], train_text_id['attention_mask']], 1)
        y_train = torch.from_numpy(np.array(train_labels)).float()
        x_valid = torch.stack([valid_text_id['input_ids'], valid_text_id['attention_mask']], 1)
        y_valid = torch.from_numpy(np.array(valid_labels)).float()
        x_test = torch.stack([test_text_id['input_ids'], test_text_id['attention_mask']], 1)
        y_test = torch.from_numpy(np.array(test_labels)).float()
        train_data = TensorDataset(x_train, y_train)
        valid_data = TensorDataset(x_valid, y_valid)
        test_data = TensorDataset(x_test, y_test)
        train_loader = DataLoader(train_data, shuffle=True, batch_size=model_config.batch_size, drop_last=True)
        valid_loader = DataLoader(valid_data, shuffle=False, batch_size=1, drop_last=False)
        test_loader = DataLoader(test_data, shuffle=False, batch_size=1, drop_last=False)

        if model_config.use_cuda:
            print('Run on GPU.')
        else:
            print('No GPU available, run on CPU.')

        train_eval_model(model_config, model, train_loader, valid_loader, test_loader)
    elif model_name == 'codebert':
        model_config = CodeBERTModelConfig()
        if not os.path.exists(model_config.save_path):
            os.mkdir(model_config.save_path)

        model = CodeBERTClassifier(model_config.codebert_path, model_config.hidden_dim, model_config.output_size)

        tokenizer = RobertaTokenizer.from_pretrained(model_config.codebert_path)
        train_text_id = tokenizer(train_text, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        valid_text_id = tokenizer(valid_text, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        test_text_id = tokenizer(test_text, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        test_text_id_2 = tokenizer(test_text_2, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        x_train = torch.stack([train_text_id['input_ids'], train_text_id['attention_mask']], 1)
        y_train = torch.from_numpy(np.array(train_labels)).float()
        x_valid = torch.stack([valid_text_id['input_ids'], valid_text_id['attention_mask']], 1)
        y_valid = torch.from_numpy(np.array(valid_labels)).float()
        x_test = torch.stack([test_text_id['input_ids'], test_text_id['attention_mask']], 1)
        y_test = torch.from_numpy(np.array(test_labels)).float()
        x_test_2 = torch.stack([test_text_id_2['input_ids'], test_text_id_2['attention_mask']], 1)
        y_test_2 = torch.from_numpy(np.array(test_labels_2)).float()
        train_data = TensorDataset(x_train, y_train)
        valid_data = TensorDataset(x_valid, y_valid)
        test_data = TensorDataset(x_test, y_test)
        test_data_2 = TensorDataset(x_test_2, y_test_2)
        train_loader = DataLoader(train_data, shuffle=True, batch_size=model_config.batch_size, drop_last=True)
        valid_loader = DataLoader(valid_data, shuffle=False, batch_size=1, drop_last=False)
        test_loader = DataLoader(test_data, shuffle=False, batch_size=1, drop_last=False)
        test_loader_2 = DataLoader(test_data_2, shuffle=False, batch_size=1, drop_last=False)

        if model_config.use_cuda:
            print('Run on GPU.')
        else:
            print('No GPU available, run on CPU.')

        train_eval_model(model_config, model, train_loader, valid_loader, test_loader, test_loader_2)
    elif model_name == 'unixcoder':
        model_config = UniXcoderModelConfig()
        if not os.path.exists(model_config.save_path):
            os.mkdir(model_config.save_path)

        model = UniXcoderClassifier(model_config.unixcoder_path, model_config.hidden_dim, model_config.output_size)

        tokenizer = RobertaTokenizer.from_pretrained(model_config.unixcoder_path)
        train_text_id = tokenizer(train_text, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        valid_text_id = tokenizer(valid_text, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        test_text_id = tokenizer(test_text, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        test_text_id_2 = tokenizer(test_text_2, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        x_train = torch.stack([train_text_id['input_ids'], train_text_id['attention_mask']], 1)
        y_train = torch.from_numpy(np.array(train_labels)).float()
        x_valid = torch.stack([valid_text_id['input_ids'], valid_text_id['attention_mask']], 1)
        y_valid = torch.from_numpy(np.array(valid_labels)).float()
        x_test = torch.stack([test_text_id['input_ids'], test_text_id['attention_mask']], 1)
        y_test = torch.from_numpy(np.array(test_labels)).float()
        x_test_2 = torch.stack([test_text_id_2['input_ids'], test_text_id_2['attention_mask']], 1)
        y_test_2 = torch.from_numpy(np.array(test_labels_2)).float()
        train_data = TensorDataset(x_train, y_train)
        valid_data = TensorDataset(x_valid, y_valid)
        test_data = TensorDataset(x_test, y_test)
        test_data_2 = TensorDataset(x_test_2, y_test_2)
        train_loader = DataLoader(train_data, shuffle=True, batch_size=model_config.batch_size, drop_last=True)
        valid_loader = DataLoader(valid_data, shuffle=False, batch_size=1, drop_last=False)
        test_loader = DataLoader(test_data, shuffle=False, batch_size=1, drop_last=False)
        test_loader_2 = DataLoader(test_data_2, shuffle=False, batch_size=1, drop_last=False)

        if model_config.use_cuda:
            print('Run on GPU.')
        else:
            print('No GPU available, run on CPU.')

        train_eval_model(model_config, model, train_loader, valid_loader, test_loader, test_loader_2)
    elif model_name == 'sptcode':
        model_config = SPTCodeModelConfig()
        if not os.path.exists(model_config.save_path):
            os.mkdir(model_config.save_path)

        model = SPTCodeClassifier(model_config.sptcode_path, model_config.hidden_dim, model_config.output_size)

        code_vocab = load_vocab(vocab_root=model_config.vocab_path, name='code')

        train_text_id = {}
        train_text_id['input_ids'], train_text_id['attention_mask'] = get_batch_inputs(batch=train_text, vocab=code_vocab, processor=Vocab.sos_processor, max_len=max_length)
        valid_text_id = {}
        valid_text_id['input_ids'], valid_text_id['attention_mask'] = get_batch_inputs(batch=valid_text, vocab=code_vocab, processor=Vocab.sos_processor, max_len=max_length)
        test_text_id = {}
        test_text_id['input_ids'], test_text_id['attention_mask'] = get_batch_inputs(batch=test_text, vocab=code_vocab, processor=Vocab.sos_processor, max_len=max_length)
        x_train = torch.stack([train_text_id['input_ids'], train_text_id['attention_mask']], 1)
        y_train = torch.from_numpy(np.array(train_labels)).float()
        x_valid = torch.stack([valid_text_id['input_ids'], valid_text_id['attention_mask']], 1)
        y_valid = torch.from_numpy(np.array(valid_labels)).float()
        x_test = torch.stack([test_text_id['input_ids'], test_text_id['attention_mask']], 1)
        y_test = torch.from_numpy(np.array(test_labels)).float()
        train_data = TensorDataset(x_train, y_train)
        valid_data = TensorDataset(x_valid, y_valid)
        test_data = TensorDataset(x_test, y_test)
        train_loader = DataLoader(train_data, shuffle=True, batch_size=model_config.batch_size, drop_last=True)
        valid_loader = DataLoader(valid_data, shuffle=False, batch_size=1, drop_last=False)
        test_loader = DataLoader(test_data, shuffle=False, batch_size=1, drop_last=False)

        if model_config.use_cuda:
            print('Run on GPU.')
        else:
            print('No GPU available, run on CPU.')

        train_eval_model(model_config, model, train_loader, valid_loader, test_loader)
    elif model_name == 'graphcodebert':
        model_config = GraphCodeBERTModelConfig()
        if not os.path.exists(model_config.save_path):
            os.mkdir(model_config.save_path)

        model = GraphCodeBERTClassifier(model_config.graphcodebert_path, model_config.hidden_dim, model_config.output_size)

        tokenizer = RobertaTokenizer.from_pretrained(model_config.graphcodebert_path)

        x_train = []
        x_valid = []
        x_test = []
        x_test_2 = []
        for text in train_text:
            code_ids, position_idx, attn_mask = data_transformer(tokenizer, text, max_length, dfg_max_length)
            x_train.append(Sample(code_ids, position_idx, attn_mask))
        for text in valid_text:
            code_ids, position_idx, attn_mask = data_transformer(tokenizer, text, max_length, dfg_max_length)
            x_valid.append(Sample(code_ids, position_idx, attn_mask))
        for text in test_text:
            code_ids, position_idx, attn_mask = data_transformer(tokenizer, text, max_length, dfg_max_length)
            x_test.append(Sample(code_ids, position_idx, attn_mask))
        for text in test_text_2:
            code_ids, position_idx, attn_mask = data_transformer(tokenizer, text, max_length, dfg_max_length)
            x_test_2.append(Sample(code_ids, position_idx, attn_mask))

        y_train = torch.from_numpy(np.array(train_labels)).float()
        y_valid = torch.from_numpy(np.array(valid_labels)).float()
        y_test = torch.from_numpy(np.array(test_labels)).float()
        y_test_2 = torch.from_numpy(np.array(test_labels_2)).float()
        train_data = TextDataset(x_train, y_train)
        valid_data = TextDataset(x_valid, y_valid)
        test_data = TextDataset(x_test, y_test)
        test_data_2 = TextDataset(x_test_2, y_test_2)
        train_loader = DataLoader(train_data, shuffle=True, batch_size=model_config.batch_size, drop_last=True)
        valid_loader = DataLoader(valid_data, shuffle=False, batch_size=1, drop_last=False)
        test_loader = DataLoader(test_data, shuffle=False, batch_size=1, drop_last=False)
        test_loader_2 = DataLoader(test_data_2, shuffle=False, batch_size=1, drop_last=False)

        if model_config.use_cuda:
            print('Run on GPU.')
        else:
            print('No GPU available, run on CPU.')

        train_eval_model_dfg(model_config, model, train_loader, valid_loader, test_loader, test_loader_2)
    elif model_name == 'codet5p':
        model_config = CodeT5PModelConfig()
        if not os.path.exists(model_config.save_path):
            os.mkdir(model_config.save_path)

        model = CodeT5PClassifier(model_config.codet5p_path)

        tokenizer = AutoTokenizer.from_pretrained(model_config.codet5p_path)

        train_text_id = tokenizer(train_text, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        valid_text_id = tokenizer(valid_text, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        test_text_id = tokenizer(test_text, padding=True, truncation=True, max_length=max_length, return_tensors='pt')

        x_train = torch.stack([train_text_id['input_ids'], train_text_id['attention_mask']], 1)
        y_train_str = []
        for labels in train_labels:
            labels_str = ''
            if labels[0] == 1:
                labels_str += 'recoder' + ' '
            if labels[1] == 1:
                labels_str += 'tare' + ' '
            if labels[2] == 1:
                labels_str += 'rewardrepair' + ' '
            if labels[3] == 1:
                labels_str += 'selfapr' + ' '
            if labels[4] == 1:
                labels_str += 'gamma' + ' '
            if labels[5] == 1:
                labels_str += 'failure' + ' '
            y_train_str.append(labels_str.strip())
        train_labels_id = tokenizer(y_train_str, padding=True, truncation=True, max_length=8, return_tensors='pt')
        y_train = train_labels_id['input_ids']

        x_valid = torch.stack([valid_text_id['input_ids'], valid_text_id['attention_mask']], 1)
        y_valid_str = []
        for labels in valid_labels:
            labels_str = ''
            if labels[0] == 1:
                labels_str += 'recoder' + ' '
            if labels[1] == 1:
                labels_str += 'tare' + ' '
            if labels[2] == 1:
                labels_str += 'rewardrepair' + ' '
            if labels[3] == 1:
                labels_str += 'selfapr' + ' '
            if labels[4] == 1:
                labels_str += 'gamma' + ' '
            if labels[5] == 1:
                labels_str += 'failure' + ' '
            y_valid_str.append(labels_str.strip())
        valid_labels_id = tokenizer(y_valid_str, padding=True, truncation=True, max_length=8, return_tensors='pt')
        y_valid = valid_labels_id['input_ids']

        x_test = torch.stack([test_text_id['input_ids'], test_text_id['attention_mask']], 1)
        y_test_str = []
        for labels in test_labels:
            labels_str = ''
            if labels[0] == 1:
                labels_str += 'recoder' + ' '
            if labels[1] == 1:
                labels_str += 'tare' + ' '
            if labels[2] == 1:
                labels_str += 'rewardrepair' + ' '
            if labels[3] == 1:
                labels_str += 'selfapr' + ' '
            if labels[4] == 1:
                labels_str += 'gamma' + ' '
            if labels[5] == 1:
                labels_str += 'failure' + ' '
            y_test_str.append(labels_str.strip())
        test_labels_id = tokenizer(y_test_str, padding=True, truncation=True, max_length=8, return_tensors='pt')
        y_test = test_labels_id['input_ids']

        train_data = TensorDataset(x_train, y_train)
        valid_data = TensorDataset(x_valid, y_valid)
        test_data = TensorDataset(x_test, y_test)
        train_loader = DataLoader(train_data, shuffle=True, batch_size=model_config.batch_size, drop_last=True)
        valid_loader = DataLoader(valid_data, shuffle=False, batch_size=1, drop_last=False)
        test_loader = DataLoader(test_data, shuffle=False, batch_size=1, drop_last=False)

        if model_config.use_cuda:
            print('Run on GPU.')
        else:
            print('No GPU available, run on CPU.')

        train_eval_model(model_config, model, tokenizer, train_loader, valid_loader, test_loader)
    elif model_name == 'lstm':
        model_config = LSTMModelConfig()
        if not os.path.exists(model_config.save_path):
            os.makedirs(model_config.save_path)

        train_text_new = []
        valid_text_new = []
        test_text_new = []
        for text in train_text:
            train_text_new.append(word_tokenize(text))
        for text in valid_text:
            valid_text_new.append(word_tokenize(text))
        for text in test_text:
            test_text_new.append(word_tokenize(text))

        word2vec = train_word2vec(train_text_new + valid_text_new + test_text_new + train_labels)
        print("saving word2vec model...")
        word2vec.save(os.path.join(model_config.save_path, 'word2vec.model'))
        print("word2vec is saved successfully!")

        preprocess_train = DataProcess(train_text_new, max_length, os.path.join(model_config.save_path, 'word2vec.model'))
        preprocess_valid = DataProcess(valid_text_new, max_length, os.path.join(model_config.save_path, 'word2vec.model'))
        preprocess_test = DataProcess(test_text_new, max_length, os.path.join(model_config.save_path, 'word2vec.model'))
        embedding_train = preprocess_train.make_embedding()
        embedding_valid = preprocess_valid.make_embedding()
        embedding_test = preprocess_test.make_embedding()
        x_train = preprocess_train.sentence_word2idx()
        y_train = torch.from_numpy(np.array(train_labels)).float()
        x_valid = preprocess_valid.sentence_word2idx()
        y_valid = torch.from_numpy(np.array(valid_labels)).float()
        x_test = preprocess_test.sentence_word2idx()
        y_test = torch.from_numpy(np.array(test_labels)).float()
        train_data = NPRDataset(x_train, y_train)
        valid_data = NPRDataset(x_valid, y_valid)
        test_data = NPRDataset(x_test, y_test)
        train_loader = DataLoader(train_data, shuffle=True, batch_size=model_config.batch_size, drop_last=True)
        valid_loader = DataLoader(valid_data, shuffle=False, batch_size=1, drop_last=False)
        test_loader = DataLoader(test_data, shuffle=False, batch_size=1, drop_last=False)

        model = LSTMClassifier(embedding_train, model_config.embedding_dim, model_config.hidden_dim, model_config.num_layers, model_config.output_size)

        if model_config.use_cuda:
            print('Run on GPU.')
        else:
            print('No GPU available, run on CPU.')

        train_eval_model(model_config, model, train_loader, valid_loader, test_loader)
    elif model_name == 'textcnn':
        model_config = TextCNNModelConfig()
        if not os.path.exists(model_config.save_path):
            os.makedirs(model_config.save_path)

        train_text_new = []
        valid_text_new = []
        test_text_new = []
        for text in train_text:
            train_text_new.append(word_tokenize(text))
        for text in valid_text:
            valid_text_new.append(word_tokenize(text))
        for text in test_text:
            test_text_new.append(word_tokenize(text))

        word2vec = train_word2vec(train_text_new + valid_text_new + test_text_new + train_labels)
        print("saving word2vec model...")
        word2vec.save(os.path.join(model_config.save_path, 'word2vec.model'))
        print("word2vec is saved-sorted successfully!")

        preprocess_train = DataProcess(train_text_new, max_length, os.path.join(model_config.save_path, 'word2vec.model'))
        preprocess_valid = DataProcess(valid_text_new, max_length, os.path.join(model_config.save_path, 'word2vec.model'))
        preprocess_test = DataProcess(test_text_new, max_length, os.path.join(model_config.save_path, 'word2vec.model'))
        embedding_train = preprocess_train.make_embedding()
        embedding_valid = preprocess_valid.make_embedding()
        embedding_test = preprocess_test.make_embedding()
        x_train = preprocess_train.sentence_word2idx()
        y_train = torch.from_numpy(np.array(train_labels)).float()
        x_valid = preprocess_valid.sentence_word2idx()
        y_valid = torch.from_numpy(np.array(valid_labels)).float()
        x_test = preprocess_test.sentence_word2idx()
        y_test = torch.from_numpy(np.array(test_labels)).float()
        train_data = NPRDataset(x_train, y_train)
        valid_data = NPRDataset(x_valid, y_valid)
        test_data = NPRDataset(x_test, y_test)
        train_loader = DataLoader(train_data, shuffle=True, batch_size=model_config.batch_size, drop_last=True)
        valid_loader = DataLoader(valid_data, shuffle=False, batch_size=1, drop_last=False)
        test_loader = DataLoader(test_data, shuffle=False, batch_size=1, drop_last=False)

        model = TextCNNClassifier(embedding_train, model_config.num_filters, model_config.embedding_dim, model_config.filter_sizes, model_config.output_size)

        if model_config.use_cuda:
            print('Run on GPU.')
        else:
            print('No GPU available, run on CPU.')

        train_eval_model(model_config, model, train_loader, valid_loader, test_loader)
    elif model_name == 'lstm-2':
        model_config = LSTMModelConfig()
        if not os.path.exists(model_config.save_path):
            os.makedirs(model_config.save_path)

        train_text_new = []
        valid_text_new = []
        test_text_new = []
        for text in train_text:
            train_text_new.append(word_tokenize(text))
        for text in valid_text:
            valid_text_new.append(word_tokenize(text))
        for text in test_text:
            test_text_new.append(word_tokenize(text))

        word2vec = train_word2vec(train_text_new + valid_text_new + test_text_new + train_labels)
        print("saving word2vec model...")
        word2vec.save(os.path.join(model_config.save_path, 'word2vec.model'))
        print("word2vec is saved successfully!")

        preprocess_train = DataProcess(train_text_new, max_length, os.path.join(model_config.save_path, 'word2vec.model'))
        preprocess_valid = DataProcess(valid_text_new, max_length, os.path.join(model_config.save_path, 'word2vec.model'))
        preprocess_test = DataProcess(test_text_new, max_length, os.path.join(model_config.save_path, 'word2vec.model'))
        embedding_train = preprocess_train.make_embedding()
        embedding_valid = preprocess_valid.make_embedding()
        embedding_test = preprocess_test.make_embedding()

        train_labels_recoder = []
        train_labels_tare = []
        train_labels_rewardrepair = []
        train_labels_selfapr = []
        train_labels_gamma = []
        train_labels_allfailure = []
        valid_labels_recoder = []
        valid_labels_tare = []
        valid_labels_rewardrepair = []
        valid_labels_selfapr = []
        valid_labels_gamma = []
        valid_labels_allfailure = []
        test_labels_recoder = []
        test_labels_tare = []
        test_labels_rewardrepair = []
        test_labels_selfapr = []
        test_labels_gamma = []
        test_labels_allfailure = []

        for labels in train_labels:
            train_labels_recoder.append(labels[0])
            train_labels_tare.append(labels[1])
            train_labels_rewardrepair.append(labels[2])
            train_labels_selfapr.append(labels[3])
            train_labels_gamma.append(labels[4])
            train_labels_allfailure.append(labels[5])
        for labels in valid_labels:
            valid_labels_recoder.append(labels[0])
            valid_labels_tare.append(labels[1])
            valid_labels_rewardrepair.append(labels[2])
            valid_labels_selfapr.append(labels[3])
            valid_labels_gamma.append(labels[4])
            valid_labels_allfailure.append(labels[5])
        for labels in test_labels:
            test_labels_recoder.append(labels[0])
            test_labels_tare.append(labels[1])
            test_labels_rewardrepair.append(labels[2])
            test_labels_selfapr.append(labels[3])
            test_labels_gamma.append(labels[4])
            test_labels_allfailure.append(labels[5])

        x_train = preprocess_train.sentence_word2idx()
        y_train_recoder = torch.from_numpy(np.array(train_labels_recoder)).float()
        y_train_tare = torch.from_numpy(np.array(train_labels_tare)).float()
        y_train_rewardrepair = torch.from_numpy(np.array(train_labels_rewardrepair)).float()
        y_train_selfapr = torch.from_numpy(np.array(train_labels_selfapr)).float()
        y_train_gamma = torch.from_numpy(np.array(train_labels_gamma)).float()
        y_train_allfailure = torch.from_numpy(np.array(train_labels_allfailure)).float()

        x_valid = preprocess_valid.sentence_word2idx()
        y_valid_recoder = torch.from_numpy(np.array(valid_labels_recoder)).float()
        y_valid_tare = torch.from_numpy(np.array(valid_labels_tare)).float()
        y_valid_rewardrepair = torch.from_numpy(np.array(valid_labels_rewardrepair)).float()
        y_valid_selfapr = torch.from_numpy(np.array(valid_labels_selfapr)).float()
        y_valid_gamma = torch.from_numpy(np.array(valid_labels_gamma)).float()
        y_valid_allfailure = torch.from_numpy(np.array(valid_labels_allfailure)).float()

        x_test = preprocess_test.sentence_word2idx()
        y_test_recoder = torch.from_numpy(np.array(test_labels_recoder)).float()
        y_test_tare = torch.from_numpy(np.array(test_labels_tare)).float()
        y_test_rewardrepair = torch.from_numpy(np.array(test_labels_rewardrepair)).float()
        y_test_selfapr = torch.from_numpy(np.array(test_labels_selfapr)).float()
        y_test_gamma = torch.from_numpy(np.array(test_labels_gamma)).float()
        y_test_allfailure = torch.from_numpy(np.array(test_labels_allfailure)).float()

        train_data_recoder = NPRDataset(x_train, y_train_recoder)
        train_data_tare = NPRDataset(x_train, y_train_tare)
        train_data_rewardrepair = NPRDataset(x_train, y_train_rewardrepair)
        train_data_selfapr = NPRDataset(x_train, y_train_selfapr)
        train_data_gamma = NPRDataset(x_train, y_train_gamma)
        train_data_allfailure = NPRDataset(x_train, y_train_allfailure)
        valid_data_recoder = NPRDataset(x_valid, y_valid_recoder)
        valid_data_tare = NPRDataset(x_valid, y_valid_tare)
        valid_data_rewardrepair = NPRDataset(x_valid, y_valid_rewardrepair)
        valid_data_selfapr = NPRDataset(x_valid, y_valid_selfapr)
        valid_data_gamma = NPRDataset(x_valid, y_valid_gamma)
        valid_data_allfailure = NPRDataset(x_valid, y_valid_allfailure)
        test_data_recoder = NPRDataset(x_test, y_test_recoder)
        test_data_tare = NPRDataset(x_test, y_test_tare)
        test_data_rewardrepair = NPRDataset(x_test, y_test_rewardrepair)
        test_data_selfapr = NPRDataset(x_test, y_test_selfapr)
        test_data_gamma = NPRDataset(x_test, y_test_gamma)
        test_data_allfailure = NPRDataset(x_test, y_test_allfailure)

        train_loader_recoder = DataLoader(train_data_recoder, shuffle=True, batch_size=model_config.batch_size, drop_last=True)
        valid_loader_recoder = DataLoader(valid_data_recoder, shuffle=False, batch_size=1, drop_last=False)
        test_loader_recoder = DataLoader(test_data_recoder, shuffle=False, batch_size=1, drop_last=False)
        train_loader_tare = DataLoader(train_data_tare, shuffle=True, batch_size=model_config.batch_size, drop_last=True)
        valid_loader_tare = DataLoader(valid_data_tare, shuffle=False, batch_size=1, drop_last=False)
        test_loader_tare = DataLoader(test_data_tare, shuffle=False, batch_size=1, drop_last=False)
        train_loader_rewardrepair = DataLoader(train_data_rewardrepair, shuffle=True, batch_size=model_config.batch_size, drop_last=True)
        valid_loader_rewardrepair = DataLoader(valid_data_rewardrepair, shuffle=False, batch_size=1, drop_last=False)
        test_loader_rewardrepair = DataLoader(test_data_rewardrepair, shuffle=False, batch_size=1, drop_last=False)
        train_loader_selfapr = DataLoader(train_data_selfapr, shuffle=True, batch_size=model_config.batch_size, drop_last=True)
        valid_loader_selfapr = DataLoader(valid_data_selfapr, shuffle=False, batch_size=1, drop_last=False)
        test_loader_selfapr = DataLoader(test_data_selfapr, shuffle=False, batch_size=1, drop_last=False)
        train_loader_gamma = DataLoader(train_data_gamma, shuffle=True, batch_size=model_config.batch_size, drop_last=True)
        valid_loader_gamma = DataLoader(valid_data_gamma, shuffle=False, batch_size=1, drop_last=False)
        test_loader_gamma = DataLoader(test_data_gamma, shuffle=False, batch_size=1, drop_last=False)
        train_loader_allfailure = DataLoader(train_data_allfailure, shuffle=True, batch_size=model_config.batch_size, drop_last=True)
        valid_loader_allfailure = DataLoader(valid_data_allfailure, shuffle=False, batch_size=1, drop_last=False)
        test_loader_allfailure = DataLoader(test_data_allfailure, shuffle=False, batch_size=1, drop_last=False)

        model = LSTMClassifier(embedding_train, model_config.embedding_dim, model_config.hidden_dim, model_config.num_layers, model_config.output_size)

        if model_config.use_cuda:
            print('Run on GPU.')
        else:
            print('No GPU available, run on CPU.')

        train_loader_list = [train_loader_recoder, train_loader_tare, train_loader_rewardrepair, train_loader_selfapr, train_loader_gamma, train_loader_allfailure]
        valid_loader_list = [valid_loader_recoder, valid_loader_tare, valid_loader_rewardrepair, valid_loader_selfapr, valid_loader_gamma, train_loader_allfailure]
        test_loader_list = [test_loader_recoder, test_loader_tare, test_loader_rewardrepair, test_loader_selfapr, test_loader_gamma, test_loader_allfailure]

        train_eval_model_2(model_config, model, train_loader_list, valid_loader_list, test_loader_list)
