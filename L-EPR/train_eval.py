import os
import torch
import torch.nn as nn
import xlsxwriter as xw
from tqdm import tqdm
from bert import BERTClassifier
from lstm import LSTMClassifier
from textcnn import TextCNNClassifier
from roberta import RoBERTaClassifier
from codebert import CodeBERTClassifier
from unixcoder import UniXcoderClassifier
from sptcode import SPTCodeClassifier
from graphcodebert import GraphCodeBERTClassifier
from codet5p import CodeT5PClassifier


def npr_train_results(true_labels, predict_labels):
    correct = 0
    total = 0
    for (t_labels, p_labels) in zip(true_labels, predict_labels):
        total += 1
        if t_labels == p_labels:
            correct += 1
    return correct, total


def npr_test_results(true_labels, predict_labels):
    correct = 0
    total = 0
    repair = 0
    repair_total = 0
    for (t_labels, p_labels) in zip(true_labels, predict_labels):
        total += 1
        if t_labels == p_labels:
            correct += 1
        if t_labels[5] == 0:
            repair_total += 1
            if t_labels == p_labels:
                repair += 1

    return correct, total, repair, repair_total


def npr_test_results_sort(true_labels, predict_labels, topN):
    correct = 0
    total = 0
    for (t_labels, p_labels) in zip(true_labels, predict_labels):
        total += 1
        sorted_indices = sorted(range(len(p_labels)), key=lambda x: p_labels[x], reverse=True)
        flag_1 = False
        for i in range(topN):
            index = sorted_indices[i]
            if t_labels[index] == 1:
                flag_1 = True
                break
            else:
                continue
        if flag_1:
            correct += 1

    return correct, total


def npr_test_results_sort_repaired(true_labels, predict_labels, topN):
    correct = 0
    total = 0
    for (t_labels, p_labels) in zip(true_labels, predict_labels):
        if t_labels[5] == 0:
            total += 1
            sorted_indices = sorted(range(len(p_labels)), key=lambda x: p_labels[x], reverse=True)
            flag_1 = False
            for i in range(topN):
                index = sorted_indices[i]
                if t_labels[index] == 1:
                    flag_1 = True
                    break
                else:
                    continue
            if flag_1:
                correct += 1

    return correct, total


def npr_results_sort_seq(true_labels, probs_list, generated_ids_list, topN):
    correct = 0
    total = 0
    predict_labels = []
    for labels, probs, generated_ids in zip(true_labels, probs_list, generated_ids_list):
        t_labels = [0, 0, 0, 0, 0, 0]
        if 'coder' in labels:
            t_labels[0] = 1
        if 't' in labels:
            t_labels[1] = 1
        if 'reward' in labels:
            t_labels[2] = 1
        if 'self' in labels:
            t_labels[3] = 1
        if 'gamma' in labels:
            t_labels[4] = 1
        if 'failure' in labels:
            t_labels[5] = 1

        total += 1
        p_labels = [0, 0, 0, 0, 0, 0]
        for i, token_id in enumerate(generated_ids[0][1:]):
            token_prob = probs[i][0, token_id].item()
            if token_id == '3396':
                p_labels[0] = token_prob
            elif token_id == '268':
                p_labels[1] = token_prob
            elif token_id == '19890':
                p_labels[2] = token_prob
            elif token_id == '365':
                p_labels[3] = token_prob
            elif token_id == '9601':
                p_labels[4] = token_prob
            elif token_id == '5166':
                p_labels[5] = token_prob
        predict_labels.append(p_labels)

        sorted_indices = sorted(range(len(p_labels)), key=lambda x: p_labels[x], reverse=True)
        flag_1 = False
        for i in range(topN):
            index = sorted_indices[i]
            if t_labels[index] == 1:
                flag_1 = True
                break
            else:
                continue
        if flag_1:
            correct += 1

    return correct, total, predict_labels


def npr_results_sort_repaired_seq(true_labels, probs_list, generated_ids_list, topN):
    correct = 0
    total = 0
    for labels, probs, generated_ids in zip(true_labels, probs_list, generated_ids_list):
        t_labels = [0, 0, 0, 0, 0, 0]
        if 'coder' in labels:
            t_labels[0] = 1
        if 't' in labels:
            t_labels[1] = 1
        if 'reward' in labels:
            t_labels[2] = 1
        if 'self' in labels:
            t_labels[3] = 1
        if 'gamma' in labels:
            t_labels[4] = 1
        if 'failure' in labels:
            t_labels[5] = 1

        if t_labels[5] == 0:
            total += 1
            p_labels = [0, 0, 0, 0, 0, 0]
            for i, token_id in enumerate(generated_ids[0][1:]):
                token_prob = probs[i][0, token_id].item()
                if token_id == '3396':
                    p_labels[0] = token_prob
                elif token_id == '268':
                    p_labels[1] = token_prob
                elif token_id == '19890':
                    p_labels[2] = token_prob
                elif token_id == '365':
                    p_labels[3] = token_prob
                elif token_id == '9601':
                    p_labels[4] = token_prob
                elif token_id == '5166':
                    p_labels[5] = token_prob

            sorted_indices = sorted(range(len(p_labels)), key=lambda x: p_labels[x], reverse=True)
            flag_1 = False
            for i in range(topN):
                index = sorted_indices[i]
                if t_labels[index] == 1:
                    flag_1 = True
                    break
                else:
                    continue
            if flag_1:
                correct += 1

    return correct, total


def save_results_xlsx(xlsx_path, predict_labels):
    workbook = xw.Workbook(xlsx_path)

    worksheet6 = workbook.add_worksheet("top6")
    worksheet6.activate()
    title = ['pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5', 'pred_6']
    worksheet6.write_row('A1', title)
    i = 2
    for p_labels in predict_labels:
        sorted_indices = sorted(range(len(p_labels)), key=lambda x: p_labels[x], reverse=True)
        index_list = sorted_indices[0:6]
        row = 'A' + str(i)
        predict_nprs = []

        for index in index_list:
            if index == 0:
                predict_nprs.append('recoder')
            elif index == 1:
                predict_nprs.append('tare')
            elif index == 2:
                predict_nprs.append('rewardrepair')
            elif index == 3:
                predict_nprs.append('selfapr')
            elif index == 4:
                predict_nprs.append('gamma')
            elif index == 5:
                predict_nprs.append('allfailure')
        worksheet6.write_row(row, predict_nprs)
        i += 1

    workbook.close()
    print(xlsx_path + ' is saved successfully!')


def train_eval_model(config, model, data_train, data_valid, data_test, data_test_2):
    net = model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device)

    if isinstance(model, BERTClassifier):
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        criterion = nn.BCEWithLogitsLoss()
        save_model = 'bert'
    elif isinstance(model, CodeBERTClassifier):
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        criterion = nn.BCEWithLogitsLoss()
        save_model = 'codebert'
    elif isinstance(model, RoBERTaClassifier):
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        criterion = nn.BCEWithLogitsLoss()
        save_model = 'roberta'
    elif isinstance(model, UniXcoderClassifier):
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        criterion = nn.BCEWithLogitsLoss()
        save_model = 'unixcoder'
    elif isinstance(model, SPTCodeClassifier):
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        criterion = nn.BCEWithLogitsLoss()
        save_model = 'sptcode'
    elif isinstance(model, LSTMClassifier):
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        criterion = nn.BCEWithLogitsLoss()
        save_model = 'lstm'
    elif isinstance(model, TextCNNClassifier):
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        criterion = nn.BCEWithLogitsLoss()
        save_model = 'textcnn'

    net.train()

    for e in tqdm(range(config.epochs)):
        total_loss = []
        true_labels = []
        predict_labels = []

        valid_true_labels = []
        valid_predict_labels = []
        test_true_labels = []
        test_predict_labels = []
        test_true_labels_2 = []
        test_predict_labels_2 = []

        # train
        for inputs, labels in data_train:
            if config.use_cuda:
                inputs, labels = inputs.to(device), labels.to(device)

            net.zero_grad()
            output = net(inputs)

            true_labels += labels.cpu().numpy().tolist()
            predict_labels += torch.where(output > 0.5, 1, 0).cpu().numpy().tolist()

            loss = criterion(output.squeeze(), labels.float())
            total_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        train_correct, train_total = npr_train_results(true_labels, predict_labels)

        net.eval()

        # valid
        with torch.no_grad():
            for inputs, labels in data_valid:
                if config.use_cuda:
                    inputs, labels = inputs.to(device), labels.to(device)

                output = net(inputs)
                valid_true_labels += labels.cpu().numpy().tolist()
                valid_predict_labels += output.cpu().numpy().tolist()

        valid_correct_top1, valid_total = npr_test_results_sort(valid_true_labels, valid_predict_labels, 1)

        # test
        with torch.no_grad():
            for inputs, labels in data_test:
                if config.use_cuda:
                    inputs, labels = inputs.to(device), labels.to(device)

                output = net(inputs)
                test_true_labels += labels.cpu().numpy().tolist()
                test_predict_labels += output.cpu().numpy().tolist()
        with torch.no_grad():
            for inputs, labels in data_test_2:
                if config.use_cuda:
                    inputs, labels = inputs.to(device), labels.to(device)

                output = net(inputs)
                test_true_labels_2 += labels.cpu().numpy().tolist()
                test_predict_labels_2 += output.cpu().numpy().tolist()

        net.train()

        print("Epoch: {}/{}, ".format(e + 1, config.epochs),
              "Train Loss: {:.6f}, ".format(torch.tensor(total_loss).mean()),
              "Train repair acc: {}/{}.".format(train_correct, train_total))

        for topN in range(1, 7):
            if topN == 6:
                valid_correct, valid_total = npr_test_results_sort(valid_true_labels, valid_predict_labels, topN)
                test_correct, test_total = npr_test_results_sort(test_true_labels, test_predict_labels, topN)
                test_correct_2, test_total_2 = npr_test_results_sort(test_true_labels_2, test_predict_labels_2, topN)
                print("Top{} - valid acc: {}/{}; defects4j test acc: {}/{}; our test acc: {}/{}.".format(topN, valid_correct, valid_total, test_correct, test_total, test_correct_2,
                                                                                                         test_total_2))
            else:
                valid_correct, valid_total = npr_test_results_sort(valid_true_labels, valid_predict_labels, topN)
                valid_correct_repair, valid_total_repair = npr_test_results_sort_repaired(valid_true_labels, valid_predict_labels, topN)
                test_correct, test_total = npr_test_results_sort(test_true_labels, test_predict_labels, topN)
                test_correct_repair, test_total_repair = npr_test_results_sort_repaired(test_true_labels, test_predict_labels, topN)
                test_correct_2, test_total_2 = npr_test_results_sort(test_true_labels_2, test_predict_labels_2, topN)
                test_correct_repair_2, test_total_repair_2 = npr_test_results_sort_repaired(test_true_labels_2, test_predict_labels_2, topN)
                print(
                    "Top{} - valid acc: {}/{}, {}/{}; defects4j test acc: {}/{}, {}/{}; our test acc: {}/{}, {}/{}.".format(topN, valid_correct, valid_total, valid_correct_repair,
                                                                                                                            valid_total_repair, test_correct, test_total,
                                                                                                                            test_correct_repair, test_total_repair, test_correct_2,
                                                                                                                            test_total_2, test_correct_repair_2,
                                                                                                                            test_total_repair_2))
        print("--------------------------------------------------------------------------------------------------------")

        # record log
        with open(os.path.join(config.save_path, 'output_log.txt'), 'a', encoding='UTF-8') as f:
            f.write("Epoch: {}/{}, ".format(e + 1, config.epochs) +
                    "Train Loss: {:.6f}, ".format(torch.tensor(total_loss).mean()) +
                    "Train repair acc: {}/{}.".format(train_correct, train_total) + '\n')
            for topN in range(1, 7):
                if topN == 6:
                    valid_correct, valid_total = npr_test_results_sort(valid_true_labels, valid_predict_labels, topN)
                    test_correct, test_total = npr_test_results_sort(test_true_labels, test_predict_labels, topN)
                    test_correct_2, test_total_2 = npr_test_results_sort(test_true_labels_2, test_predict_labels_2, topN)
                    f.write("Top{} - valid acc: {}/{}; defects4j test acc: {}/{}; our test acc: {}/{}.".format(topN, valid_correct, valid_total, test_correct, test_total,
                                                                                                               test_correct_2, test_total_2) + '\n')
                else:
                    valid_correct, valid_total = npr_test_results_sort(valid_true_labels, valid_predict_labels, topN)
                    valid_correct_repair, valid_total_repair = npr_test_results_sort_repaired(valid_true_labels, valid_predict_labels, topN)
                    test_correct, test_total = npr_test_results_sort(test_true_labels, test_predict_labels, topN)
                    test_correct_repair, test_total_repair = npr_test_results_sort_repaired(test_true_labels, test_predict_labels, topN)
                    test_correct_2, test_total_2 = npr_test_results_sort(test_true_labels_2, test_predict_labels_2, topN)
                    test_correct_repair_2, test_total_repair_2 = npr_test_results_sort_repaired(test_true_labels_2, test_predict_labels_2, topN)
                    f.write(
                        "Top{} - valid acc: {}/{}, {}/{}; defects4j test acc: {}/{}, {}/{}; our test acc: {}/{}, {}/{}.".format(topN, valid_correct, valid_total,
                                                                                                                                valid_correct_repair, valid_total_repair,
                                                                                                                                test_correct, test_total, test_correct_repair,
                                                                                                                                test_total_repair, test_correct_2, test_total_2,
                                                                                                                                test_correct_repair_2, test_total_repair_2) + '\n')
            f.write("--------------------------------------------------------------------------------------------------------" + '\n')

        save_results_xlsx(config.save_path + '/valid_results_' + str(e + 1) + '.xlsx', valid_predict_labels)
        save_results_xlsx(config.save_path + '/test_results_' + str(e + 1) + '.xlsx', test_predict_labels)
        save_results_xlsx(config.save_path + '/test_2_results_' + str(e + 1) + '.xlsx', test_predict_labels_2)


def train_eval_model_2(config, model, data_train_list, data_valid_list, data_test_list):
    net = model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device)

    if isinstance(model, LSTMClassifier):
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        criterion = nn.CrossEntropyLoss()
        save_model = 'lstm'
    elif isinstance(model, TextCNNClassifier):
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        criterion = nn.CrossEntropyLoss()
        save_model = 'textcnn'

    net.train()

    valid_true_labels_list = []
    valid_predict_labels_list = []
    test_true_labels_list = []
    test_predict_labels_list = []

    for i in tqdm(range(len(data_train_list))):
        data_train = data_train_list[i]
        data_valid = data_valid_list[i]
        data_test = data_test_list[i]

        valid_true_labels = []
        valid_predict_labels = []
        test_true_labels = []
        test_predict_labels = []

        # train
        for e in range(config.epochs[i]):
            total_loss = []
            train_correct = 0
            train_total = 0
            for inputs, labels in data_train:
                if config.use_cuda:
                    inputs, labels = inputs.to(device), labels.to(device)

                net.zero_grad()
                output = net(inputs)

                _, preds = torch.max(output, dim=1)
                train_correct += torch.sum(preds == labels)
                train_total += len(labels)

                loss = criterion(output.squeeze(), labels.long())
                total_loss.append(loss.item())
                loss.backward()
                optimizer.step()

            print("Epoch: {}/{}, ".format(e + 1, config.epochs[i]),
                  "Train Loss: {:.6f}, ".format(torch.tensor(total_loss).mean()),
                  "Train repair acc: {}/{}.".format(train_correct, train_total))

        net.eval()

        # valid
        valid_correct = 0
        with torch.no_grad():
            for inputs, labels in data_valid:
                if config.use_cuda:
                    inputs, labels = inputs.to(device), labels.to(device)

                output = net(inputs)
                _, preds = torch.max(output, dim=1)
                valid_correct += torch.sum(preds == labels)
                valid_true_labels += labels.cpu().numpy().tolist()
                valid_predict_labels += preds.cpu().numpy().tolist()

        # test
        test_correct = 0
        with torch.no_grad():
            for inputs, labels in data_test:
                if config.use_cuda:
                    inputs, labels = inputs.to(device), labels.to(device)

                output = net(inputs)
                _, preds = torch.max(output, dim=1)
                test_correct += torch.sum(preds == labels)
                test_true_labels += labels.cpu().numpy().tolist()
                test_predict_labels += preds.cpu().numpy().tolist()

        net.train()

        valid_true_labels_list.append(valid_true_labels)
        valid_predict_labels_list.append(valid_predict_labels)
        test_true_labels_list.append(test_true_labels)
        test_predict_labels_list.append(test_predict_labels)

        print("--------------------------------------------------------------------------------------------------------")

    valid_true_labels_total = []
    valid_predict_labels_total = []
    test_true_labels_total = []
    test_predict_labels_total = []

    for i in range(len(valid_true_labels_list[0])):
        valid_true_labels_total.append([valid_true_labels_list[0][i], valid_true_labels_list[1][i], valid_true_labels_list[2][i], valid_true_labels_list[3][i],
                                        valid_true_labels_list[4][i], valid_true_labels_list[5][i]])
        valid_predict_labels_total.append([valid_predict_labels_list[0][i], valid_predict_labels_list[1][i], valid_predict_labels_list[2][i], valid_predict_labels_list[3][i],
                                           valid_predict_labels_list[4][i], valid_predict_labels_list[5][i]])

    for i in range(len(test_true_labels_list[0])):
        test_true_labels_total.append([test_true_labels_list[0][i], test_true_labels_list[1][i], test_true_labels_list[2][i], test_true_labels_list[3][i],
                                       test_true_labels_list[4][i], test_true_labels_list[5][i]])
        test_predict_labels_total.append([test_predict_labels_list[0][i], test_predict_labels_list[1][i], test_predict_labels_list[2][i], test_predict_labels_list[3][i],
                                          test_predict_labels_list[4][i], test_predict_labels_list[5][i]])
    for topN in range(1, 7):
        if topN == 6:
            valid_correct, valid_total = npr_test_results_sort(valid_true_labels_total, valid_predict_labels_total, topN)
            test_correct, test_total = npr_test_results_sort(test_true_labels_total, test_predict_labels_total, topN)
            print("Top{} - valid acc: {}/{}; test acc: {}/{}.".format(topN, valid_correct, valid_total, test_correct, test_total))
        else:
            valid_correct, valid_total = npr_test_results_sort(valid_true_labels_total, valid_predict_labels_total, topN)
            test_correct, test_total = npr_test_results_sort(test_true_labels_total, test_predict_labels_total, topN)
            test_correct_repair, test_total_repair = npr_test_results_sort_repaired(test_true_labels_total, test_predict_labels_total, topN)
            print("Top{} - valid acc: {}/{}; test acc: {}/{}, {}/{}".format(topN, valid_correct, valid_total, test_correct, test_total, test_correct_repair, test_total_repair))


def train_eval_model_dfg(config, model, data_train, data_valid, data_test, data_test_2):
    net = model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.BCEWithLogitsLoss()
    save_model = 'graphcodebert'

    net.train()

    for e in tqdm(range(config.epochs)):
        total_loss = []
        true_labels = []
        predict_labels = []

        valid_true_labels = []
        valid_predict_labels = []
        test_true_labels = []
        test_predict_labels = []
        test_true_labels_2 = []
        test_predict_labels_2 = []

        # train
        for code_ids, position_idx, attn_mask, labels in data_train:
            if config.use_cuda:
                code_ids, position_idx, attn_mask, labels = code_ids.to(device), position_idx.to(device), attn_mask.to(device), labels.to(device)

            net.zero_grad()
            output = net(code_ids, position_idx, attn_mask)

            true_labels += labels.cpu().numpy().tolist()
            predict_labels += torch.where(output > 0.5, 1, 0).cpu().numpy().tolist()

            loss = criterion(output.squeeze(), labels.float())
            total_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        train_correct, train_total = npr_train_results(true_labels, predict_labels)

        net.eval()

        # valid
        with torch.no_grad():
            for code_ids, position_idx, attn_mask, labels in data_valid:
                if config.use_cuda:
                    code_ids, position_idx, attn_mask, labels = code_ids.to(device), position_idx.to(device), attn_mask.to(device), labels.to(device)

                output = net(code_ids, position_idx, attn_mask)
                valid_true_labels += labels.cpu().numpy().tolist()
                valid_predict_labels += output.cpu().numpy().tolist()

        valid_correct_top1, valid_total = npr_test_results_sort(valid_true_labels, valid_predict_labels, 1)

        # test
        with torch.no_grad():
            for code_ids, position_idx, attn_mask, labels in data_test:
                if config.use_cuda:
                    code_ids, position_idx, attn_mask, labels = code_ids.to(device), position_idx.to(device), attn_mask.to(device), labels.to(device)

                output = net(code_ids, position_idx, attn_mask)
                test_true_labels += labels.cpu().numpy().tolist()
                test_predict_labels += output.cpu().numpy().tolist()
        with torch.no_grad():
            for code_ids, position_idx, attn_mask, labels in data_test_2:
                if config.use_cuda:
                    code_ids, position_idx, attn_mask, labels = code_ids.to(device), position_idx.to(device), attn_mask.to(device), labels.to(device)

                output = net(code_ids, position_idx, attn_mask)
                test_true_labels_2 += labels.cpu().numpy().tolist()
                test_predict_labels_2 += output.cpu().numpy().tolist()

        net.train()

        print("Epoch: {}/{}, ".format(e + 1, config.epochs),
              "Train Loss: {:.6f}, ".format(torch.tensor(total_loss).mean()),
              "Train repair acc: {}/{}.".format(train_correct, train_total))

        for topN in range(1, 7):
            if topN == 6:
                valid_correct, valid_total = npr_test_results_sort(valid_true_labels, valid_predict_labels, topN)
                test_correct, test_total = npr_test_results_sort(test_true_labels, test_predict_labels, topN)
                test_correct_2, test_total_2 = npr_test_results_sort(test_true_labels_2, test_predict_labels_2, topN)
                print("Top{} - valid acc: {}/{}; defects4j test acc: {}/{}; our test acc: {}/{}.".format(topN, valid_correct, valid_total, test_correct, test_total, test_correct_2,
                                                                                                         test_total_2))
            else:
                valid_correct, valid_total = npr_test_results_sort(valid_true_labels, valid_predict_labels, topN)
                valid_correct_repair, valid_total_repair = npr_test_results_sort_repaired(valid_true_labels, valid_predict_labels, topN)
                test_correct, test_total = npr_test_results_sort(test_true_labels, test_predict_labels, topN)
                test_correct_repair, test_total_repair = npr_test_results_sort_repaired(test_true_labels, test_predict_labels, topN)
                test_correct_2, test_total_2 = npr_test_results_sort(test_true_labels_2, test_predict_labels_2, topN)
                test_correct_repair_2, test_total_repair_2 = npr_test_results_sort_repaired(test_true_labels_2, test_predict_labels_2, topN)
                print(
                    "Top{} - valid acc: {}/{}, {}/{}; defects4j test acc: {}/{}, {}/{}; our test acc: {}/{}, {}/{}.".format(topN, valid_correct, valid_total, valid_correct_repair,
                                                                                                                            valid_total_repair, test_correct, test_total,
                                                                                                                            test_correct_repair, test_total_repair, test_correct_2,
                                                                                                                            test_total_2, test_correct_repair_2,
                                                                                                                            test_total_repair_2))
        print("--------------------------------------------------------------------------------------------------------")

        # record log
        with open(os.path.join(config.save_path, 'output_log.txt'), 'a', encoding='UTF-8') as f:
            f.write("Epoch: {}/{}, ".format(e + 1, config.epochs) +
                    "Train Loss: {:.6f}, ".format(torch.tensor(total_loss).mean()) +
                    "Train repair acc: {}/{}.".format(train_correct, train_total) + '\n')
            for topN in range(1, 7):
                if topN == 6:
                    valid_correct, valid_total = npr_test_results_sort(valid_true_labels, valid_predict_labels, topN)
                    test_correct, test_total = npr_test_results_sort(test_true_labels, test_predict_labels, topN)
                    test_correct_2, test_total_2 = npr_test_results_sort(test_true_labels_2, test_predict_labels_2, topN)
                    f.write("Top{} - valid acc: {}/{}; defects4j test acc: {}/{}; our test acc: {}/{}.".format(topN, valid_correct, valid_total, test_correct, test_total,
                                                                                                               test_correct_2, test_total_2) + '\n')
                else:
                    valid_correct, valid_total = npr_test_results_sort(valid_true_labels, valid_predict_labels, topN)
                    valid_correct_repair, valid_total_repair = npr_test_results_sort_repaired(valid_true_labels, valid_predict_labels, topN)
                    test_correct, test_total = npr_test_results_sort(test_true_labels, test_predict_labels, topN)
                    test_correct_repair, test_total_repair = npr_test_results_sort_repaired(test_true_labels, test_predict_labels, topN)
                    test_correct_2, test_total_2 = npr_test_results_sort(test_true_labels_2, test_predict_labels_2, topN)
                    test_correct_repair_2, test_total_repair_2 = npr_test_results_sort_repaired(test_true_labels_2, test_predict_labels_2, topN)
                    f.write(
                        "Top{} - valid acc: {}/{}, {}/{}; defects4j test acc: {}/{}, {}/{}; our test acc: {}/{}, {}/{}.".format(topN, valid_correct, valid_total,
                                                                                                                                valid_correct_repair, valid_total_repair,
                                                                                                                                test_correct, test_total, test_correct_repair,
                                                                                                                                test_total_repair, test_correct_2, test_total_2,
                                                                                                                                test_correct_repair_2, test_total_repair_2) + '\n')
            f.write("--------------------------------------------------------------------------------------------------------" + '\n')

        save_results_xlsx(config.save_path + '/valid_results_' + str(e + 1) + '.xlsx', valid_predict_labels)
        save_results_xlsx(config.save_path + '/test_results_' + str(e + 1) + '.xlsx', test_predict_labels)
        save_results_xlsx(config.save_path + '/test_2_results_' + str(e + 1) + '.xlsx', test_predict_labels_2)


def train_eval_model_seq(config, model, tokenizer, data_train, data_valid, data_test):
    net = model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    # criterion = nn.BCEWithLogitsLoss()
    save_model = 'codet5p'

    net.train()

    for e in tqdm(range(config.epochs)):
        total_loss = []

        valid_true_labels = []
        valid_probs_list = []
        valid_generated_ids_list = []
        test_true_labels = []
        test_probs_list = []
        test_generated_ids_list = []

        # train
        for inputs, labels in data_train:
            if config.use_cuda:
                inputs, labels = inputs.to(device), labels.to(device)

            net.zero_grad()
            loss = net(inputs, labels)  # e.g. tensor([11.6820, 10.8856], device='cuda:0', grad_fn=<GatherBackward>)

            total_loss.append(torch.mean(loss, dim=0).item())
            loss.backward(loss.clone().detach())
            optimizer.step()

        print("Epoch: {}/{}, ".format(e + 1, config.epochs),
              "Train Loss: {:.6f}.".format(torch.tensor(total_loss).mean()))

        if e == 0 or e % 10 != 0:
            continue

        net.eval()

        # test
        with torch.no_grad():
            for inputs, labels in data_test:
                if config.use_cuda:
                    inputs, labels = inputs.to(device), labels.to(device)

                output = net.module.generate(inputs)
                logits = output.scores
                probs = [torch.softmax(log, dim=-1) for log in logits]
                generated_ids = output.sequences.tolist()

                test_probs_list.append(probs)
                test_generated_ids_list.append(generated_ids)
                # predict = tokenizer.batch_decode(output, skip_special_tokens=True)
                # test_predict_labels.append(predict)
                label_decode = tokenizer.batch_decode(labels.cpu(), skip_special_tokens=True)
                test_true_labels.append(label_decode)

        valid_flag = False
        test_correct, test_total_repair = npr_results_sort_repaired_seq(test_true_labels, test_probs_list, test_generated_ids_list, 1)
        if test_correct >= 90:
            valid_flag = True
            # valid
            with torch.no_grad():
                for inputs, labels in data_valid:
                    if config.use_cuda:
                        inputs, labels = inputs.to(device), labels.to(device)

                    output = net.module.generate(inputs)
                    logits = output.scores
                    probs = [torch.softmax(log, dim=-1) for log in logits]
                    generated_ids = output.sequences.tolist()

                    valid_probs_list.append(probs)
                    valid_generated_ids_list.append(generated_ids)
                    # predict = tokenizer.batch_decode(output, skip_special_tokens=True)
                    # valid_predict_labels.append(predict)
                    label_decode = tokenizer.batch_decode(labels.cpu(), skip_special_tokens=True)
                    valid_true_labels.append(label_decode)

        net.train()

        for topN in range(1, 7):
            if topN == 6:
                test_correct, test_total, test_predict_labels = npr_results_sort_seq(test_true_labels, test_probs_list, test_generated_ids_list, topN)
                if valid_flag:
                    valid_correct, valid_total, valid_predict_labels = npr_results_sort_seq(valid_true_labels, valid_probs_list, valid_generated_ids_list, topN)
                    print("Top{} - valid acc: {}/{}; test acc: {}/{}.".format(topN, valid_correct, valid_total, test_correct, test_total))
                else:
                    print("Top{} - valid acc: {}/{}; test acc: {}/{}.".format(topN, '-', '-', test_correct, test_total))
            else:
                test_correct, test_total, test_predict_labels = npr_results_sort_seq(test_true_labels, test_probs_list, test_generated_ids_list, topN)
                test_correct_repair, test_total_repair = npr_results_sort_repaired_seq(test_true_labels, test_probs_list, test_generated_ids_list, topN)
                if valid_flag:
                    valid_correct, valid_total, valid_predict_labels = npr_results_sort_seq(valid_true_labels, valid_probs_list, valid_generated_ids_list, topN)
                    print("Top{} - valid acc: {}/{}; test acc: {}/{}, {}/{}".format(topN, valid_correct, valid_total, test_correct, test_total, test_correct_repair,
                                                                                    test_total_repair))
                else:
                    print("Top{} - valid acc: {}/{}; test acc: {}/{}, {}/{}".format(topN, '-', '-', test_correct, test_total, test_correct_repair, test_total_repair))
        print("--------------------------------------------------------------------------------------------------------")

        # record log
        with open(os.path.join(config.save_path, 'output_log.txt'), 'a', encoding='UTF-8') as f:
            f.write("Epoch: {}/{}, ".format(e + 1, config.epochs) +
                    "Train Loss: {:.6f}, ".format(torch.tensor(total_loss).mean()) + '\n')
            for topN in range(1, 7):
                if topN == 6:
                    test_correct, test_total, test_predict_labels = npr_results_sort_seq(test_true_labels, test_probs_list, test_generated_ids_list, topN)
                    if valid_flag:
                        valid_correct, valid_total, valid_predict_labels = npr_results_sort_seq(valid_true_labels, valid_probs_list, valid_generated_ids_list, topN)
                        f.write("Top{} - valid acc: {}/{}; test acc: {}/{}.".format(topN, valid_correct, valid_total, test_correct, test_total) + '\n')
                    else:
                        f.write("Top{} - valid acc: {}/{}; test acc: {}/{}.".format(topN, '-', '-', test_correct, test_total) + '\n')
                else:
                    test_correct, test_total, test_predict_labels = npr_results_sort_seq(test_true_labels, test_probs_list, test_generated_ids_list, topN)
                    test_correct_repair, test_total_repair = npr_results_sort_repaired_seq(test_true_labels, test_probs_list, test_generated_ids_list, topN)
                    if valid_flag:
                        valid_correct, valid_total, valid_predict_labels = npr_results_sort_seq(valid_true_labels, valid_probs_list, valid_generated_ids_list, topN)
                        f.write("Top{} - valid acc: {}/{}; test acc: {}/{}, {}/{}".format(topN, valid_correct, valid_total, test_correct, test_total, test_correct_repair,
                                                                                          test_total_repair) + '\n')
                    else:
                        f.write("Top{} - valid acc: {}/{}; test acc: {}/{}, {}/{}".format(topN, '-', '-', test_correct, test_total, test_correct_repair, test_total_repair) + '\n')
            f.write("--------------------------------------------------------------------------------------------------------" + '\n')

        print(test_predict_labels)
        if valid_flag:
            save_results_xlsx(config.save_path + '/valid_results_' + str(e + 1) + '.xlsx', valid_predict_labels)
        save_results_xlsx(config.save_path + '/test_results_' + str(e + 1) + '.xlsx', test_predict_labels)
