import openpyxl
import json
import os
from tqdm import tqdm

datasetinfo = {'recoder': [], 'tare': [], 'rewardrepair': [], 'selfapr': [], 'gamma': [], 'allfailure': []}
# datasetinfo = {'recoder': [], 'tare': [], 'rewardrepair': [], 'selfapr': [], 'gamma': []}

workbook = openpyxl.load_workbook('Meta/train1_meta.xlsx')
worksheet = workbook.active

index = 0
for row in tqdm(worksheet.iter_rows()):
    if index == 0:
        index += 1
        continue
    row_data = [cell.value for cell in row]
    bugid = row_data[0]

    with open('Meta/metas/' + bugid + '.txt', 'r', encoding='UTF-8') as f:
        meta = f.readline()
        fix_lines = meta.split('<sep>')[3]
        fix_lines_start = int(fix_lines.replace('[', '').replace(']', '').split(':')[0]) + 1

    java_folder = 'Meta/javas/' + bugid
    java_file = os.listdir(java_folder)[0]
    java_file_path = os.path.join('/home/P-EPR/P-EPR-Artefact-master/P-EPR/javas/' + bugid + '/' + java_file)

    if int(row_data[1]) == 1:
        recoder_info = {'file_path': java_file_path, 'fault_location': str(fix_lines_start), 'repair_result': 'Correct', 'test_error_type': 'junit.framework.AssertionFailedError'}
    else:
        recoder_info = {'file_path': java_file_path, 'fault_location': str(fix_lines_start), 'repair_result': 'Failed', 'test_error_type': 'junit.framework.AssertionFailedError'}
    datasetinfo['recoder'].append(recoder_info)

    if int(row_data[2]) == 1:
        tare_info = {'file_path': java_file_path, 'fault_location': str(fix_lines_start), 'repair_result': 'Correct', 'test_error_type': 'junit.framework.AssertionFailedError'}
    else:
        tare_info = {'file_path': java_file_path, 'fault_location': str(fix_lines_start), 'repair_result': 'Failed', 'test_error_type': 'junit.framework.AssertionFailedError'}
    datasetinfo['tare'].append(tare_info)

    if int(row_data[3]) == 1:
        rewardrepair_info = {'file_path': java_file_path, 'fault_location': str(fix_lines_start), 'repair_result': 'Correct',
                             'test_error_type': 'junit.framework.AssertionFailedError'}
    else:
        rewardrepair_info = {'file_path': java_file_path, 'fault_location': str(fix_lines_start), 'repair_result': 'Failed',
                             'test_error_type': 'junit.framework.AssertionFailedError'}
    datasetinfo['rewardrepair'].append(rewardrepair_info)

    if int(row_data[4]) == 1:
        selfapr_info = {'file_path': java_file_path, 'fault_location': str(fix_lines_start), 'repair_result': 'Correct', 'test_error_type': 'junit.framework.AssertionFailedError'}
    else:
        selfapr_info = {'file_path': java_file_path, 'fault_location': str(fix_lines_start), 'repair_result': 'Failed', 'test_error_type': 'junit.framework.AssertionFailedError'}
    datasetinfo['selfapr'].append(selfapr_info)

    if int(row_data[5]) == 1:
        gamma_info = {'file_path': java_file_path, 'fault_location': str(fix_lines_start), 'repair_result': 'Correct', 'test_error_type': 'junit.framework.AssertionFailedError'}
    else:
        gamma_info = {'file_path': java_file_path, 'fault_location': str(fix_lines_start), 'repair_result': 'Failed', 'test_error_type': 'junit.framework.AssertionFailedError'}
    datasetinfo['gamma'].append(gamma_info)

    if int(row_data[6]) == 1:
        allfailure_info = {'file_path': java_file_path, 'fault_location': str(fix_lines_start), 'repair_result': 'Correct', 'test_error_type': 'junit.framework.AssertionFailedError'}
    else:
        allfailure_info = {'file_path': java_file_path, 'fault_location': str(fix_lines_start), 'repair_result': 'Failed', 'test_error_type': 'junit.framework.AssertionFailedError'}
    datasetinfo['allfailure'].append(allfailure_info)

with open('P-EPR/DatasetInfo.json', 'w', encoding='UTF-8') as f1:
    json.dump(datasetinfo, f1)
