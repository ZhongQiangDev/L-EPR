import json
import os
import openpyxl
from tqdm import tqdm

tool_config_dir_path = './P-EPR/tool_configs_initialized'

with open('P-EPR/single_function_repair.json', 'r', encoding='UTF-8') as f:
    meta_data = json.load(f)
    for key in tqdm(meta_data.keys()):
        input_file_folder = 'P-EPR/D4j_files/' + key
        input_file_list = os.listdir(input_file_folder)
        for input_file in input_file_list:
            input_file_path = os.path.join(input_file_folder, input_file)
            faulty_line_ids = str(meta_data[key]['start']) + '-' + str(meta_data[key]['end'])
            result_file_path = './P-EPR/results-defects4j/result_'+key+'_'+input_file.split('.')[0]+'.json'
            print('java -jar ./ToolRanker.jar ' +
                      '-mode inference ' +
                      '-tool_config_dir ' + tool_config_dir_path + ' ' +
                      '-result_file ' + result_file_path + ' ' +
                      '-input_file ' + input_file_path + ' ' +
                      '-faulty_line_ids ' + faulty_line_ids + ' ' +
                      '-test_err_type junit.framework.AssertionFailedError')
            os.system('java -jar ./ToolRanker.jar ' +
                      '-mode inference ' +
                      '-tool_config_dir ' + tool_config_dir_path + ' ' +
                      '-result_file ' + result_file_path + ' ' +
                      '-input_file ' + input_file_path + ' ' +
                      '-faulty_line_ids ' + faulty_line_ids + ' ' +
                      '-test_err_type junit.framework.AssertionFailedError')

workbook = openpyxl.load_workbook('P-EPR/test5000_meta.xlsx')
worksheet = workbook.active
index = 0
for row in tqdm(worksheet.iter_rows()):
    if index == 0:
        index += 1
        continue
    row_data = [cell.value for cell in row]
    bugid = row_data[0]

    input_file_folder = 'P-EPR/javas/' + bugid
    input_file_list = os.listdir(input_file_folder)
    for input_file in input_file_list:
        input_file_path = os.path.join(input_file_folder, input_file)
        with open('P-EPR/metas/' + bugid + '.txt', 'r', encoding='UTF-8') as f:
            meta = f.readline()
            fix_lines = meta.split('<sep>')[3]  # e.g. str([1:2])
            fix_lines_start = int(fix_lines.replace('[', '').replace(']', '').split(':')[0]) + 1
            faulty_line_ids = str(fix_lines_start)
            result_file_path = './P-EPR/results-test5000/result_' + bugid + '.json'
            print('java -jar ./ToolRanker.jar ' +
                      '-mode inference ' +
                      '-tool_config_dir ' + tool_config_dir_path + ' ' +
                      '-result_file ' + result_file_path + ' ' +
                      '-input_file ' + input_file_path + ' ' +
                      '-faulty_line_ids ' + faulty_line_ids + ' ' +
                      '-test_err_type junit.framework.AssertionFailedError')
            os.system('java -jar ./ToolRanker.jar ' +
                      '-mode inference ' +
                      '-tool_config_dir ' + tool_config_dir_path + ' ' +
                      '-result_file ' + result_file_path + ' ' +
                      '-input_file ' + input_file_path + ' ' +
                      '-faulty_line_ids ' + faulty_line_ids + ' ' +
                      '-test_err_type junit.framework.AssertionFailedError')