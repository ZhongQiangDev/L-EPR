import csv
import os

buggy_folder_path = 'Valid/buggy_methods'
fix_folder_path = 'Valid/fix_methods'
meta_folder_path = 'Valid/metas'
buggy_file_list = os.listdir(buggy_folder_path)
fix_file_list = os.listdir(fix_folder_path)
meta_file_list = os.listdir(meta_folder_path)

csv_data = [['bugid', 'buggy', 'patch']]
bugid = 1

for buggy_file, fix_file, meta_file in zip(buggy_file_list, fix_file_list, meta_file_list):
    buggy_lines_start = 0
    buggy_lines_end = -1
    fix_lines_start = 0
    fix_lines_end = -1
    context = ""
    buggy = ""
    patch = ""
    # meta data
    with open(os.path.join(meta_folder_path, meta_file), 'r', encoding='UTF-8') as f1:
        meta = f1.readline()
        buggy_lines = meta.split('<sep>')[2]  # e.g. str([1:2])
        buggy_lines_start = int(buggy_lines.replace('[', '').replace(']', '').split(':')[0])
        buggy_lines_end = int(buggy_lines.replace('[', '').replace(']', '').split(':')[1])
        fix_lines = meta.split('<sep>')[3]  # e.g. str([1:3])
        fix_lines_start = int(fix_lines.replace('[', '').replace(']', '').split(':')[0])
        fix_lines_end = int(fix_lines.replace('[', '').replace(']', '').split(':')[1])
    # buggy data:
    with open(os.path.join(buggy_folder_path, buggy_file), 'r', encoding='UTF-8') as f2:
        index = 0
        for line in f2.readlines():
            context += line.replace('\n', '')
            if buggy_lines_start <= index < buggy_lines_end:
                buggy += line.replace('\n', '')
            index += 1
    # fix data
    with open(os.path.join(fix_folder_path, fix_file), 'r', encoding='UTF-8') as f3:
        index = 0
        for line in f3.readlines():
            if fix_lines_start <= index < fix_lines_end:
                patch += line.replace('\n', '')
            index += 1

    csv_data.append([bugid, "buggy:" + buggy + "    context:" + context, patch])
    bugid += 1

with open('./RewardRepair/valid_data.csv', 'w', encoding='UTF-8', newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(csv_data)
print("write csv successful!")
