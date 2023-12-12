import os
import re
from tqdm import tqdm

two_operators = ['++', '--', '<=', '>=', '==', '!=', '&&', '||', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=']
two_operators_special = ['<<', '>>']
three_operators = ['>>>', '>>=']


def find_special_characters(text):
    pattern = r'[^\w\s]'  # 定义特殊字符的模式为非单词、非空白字符
    matches = []
    new_matches = []
    new_matches_2 = []
    new_matches_3 = []

    for match in re.finditer(pattern, text):
        start = match.start()
        matches.append(start)

    # 去掉两个引号之间匹配到的特殊字符
    flag = True
    for m in matches:
        if flag:
            if text[m] == '"' or text[m] == "'":
                flag = False
            else:
                new_matches.append(m)
        else:
            if text[m] == '"' or text[m] == "'":
                flag = True

    # 去掉组成三字符运算符的特殊字符
    new_matches_2 = new_matches
    if len(new_matches) >= 3:
        for i in range(len(new_matches)):
            if i < len(new_matches) - 2:
                # i, i+1, i+2 is together
                if new_matches[i + 2] - new_matches[i] == 2:
                    operator = str(text[new_matches[i]]) + str(text[new_matches[i + 1]]) + str([new_matches[i + 2]])
                    # is '>>>' or '>>='
                    if operator in three_operators:
                        if new_matches[i] == 0:
                            continue
                        else:
                            if text[new_matches[i] - 1] == ' ':
                                del new_matches_2[i: i + 3]

    # 去掉组成双字符运算符的特殊字符
    new_matches_3 = new_matches_2
    if len(new_matches_2) >= 2:
        for i in range(len(new_matches_2)):
            if i < len(new_matches_2) - 1:
                # i, i+1 is together
                if new_matches_2[i + 1] - new_matches_2[i] == 1:
                    operator = str(text[new_matches_2[i]]) + str(text[new_matches_2[i + 1]])
                    if operator in two_operators:
                        del new_matches_3[i: i + 2]
                    elif operator == two_operators_special[0]:
                        if new_matches_2[i] == len(text) - 2:
                            continue
                        else:
                            if text[new_matches_2[i + 1] + 1] == ' ':
                                del new_matches_3[i: i + 2]
                    elif operator == two_operators_special[1]:
                        if new_matches_2[i] == 0:
                            continue
                        else:
                            if text[new_matches_2[i] - 1] == ' ':
                                del new_matches_3[i: i + 2]
                    elif operator == '->':
                        if new_matches_2[i] == 0:
                            if text[new_matches_2[i + 1] + 1] == ' ':
                                del new_matches_3[i: i + 2]
                        elif new_matches_2[i] == len(text) - 2:
                            if text[new_matches_2[i] - 1] == ' ':
                                del new_matches_3[i: i + 2]
                        else:
                            if text[new_matches_2[i] - 1] == ' ' and text[new_matches_2[i + 1] + 1] == ' ':
                                del new_matches_3[i: i + 2]

    return new_matches_3


buggy_folder_path = 'Valid/buggy_methods'
fix_folder_path = 'Valid/fix_methods'
meta_folder_path = 'Valid/metas'
buggy_file_list = os.listdir(buggy_folder_path)
fix_file_list = os.listdir(fix_folder_path)
meta_file_list = os.listdir(meta_folder_path)

src_data = []
tgt_data = []

for buggy_file in tqdm(buggy_file_list):
    buggy_code = ""
    with open(os.path.join(buggy_folder_path, buggy_file), 'r', encoding='UTF-8') as f2:
        lines = []
        for line in f2.readlines():
            line = line.strip()
            # 去除注释和空白行
            if line.startswith('//') or line.startswith('/*') or line.startswith('*') or line.startswith('*/') or line.startswith('/**') or line == '':
                continue
            matches = find_special_characters(line)
            # 从后往前遍历
            for m in matches[::-1]:
                if m == len(line) - 1:
                    if line[m - 1] != ' ':
                        line = line[:m] + ' ' + line[m:]
                elif m == 0:
                    if line[m + 1] != ' ':
                        line = line[:m + 1] + ' ' + line[m + 1:]
                else:
                    if line[m + 1] != ' ':
                        line = line[:m + 1] + ' ' + line[m + 1:]
                    if line[m - 1] != ' ':
                        line = line[:m] + ' ' + line[m:]
            lines.append(line)
    for line in lines:
        buggy_code += line + ' '
    src_data.append(buggy_code.strip() + '\n')

with open("./SequenceR/valid_data.txt", 'w', encoding='UTF-8') as f:
    f.writelines(src_data)
print("write valid_data.txt successful!")

for fix_file, meta_file in tqdm(zip(fix_file_list, meta_file_list)):
    fix_code = ""
    fix_lines_start = 0
    fix_lines_end = -1

    with open(os.path.join(meta_folder_path, meta_file), 'r', encoding='UTF-8') as f1:
        meta = f1.readline()
        fix_lines = meta.split('<sep>')[3]
        fix_lines_start = int(fix_lines.replace('[', '').replace(']', '').split(':')[0])
        fix_lines_end = int(fix_lines.replace('[', '').replace(']', '').split(':')[1])

    with open(os.path.join(fix_folder_path, fix_file), 'r', encoding='UTF-8') as f3:
        index = 0
        lines = []
        for line in f3.readlines():
            if fix_lines_start <= index < fix_lines_end:
                line = line.strip()
                # 去除注释和空白行
                if line.startswith('//') or line.startswith('/*') or line.startswith('*') or line.startswith('*/') or line.startswith('/**') or line == '':
                    continue
                matches = find_special_characters(line)
                # 从后往前遍历
                for m in matches[::-1]:
                    if m == len(line) - 1:
                        if line[m - 1] != ' ':
                            line = line[:m] + ' ' + line[m:]
                    elif m == 0:
                        if line[m + 1] != ' ':
                            line = line[:m + 1] + ' ' + line[m + 1:]
                    else:
                        if line[m + 1] != ' ':
                            line = line[:m + 1] + ' ' + line[m + 1:]
                        if line[m - 1] != ' ':
                            line = line[:m] + ' ' + line[m:]
                lines.append(line)
            index += 1
        for line in lines:
            fix_code += line + ' '
        tgt_data.append(fix_code.strip() + '\n')

with open("./SequenceR/fix_data.txt", 'w', encoding='UTF-8') as f:
    f.writelines(src_data)
print("write valid_data_tgt.txt successful!")