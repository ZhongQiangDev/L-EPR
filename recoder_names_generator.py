import os
import json
from tqdm import tqdm

meta_folder_path = 'Valid/metas'
meta_file_list = os.listdir(meta_folder_path)

for meta_file in tqdm(meta_file_list):
    # 从meta文件中获取类似于../bugs-QuixBugs/bugs/%s/names.json的数据，但去掉最外层的[]
    with open(os.path.join(meta_folder_path, meta_file), 'r', encoding='UTF-8') as f1:
        meta = f1.readline()
        # filename
        names = {'filename': meta.split('<sep>')[4].split('@')[0].split('\\')[-1]}
        # classes.methods
        methods = meta.split('<sep>')[4].split('@', maxsplit=1)[1].strip()
        # classes.methods.name
        methods_name = ""
        for m in methods.split(' '):
            if m.find('(') != -1:
                methods_name = m.split('(')[0]
                break
        # classes.methods.type
        # 夹逼准则
        methods_type = ''
        start = 0
        end = len(methods) - 1
        for m in methods.split(' '):
            if m in ['public', 'protected', 'private', 'static', 'final', 'abstract', 'synchronized']:
                start = methods.find(m) + len(m)
        end = methods.find(methods_name)
        methods_type = methods[start: end].strip()
        # classes.methods.params
        methods_params = []
        params_init = methods[methods.find('('):]
        param_init_list = list(params_init)
        # 清理掉多余干扰信息, 例如：params = '@RequestParam() int taskId, HttpServletResponse response'中的'@RequestParam()'
        if params_init.find('@') != -1:
            flag = False
            for i in range(len(param_init_list)):
                if param_init_list[i] == '@':
                    flag = True
                if param_init_list[i] == ' ':
                    flag = False
                if flag:
                    param_init_list[i] = ''
            params_init = ''.join(param_init_list).strip()
        params = params_init[: params_init.find(')')]
        if params == '(':
            methods_params.append({})
        else:
            params = params[1:]  # 去掉'('
            if params.find('<') == -1 or params.find('>') == -1:
                for p in params.split(','):
                    methods_params.append({'type': p.split(' ')[0].strip(), 'name': p.split(' ')[1].strip()})
            else:
                flag = 0
                split_index = []
                params_list = []
                for i in range(len(params)):
                    if params[i] == '<':
                        flag += 1
                        continue
                    if params[i] == '>':
                        flag -= 1
                        continue
                    if flag == 0:
                        if params[i] == ',':
                            split_index.append(i)
                        else:
                            continue
                if len(split_index) == 0:
                    params_list.append(params)
                else:
                    for i in range(len(split_index)):
                        if i == 0:
                            params_list.append(params[: split_index[i]].strip())
                        elif i == len(split_index) - 1:
                            params_list.append(params[split_index[i] + 1:].strip())
                        else:
                            params_list.append(params[split_index[i - 1] + 1: split_index[i]].strip())
                # print(meta_file + '-' + methods)
                # print(methods_type + '-' + methods_name)
                # print(params_list)
                for p in params_list:
                    methods_params.append({'type': p.split(' ')[0].strip(), 'name': p.split(' ')[1].strip()})

        classes = {'name': meta.split('<sep>')[4].split('@')[0].split('\\')[-1].split('.')[0],
                   'methods': [{'type': methods_type, 'name': methods_name, 'params': methods_params}],
                   'fields': []}
        names['classes'] = [classes]

        with open('Valid/names/' + meta_file.split('.')[0] + '.json', 'w', encoding='UTF-8') as f:
            json.dump(names, f, indent=4)
