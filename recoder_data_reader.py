import pickle

# 读取train样本
# 从pkl文件中反序列化对象
with open('Recoder/data0.pkl', 'rb') as file:
    loaded_data = pickle.load(file)  # list

# 打印加载的对象
# print(loaded_data[0])  # dict_keys(['old', 'new', 'oldtree', 'newtree'])
# loaded_data_one = loaded_data[0]
# print(loaded_data_one['old'])
# print(loaded_data_one['new'])
# print(loaded_data_one['oldtree'])
# print(loaded_data_one['newtree'])

# 读取test样本
with open('Recoder/bugs-QuixBugs/bugs/GCD.java/src/java_programs/GCD.java', 'r') as java:
    print(java.read().strip().splitlines())