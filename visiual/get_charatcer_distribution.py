import lmdb
import re
import os
import json
from collections import Counter
import matplotlib.pyplot as plt

# main_directory = '/home/mdisk1/luoyu/datasets/data_CVPR2021/evaluation/benchmark'
main_directory = '/home/mdisk1/xukeke/Consistency_Regularization_STR/train_dataset/label'
dataset_directories = [os.path.join(main_directory, d) for d in os.listdir(main_directory) if os.path.isdir(os.path.join(main_directory, d))]
with open('charset_67.txt','r') as f:
    character = [line.strip('\n') for line in f.readlines()]
    character = ''.join(character)
total_char_counter = Counter()

# 初始化全局字符频次计数器
for char in character:
    total_char_counter[char] = 0

# LMDB路径和环境设置
for lmdb_path in dataset_directories:
    env = lmdb.open(lmdb_path, max_readers=32,
                                readonly=True,
                                lock=False,
                                readahead=False,
                                meminit=False)

    # 初始化字符频次计数器
    char_counter = Counter({char: 0 for char in character})

    with env.begin(write=False) as txn:
        nSamples = int(txn.get('num-samples'.encode()))
        for index in range(nSamples):
            index += 1  # lmdb starts with 1
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')
            label = label.lower()
            out_of_char = f'[^{character}]'
            label = re.sub(out_of_char, '', label)
            char_counter.update(label)

        sorted_chars, frequencies = zip(*char_counter.most_common())
        save_path = lmdb_path.split('/')[-1]
        plt.bar(sorted_chars, frequencies)
        plt.xlabel('Character')
        plt.ylabel('Frequency')
        plt.title(f'Character Frequency in {save_path} Dataset')
        
        # plt.savefig(f'frequency_collection/{save_path}_frequency.jpg')
        plt.savefig(f'debug/{save_path}_frequency.jpg')
        plt.close()
        
    total_char_counter.update(char_counter)

char_num_list = [total_char_counter[char] for char in total_char_counter.keys()]
with open('char_67_num.json','w') as f: 
    f.write(json.dumps(char_num_list, ensure_ascii=False))
# 排序字符频次并生成柱状图
sorted_chars, frequencies = zip(*total_char_counter.most_common())
plt.bar(sorted_chars, frequencies)
plt.xlabel('Character')
plt.ylabel('Frequency')
plt.title('Character Frequency in total Dataset')

plt.savefig('frequency_collection/train_character_distribution.jpg')
plt.close()
