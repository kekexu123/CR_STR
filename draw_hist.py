import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def load_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def process_data(json_data, head_list):
    confidence_scores = [conf for entry in json_data for conf in entry[1]]
    labels = [char for entry in json_data for char in list(entry[2])]
    head_data = [conf for label, conf in zip(labels, confidence_scores) if label in head_list ]
    tail_data = [conf for label, conf in zip(labels, confidence_scores) if label not in head_list ]
    return head_data, tail_data

def plot_hist(dfs, file_name):
    colors = ['green', 'orange']
    labels = ['Baseline', 'DCSS']

    plt.figure(figsize=(10, 5))

    # Plotting all the histograms with specified colors
    for df, color, label in zip(dfs, colors, labels):
        plt.hist(df, bins=100, alpha=0.5, edgecolor='black', color=color, label=label)

    # Adding titles and labels
    # plt.title('Confidence Score Distribution for Head and Tail Classes')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.grid(axis='y', alpha=0.75)

    # Adding legend to the plot in the top right corner
    plt.legend(title='Method' )   # loc='upper right'
    plt.savefig(file_name)
    plt.close()

def create_dataframe(data, class_name, method):
    return pd.DataFrame({
        'Confidence': data,
        'Class': [class_name] * len(data),
        'Method': [method] * len(data),
    })



head_list = ['e', 'a', 'o', 's', 't', 'r', 'i', 'n', 'l', 'c']
json_data_ce = load_data('exp_json/baseline_sta.json')
json_data_ls = load_data('exp_json/test_sta.json')

head_data_ce, tail_data_ce = process_data(json_data_ce, head_list)
head_data_ls, tail_data_ls = process_data(json_data_ls, head_list)


plot_hist([head_data_ce, head_data_ls], 'head_hist.png')
plot_hist([tail_data_ce, tail_data_ls], 'tail_hist.png')