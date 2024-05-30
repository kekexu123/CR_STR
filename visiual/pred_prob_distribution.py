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
    head_data = [conf for label, conf in zip(labels, confidence_scores) if label in head_list]
    tail_data = [conf for label, conf in zip(labels, confidence_scores) if label not in head_list]
    return head_data, tail_data

def create_dataframe(data, class_name, method):
    return pd.DataFrame({
        'Predicted Probability': data,
        'Class': [class_name] * len(data),
        'Method': [method] * len(data),
    })

def plot_violin(df, class_name, file_name):
    plt.figure(figsize=(15,6))
    sns.violinplot(y='Class', x='Predicted Probability', hue='Method', data=df, 
                   split=True, inner='quartile', palette=['skyblue', 'lightgrey'])
    plt.title(f'Violin Plot of Predicted Probability Distributions for {class_name} Class')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.grid(True, which='major', linestyle='--', linewidth=0.5)
    plt.legend(title='Method', loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig(file_name)
    plt.close()

def plot_adjusted_violin(df, class_name, file_name, focus_threshold=0.05, dpi=300, bw=0.3):
    plt.figure(figsize=(15, 6), dpi=dpi)
    ax = sns.violinplot(y='Class', x='Predicted Probability', hue='Method', data=df, 
                        split=True, inner='quartile', palette=['skyblue', 'lightgrey'], bw=bw)
    
    # Adjust the x-axis limit to focus on areas with more information
    ax.set_xlim(left=df['Predicted Probability'].quantile(focus_threshold))
    
    # Improve the legend location and visibility
    plt.legend(title='Method', loc='lower center', bbox_to_anchor=(0.2, 0.8), ncol=2, frameon=True)

    plt.title(f'Adjusted Violin Plot of Predicted Probability Distributions for {class_name} Class')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.grid(True, which='major', linestyle='--', linewidth=0.5)

    # Save the plot with the specified DPI for better clarity
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()

head_list = ['e', 'a', 'o', 's', 't', 'r', 'i', 'n', 'l', 'c']
json_data_ce = load_data('exp_json/trba_ce_part_fullchar.json')
json_data_ls = load_data('exp_json/trba_las_part_fullchar.json')

head_data_ce, tail_data_ce = process_data(json_data_ce, head_list)
head_data_ls, tail_data_ls = process_data(json_data_ls, head_list)

df_head_combined = pd.concat([
create_dataframe(head_data_ce, 'Head', 'Cross-Entropy'),
create_dataframe(head_data_ls, 'Head', 'Label-Aware Smoothing')
])

df_tail_combined = pd.concat([
create_dataframe(tail_data_ce, 'Tail', 'Cross-Entropy'),
create_dataframe(tail_data_ls, 'Tail', 'Label-Aware Smoothing')
])

# plot_violin(df_head_combined, 'Head', 'head_distribution.jpg')
# plot_violin(df_tail_combined, 'Tail', 'tail_distribution.jpg')

plot_adjusted_violin(df_head_combined, 'Head', 'head_distribution_adjusted.jpg')
plot_adjusted_violin(df_tail_combined, 'Tail', 'tail_distribution_adjusted.jpg')
