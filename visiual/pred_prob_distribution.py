import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.collections as collections  # 正确的导入路径

plt.rcParams["font.family"] = "serif"
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
    plt.figure(figsize=(8,10))
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
    plt.figure(figsize=(10, 16), dpi=dpi)
    ax = sns.violinplot(y='Class', x='Predicted Probability', hue='Method', data=df, 
                        split=False, dodge=False, inner='quartile', palette=["#91d1c2cc", "#f39b7fcc"], bw=bw, cut=0)   # dodge=False   palette=["#3498db", "#e67e22"]
    for art in ax.findobj():
        if isinstance(art, collections.PolyCollection):  # 正确的检查方式
            art.set_alpha(0.5)  # 设置透明度为0.5
    
    # Adjust the x-axis limit to focus on areas with more information
    ax.set_xlim(left=df['Predicted Probability'].quantile(focus_threshold))
    
    # Improve the legend location and visibility
    plt.legend(title='Method', loc='lower center', bbox_to_anchor=(0.19, 0.9), ncol=2, frameon=True)

    plt.title(f'Predicted Probability Distributions for {class_name} Class')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.grid(True, which='major', linestyle='--', linewidth=0.5)

    # Save the plot with the specified DPI for better clarity
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()

def plot_adjusted_violin_v2(df, class_name, file_name, focus_threshold=0.05, dpi=300, bw=0.3, legend_fontsize=20):
    plt.figure(figsize=(10, 16), dpi=dpi)
    # Plot 'Baseline' without transparency
    sns.violinplot(y='Class', x='Predicted Probability', hue='Method', data=df[df['Method'] == 'Baseline'], 
                   split=False, inner='quartile', palette=['skyblue'], bw=bw, cut=0)
    # Plot 'DSCS' with transparency
    ax = sns.violinplot(y='Class', x='Predicted Probability', hue='Method', data=df[df['Method'] == 'DSCS'], 
                        split=False, inner='quartile', palette=["#f39b7fcc"], bw=bw, cut=0)
    for art in ax.findobj():
        if isinstance(art, collections.PolyCollection):
            art.set_alpha(0.5)  # Apply transparency to 'DSCS' only
    
    # Adjust the x-axis limit to focus on areas with more information
    ax.set_xlim(left=df['Predicted Probability'].quantile(focus_threshold))
    
    # Set the title and labels with specific font sizes
    # plt.title(f'Predicted Probability Distributions for {class_name} Class', fontsize=20)
    plt.xlabel('Predicted Probability', fontsize=20)
    plt.ylabel('Density', fontsize=20)
    
    # Improve the legend location and visibility with font size adjustments
    plt.legend(title='Method', loc='lower center', bbox_to_anchor=(0.3, 0.905), ncol=2, frameon=True, 
               title_fontsize=legend_fontsize, fontsize=legend_fontsize)
    
    plt.grid(True, which='major', linestyle='--', linewidth=0.5)

    # Save the plot with the specified DPI for better clarity
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()

head_list = ['e', 'a', 'o', 's', 't', 'r', 'i', 'n', 'l', 'c']
json_data_ce = load_data('exp_json/baseline_sta.json')
json_data_ls = load_data('exp_json/test_sta.json')

head_data_ce, tail_data_ce = process_data(json_data_ce, head_list)
head_data_ls, tail_data_ls = process_data(json_data_ls, head_list)

df_head_combined = pd.concat([
create_dataframe(head_data_ce, 'Head', 'Baseline'),
create_dataframe(head_data_ls, 'Head', 'DSCS')
])

df_tail_combined = pd.concat([
create_dataframe(tail_data_ce, 'Tail', 'Baseline'),
create_dataframe(tail_data_ls, 'Tail', 'DCSS')
])

# plot_violin(df_head_combined, 'Head', 'head_distribution.jpg')
# plot_violin(df_tail_combined, 'Tail', 'tail_distribution.jpg')

plot_adjusted_violin(df_head_combined, 'Head', 'head_distribution_v3.jpg')
plot_adjusted_violin_v2(df_head_combined, 'Head', 'head_distribution_v4.png')
# plot_adjusted_violin(df_tail_combined, 'Tail', 'tail_distribution_adjusted.png')
