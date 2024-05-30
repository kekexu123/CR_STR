import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"

# 数据
categories = ['Head', 'Tail']
baseline_values = [90.29, 85.1]
ours_values = [92.66, 87.3]

# 定义柱状图的位置
bar_width = 0.27
index = range(len(categories))
plt.figure(figsize=(4, 4))

plt.ylim(80, 100)
plt.yticks([80, 85, 90, 95, 100])
# 创建柱状图
plt.bar(index, baseline_values, bar_width, label='Baseline', color='#87CEEB', edgecolor='grey')
plt.bar([i + bar_width for i in index], ours_values, bar_width, label='Ours', color='#FBD6D6', edgecolor='grey')

for i in index:
    plt.text(i, baseline_values[i] + 1, str(baseline_values[i]), ha='center', va='bottom')
    plt.text(i + bar_width, ours_values[i] + 1, str(ours_values[i]), ha='center', va='bottom')
# 添加标签和标题
plt.ylabel('Accuracy', fontsize=15, labelpad=-6)
plt.gca().spines['bottom'].set_linewidth(1)  # 底部坐标轴线
plt.gca().spines['left'].set_linewidth(1)    # 左侧坐标轴线
plt.gca().spines['top'].set_linewidth(1)     # 顶部坐标轴线
plt.gca().spines['right'].set_linewidth(1)   # 右侧坐标轴线
# plt.yticks()
# plt.xticks([i + bar_width / 4 for i in index], categories)
plt.xticks([i + bar_width / 2 for i in index], categories, fontsize=12)
plt.legend(frameon=False, loc='upper right')
# 显示图形
# plt.grid(True)
plt.savefig('figure_acc.png', dpi=600)
plt.show()

# import matplotlib.pyplot as plt
# import numpy as np

# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.size"] = 12
# plt.rcParams["axes.labelsize"] = 15
# plt.rcParams["axes.titlesize"] = 15
# plt.rcParams["legend.fontsize"] = 12
# plt.rcParams["xtick.labelsize"] = 12
# plt.rcParams["ytick.labelsize"] = 12

# # 数据
# categories = ['Head', 'Tail']
# baseline_values = [90.29, 85.1]
# ours_values = [92.66, 87.3]

# # 定义柱状图的位置
# bar_width = 0.27
# index = np.array(range(len(categories)))
# plt.figure(figsize=(8, 6))

# plt.ylim(80, 100)
# plt.yticks([80, 85, 90, 95, 100])
# # 创建柱状图
# baseline_bars = plt.bar(index, baseline_values, bar_width, label='Baseline', color='#87CEEB', edgecolor='black')
# ours_bars = plt.bar(index + bar_width, ours_values, bar_width, label='Ours', color='#F08080', edgecolor='black')

# # 添加数据标签
# for bars in [baseline_bars, ours_bars]:
#     for bar in bars:
#         yval = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, round(yval, 2), ha='center', va='bottom')

# # 添加标签和标题
# plt.ylabel('Accuracy (%)')
# plt.gca().spines['bottom'].set_linewidth(1.2)  # 底部坐标轴线
# plt.gca().spines['left'].set_linewidth(1.2)    # 左侧坐标轴线
# plt.gca().spines['top'].set_visible(False)     # 隐藏顶部坐标轴线
# plt.gca().spines['right'].set_visible(False)   # 隐藏右侧坐标轴线

# # 设置x轴刻度位置和标签
# plt.xticks(index + bar_width / 2, categories)

# # 设置图例
# plt.legend(frameon=False, bbox_to_anchor=(1, 1), loc='upper left')

# # 添加轻微的网格线
# plt.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5, color='grey')
# plt.grid(False, which='major', axis='x')

# # 调整布局
# plt.tight_layout()

# # 保存和显示图形
# plt.savefig('figure_acc_enhanced.png', dpi=600)
# plt.show()