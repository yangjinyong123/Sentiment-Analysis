# ======================== 情感分析模型（含ROC/AUC、批量预测验证、图表规范化完整版+class·weight） ========================
import pandas as pd
import jieba
import re
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_curve, auc, f1_score)

# 屏蔽无关警告
warnings.filterwarnings("ignore")

# ===================== 1. 路径与全局配置 =====================
SAVE_DIR = r"D:\运行结果"
os.makedirs(SAVE_DIR, exist_ok=True)

# 统一图表命名规范（对应论文章节编号，方便直接引用）
FIG_PREFIX = "fig5_"

# ===================== 2. 数据读取与预处理 =====================
print("正在读取 OCEMOTION.csv 数据集...")
df = pd.read_csv(
    r"D:\情感分析\OCEMOTION.csv",
    encoding="utf-8", sep="\t", on_bad_lines="skip", engine="python"
)
df = df.iloc[:, :3]
df.columns = ["id", "text", "label"]
df = df.dropna(subset=["text"])
print(f"数据集有效样本总量：{len(df)} 条")


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\u4e00-\u9fa5]", "", text)
    return " ".join(jieba.lcut(text))


df["clean_text"] = df["text"].apply(clean_text)


def map_label(lb):
    positive_labels = ["happiness", "love", "surprise"]  # 注意：请确认你的数据集中是love还是like
    return 1 if lb in positive_labels else 0


df["y"] = df["label"].apply(map_label)

# ===================== 3. 数据集三层划分 (8:1:1) =====================
X_train_all, X_temp, y_train_all, y_temp = train_test_split(
    df["clean_text"], df["y"], test_size=0.2, random_state=42, stratify=df["y"]
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)
X_train, y_train = X_train_all, y_train_all

print(f"训练集: {len(X_train)} | 验证集: {len(X_val)} | 测试集: {len(X_test)}")

# ===================== 4. TF-IDF特征提取 =====================
tfidf = TfidfVectorizer(max_features=10000)  # 建议限制特征维度以复现论文设定
X_train_vec = tfidf.fit_transform(X_train)
X_val_vec = tfidf.transform(X_val)
X_test_vec = tfidf.transform(X_test)

# ===================== 5. 双模型训练（基础 vs 类别平衡）=====================
print("\n正在训练双模型用于ROC对比...")
# 基础模型（无类别平衡）
model_base = LogisticRegression(max_iter=1000)
model_base.fit(X_train_vec, y_train)

# 优化模型（类别平衡）
model_balanced = LogisticRegression(max_iter=1000, class_weight='balanced')
model_balanced.fit(X_train_vec, y_train)

# 获取测试集预测概率（用于ROC）
prob_base = model_base.predict_proba(X_test_vec)[:, 1]
prob_balanced = model_balanced.predict_proba(X_test_vec)[:, 1]

# ===================== 6. 【核心升级一】ROC曲线与AUC分析 =====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fpr_base, tpr_base, _ = roc_curve(y_test, prob_base)
auc_base = auc(fpr_base, tpr_base)

fpr_bal, tpr_bal, _ = roc_curve(y_test, prob_balanced)
auc_bal = auc(fpr_bal, tpr_bal)

plt.figure(figsize=(8, 6))
plt.plot(fpr_base, tpr_base, 'b--', linewidth=2, label=f'基础模型 (AUC = {auc_base:.3f})')
plt.plot(fpr_bal, tpr_bal, 'r-', linewidth=2, label=f'类别平衡模型 (AUC = {auc_bal:.3f})')
plt.plot([0, 1], [0, 1], 'k:', alpha=0.5, label='随机基线')
plt.xlabel('假正率 (False Positive Rate)', fontsize=12)
plt.ylabel('真正率 (True Positive Rate)', fontsize=12)
plt.title('ROC曲线对比：基础模型 vs 类别平衡模型', fontsize=14)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(SAVE_DIR, f"{FIG_PREFIX}roc_comparison.png"), dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ ROC曲线已保存，基础AUC={auc_base:.3f}, 优化AUC={auc_bal:.3f}")

# ===================== 7. 优化模型三集评估与可视化 =====================
train_acc = accuracy_score(y_train, model_balanced.predict(X_train_vec))
val_acc = accuracy_score(y_val, model_balanced.predict(X_val_vec))
test_acc = accuracy_score(y_test, model_balanced.predict(X_test_vec))
y_pred = model_balanced.predict(X_test_vec)
test_report = classification_report(y_test, y_pred, target_names=["负面", "正面"])

# 保存带图表引用提示的评估报告
report_path = os.path.join(SAVE_DIR, "三集模型评估结果.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("========== 情感分析模型三数据集评估结果 ==========\n")
    f.write(f"训练集准确率：{train_acc:.2%}\n")
    f.write(f"验证集准确率：{val_acc:.2%}\n")
    f.write(f"测试集准确率：{test_acc:.2%}\n\n")
    f.write("========== 测试集详细分类报告 ==========\n")
    f.write(test_report + "\n\n")
    f.write("========== 论文图表引用提示（自动生成）==========\n")
    f.write(f"- 如图5-1所示（{FIG_PREFIX}sample_distribution.png）：数据集正负情感样本分布...\n")
    f.write(f"- 如图5-2所示（{FIG_PREFIX}acc_comparison.png）：三集准确率对比...\n")
    f.write(f"- 如图5-3所示（{FIG_PREFIX}confusion_matrix.png）：测试集混淆矩阵...\n")
    f.write(f"- 如图5-4所示（{FIG_PREFIX}roc_comparison.png）：ROC曲线对比，AUC从{auc_base:.3f}提升至{auc_bal:.3f}...\n")

# 绘制其余标准化图表
# 样本分布图
plt.figure(figsize=(6, 5))
df["y"].value_counts().plot(kind="bar", color=["salmon", "skyblue"])
plt.title("数据集正负情感样本分布")
plt.xticks([0, 1], ["负面", "正面"], rotation=0)
plt.savefig(os.path.join(SAVE_DIR, f"{FIG_PREFIX}sample_distribution.png"), dpi=300)
plt.close()

# 三集准确率对比
plt.figure(figsize=(7, 5))
sets_name = ["训练集", "验证集", "测试集"]
acc_list = [train_acc, val_acc, test_acc]
plt.bar(sets_name, acc_list, color=["#409EFF", "#67C23A", "#E6A23C"])
plt.title("训练集/验证集/测试集 准确率对比")
plt.ylim(0, 1)
plt.ylabel("准确率")
for i, acc in enumerate(acc_list):
    plt.text(i, acc + 0.01, f"{acc:.2%}", ha="center")
plt.savefig(os.path.join(SAVE_DIR, f"{FIG_PREFIX}acc_comparison.png"), dpi=300)
plt.close()

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["负面", "正面"], yticklabels=["负面", "正面"])
plt.title("测试集混淆矩阵")
plt.xlabel("预测标签")
plt.ylabel("真实标签")
plt.savefig(os.path.join(SAVE_DIR, f"{FIG_PREFIX}confusion_matrix.png"), dpi=300)
plt.close()

# ===================== 8. 【核心升级二】实时预测 + 批量统计验证 =====================
print("\n==================== 实时情感预测与批量验证系统 ====================")
record_path = os.path.join(SAVE_DIR, "实时预测记录.txt")

# --- 批量验证模式（使用测试集前100条作为模拟实时输入）---
BATCH_SIZE = 100
batch_texts = X_test.iloc[:BATCH_SIZE].tolist()
batch_true_labels = y_test.iloc[:BATCH_SIZE].tolist()

batch_preds = []
batch_records = []
print(f"正在执行批量实时预测验证（样本数：{BATCH_SIZE}）...")

for text, true_lbl in zip(batch_texts, batch_true_labels):
    sent_vec = tfidf.transform([text])
    prob = model_balanced.predict_proba(sent_vec)[0]
    pred_lbl = 1 if prob[1] > prob[0] else 0
    batch_preds.append(pred_lbl)

    tag = "正面" if pred_lbl == 1 else "负面"
    record_line = f"输入：{text[:30]}... | 预测：{tag}({prob[pred_lbl]:.2%}) | 真实：{'正面' if true_lbl == 1 else '负面'}"
    batch_records.append(record_line)

# 计算批量预测指标
batch_acc = accuracy_score(batch_true_labels, batch_preds)
batch_f1 = f1_score(batch_true_labels, batch_preds, average='weighted')

# 保存批量验证结果与交互式记录
with open(record_path, "w", encoding="utf-8") as f:
    f.write("========== 批量实时预测性能验证 ==========\n")
    f.write(f"验证样本数：{BATCH_SIZE}\n")
    f.write(f"批量预测准确率：{batch_acc:.2%}\n")
    f.write(f"批量预测加权F1：{batch_f1:.4f}\n")
    f.write(f"离线测试集准确率：{test_acc:.2%}\n")
    f.write(f"差异说明：批量验证仅抽取了{BATCH_SIZE}条样本，受抽样波动影响，指标与全量测试集存在合理偏差。\n\n")
    f.write("========== 逐条预测明细 ==========\n")
    for line in batch_records:
        f.write(line + "\n")

print(f"✅ 批量验证完成！准确率: {batch_acc:.2%}, F1: {batch_f1:.4f}")
print(f"✅ 结果已保存至: {record_path}")

# --- 保留原有的交互式预测入口 ---
print("\n进入手动交互模式（输入q退出）...")
while True:
    sentence = input("\n请输入待分析文本：")
    if sentence.lower() == "q":
        break
    clean_sent = clean_text(sentence)
    sent_vec = tfidf.transform([clean_sent])
    prob = model_balanced.predict_proba(sent_vec)[0]
    tag, p = ("正面", prob[1]) if prob[1] > prob[0] else ("负面", prob[0])
    result_str = f"→ {tag}情感，概率：{p:.2%}"
    print(result_str)
    with open(record_path, "a", encoding="utf-8") as f:
        f.write(f"[手动输入] {sentence} | {result_str}\n")

print("\n🎉 全部流程执行完毕！请前往 D:\\运行结果 查看论文所需素材。")