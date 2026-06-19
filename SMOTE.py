# ======================== 情感分析模型（Borderline-SMOTE 优化完整版） ========================
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
# ⚠️ 新增：需提前 pip install imbalanced-learn
from imblearn.over_sampling import BorderlineSMOTE

warnings.filterwarnings("ignore")

# ===================== 1. 路径与全局配置 =====================
SAVE_DIR = r"D:\运行结果_SMOTE"  # ⚠️ 独立保存路径，防止覆盖class_weight结果
os.makedirs(SAVE_DIR, exist_ok=True)
FIG_PREFIX = "fig_smote_"

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
    positive_labels = ["happiness", "love", "surprise"]
    return 1 if lb in positive_labels else 0

df["y"] = df["label"].apply(map_label)

# ===================== 3. 数据集三层划分 (8:1:1) =====================
# ⚠️ 关键：必须先划分，再对训练集做过采样！
X_train_all, X_temp, y_train_all, y_temp = train_test_split(
    df["clean_text"], df["y"], test_size=0.2, random_state=42, stratify=df["y"]
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)
X_train, y_train = X_train_all, y_train_all

print(f"原始训练集: {len(X_train)} | 验证集: {len(X_val)} | 测试集: {len(X_test)}")

# ===================== 4. TF-IDF特征提取 =====================
tfidf = TfidfVectorizer(max_features=10000)
X_train_vec = tfidf.fit_transform(X_train)
X_val_vec = tfidf.transform(X_val)
X_test_vec = tfidf.transform(X_test)

# ===================== 5. 【核心升级】Borderline-SMOTE 过采样 =====================
print("\n正在执行 Borderline-SMOTE 过采样（仅针对训练集）...")
smote = BorderlineSMOTE(random_state=42)
# ⚠️ 仅对训练集特征和标签进行过采样，验证集/测试集绝对不参与
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_vec, y_train)

print(f"过采样后训练集样本数：{X_train_resampled.shape[0]}")
print(f"过采样后正负样本比例：{np.bincount(y_train_resampled)}")

# ===================== 6. 模型训练（⚠️ 移除class_weight）=====================
print("正在训练逻辑回归模型（默认权重 + SMOTE增强数据）...")
# ⚠️ 注意：使用了SMOTE后，必须使用默认权重(None)，不可再叠加class_weight
model_smote = LogisticRegression(max_iter=1000)
model_smote.fit(X_train_resampled, y_train_resampled)

# 获取测试集预测概率（用于ROC）
prob_smote = model_smote.predict_proba(X_test_vec)[:, 1]
y_pred = model_smote.predict(X_test_vec)

# ===================== 7. 三集评估与ROC/AUC计算 =====================
train_acc = accuracy_score(y_train_resampled, model_smote.predict(X_train_resampled))
val_acc = accuracy_score(y_val, model_smote.predict(X_val_vec))
test_acc = accuracy_score(y_test, y_pred)

test_report = classification_report(y_test, y_pred, target_names=["负面", "正面"])
fpr, tpr, _ = roc_curve(y_test, prob_smote)
auc_val = auc(fpr, tpr)
test_f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\n📊 SMOTE模型测试集准确率: {test_acc:.2%}")
print(f"📊 SMOTE模型加权 F1: {test_f1:.4f}")
print(f"📊 SMOTE模型 AUC 值: {auc_val:.3f}")
print("\n" + test_report)

# ===================== 8. 保存评估报告与可视化图表 =====================
report_path = os.path.join(SAVE_DIR, "SMOTE模型评估结果.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("========== Borderline-SMOTE 优化模型评估结果 ==========\n")
    f.write(f"过采样后训练集样本数：{X_train_resampled.shape[0]}\n")
    f.write(f"训练集准确率：{train_acc:.2%}\n")
    f.write(f"验证集准确率：{val_acc:.2%}\n")
    f.write(f"测试集准确率：{test_acc:.2%}\n")
    f.write(f"测试集加权F1：{test_f1:.4f}\n")
    f.write(f"AUC值：{auc_val:.3f}\n\n")
    f.write("========== 测试集详细分类报告 ==========\n")
    f.write(test_report + "\n\n")
    f.write("========== 论文图表引用提示 ==========\n")
    f.write(f"- 如图5-X所示（{FIG_PREFIX}roc.png）：SMOTE优化模型ROC曲线，AUC={auc_val:.3f}\n")
    f.write(f"- 如图5-X所示（{FIG_PREFIX}confusion_matrix.png）：SMOTE模型测试集混淆矩阵\n")

# 绘制SMOTE模型ROC曲线
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, 'r-', linewidth=2, label=f'Borderline-SMOTE 模型 (AUC = {auc_val:.3f})')
plt.plot([0,1], [0,1], 'k:', alpha=0.5, label='随机基线')
plt.xlabel('假正率 (False Positive Rate)', fontsize=12)
plt.ylabel('真正率 (True Positive Rate)', fontsize=12)
plt.title('Borderline-SMOTE 优化模型 ROC曲线', fontsize=14)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(SAVE_DIR, f"{FIG_PREFIX}roc.png"), dpi=300, bbox_inches='tight')
plt.close()

# 绘制SMOTE模型混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", xticklabels=["负面", "正面"], yticklabels=["负面", "正面"])
plt.title("Borderline-SMOTE 模型测试集混淆矩阵")
plt.xlabel("预测标签")
plt.ylabel("真实标签")
plt.savefig(os.path.join(SAVE_DIR, f"{FIG_PREFIX}confusion_matrix.png"), dpi=300)
plt.close()

# ===================== 9. 实时预测模块（复用原始TF-IDF与SMOTE模型）=====================
print("\n==================== 实时情感预测系统（输入q退出）====================")
record_path = os.path.join(SAVE_DIR, "实时预测记录.txt")
while True:
    sentence = input("\n请输入待分析文本：")
    if sentence.lower() == "q":
        break

    clean_sent = clean_text(sentence)
    sent_vec = tfidf.transform([clean_sent])

    prob = model_smote.predict_proba(sent_vec)[0]
    neg_p, pos_p = prob[0], prob[1]
    result_str = f"→ {'正面' if pos_p > neg_p else '负面'}情感，概率：{max(pos_p, neg_p):.2%}"
    print(result_str)

    with open(record_path, "a", encoding="utf-8") as f:
        f.write(f"输入：{sentence} | {result_str}\n")

print(f"\n🎉 所有结果已保存至: {SAVE_DIR}")
print("💡 请将此文件夹中的指标与 D:\\运行结果 (class_weight版) 。")