# ======================== 情感分析模型（组合优化策略：特征选择 + Borderline-SMOTE） ========================
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
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_curve, auc, f1_score)
# 引入 Borderline-SMOTE
from imblearn.over_sampling import BorderlineSMOTE

warnings.filterwarnings("ignore")

# ===================== 1. 路径与全局配置 =====================
SAVE_DIR = r"D:\运行结果_组合优化"
os.makedirs(SAVE_DIR, exist_ok=True)
FIG_PREFIX = "fig_combo_"

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
X_train_all, X_temp, y_train_all, y_temp = train_test_split(
    df["clean_text"], df["y"], test_size=0.2, random_state=42, stratify=df["y"]
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)
X_train, y_train = X_train_all, y_train_all

print(f"训练集: {len(X_train)} | 验证集: {len(X_val)} | 测试集: {len(X_test)}")

# ===================== 4. TF-IDF特征提取（保留10000维候选池）=====================
tfidf = TfidfVectorizer(max_features=10000)
X_train_vec = tfidf.fit_transform(X_train)
X_val_vec = tfidf.transform(X_val)
X_test_vec = tfidf.transform(X_test)
print(f"原始TF-IDF特征维度：{X_train_vec.shape}")

# ===================== 5. 【第一步：做减法】卡方+互信息联合特征选择 =====================
K_BEST = 3000  # 目标维度
print(f"\n[Step 1] 正在执行联合特征选择（目标维度：{K_BEST}）...")

# 仅在训练集上计算统计得分
chi_scores, _ = chi2(X_train_vec, y_train)
mi_scores = mutual_info_classif(X_train_vec, y_train, random_state=42)

# 归一化并联合打分
chi_norm = chi_scores / chi_scores.max()
mi_norm = mi_scores / mi_scores.max()
combined_scores = chi_norm + mi_norm

# 获取Top K特征索引
top_k_indices = np.argsort(combined_scores)[-K_BEST:]

# 对三集数据进行降维（仅保留选中的特征）
X_train_selected = X_train_vec[:, top_k_indices]
X_val_selected = X_val_vec[:, top_k_indices]
X_test_selected = X_test_vec[:, top_k_indices]
print(f"✅ 特征选择完成：10000维 -> {K_BEST}维")

# ===================== 6. 【第二步：做加法】Borderline-SMOTE 过采样 =====================
print(f"\n[Step 2] 正在执行 Borderline-SMOTE 过采样（仅针对降维后的训练集）...")
smote = BorderlineSMOTE(random_state=42)

# ⚠️ 核心防线：仅对训练集做过采样
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)

print(f"✅ SMOTE完成！训练集样本数：{X_train_selected.shape} -> {X_train_resampled.shape}")
print(f"   过采样后正负样本比例：{np.bincount(y_train_resampled)}")

# ===================== 7. 模型训练（在降维+增强后的数据上训练）=====================
print("\n[Step 3] 正在训练逻辑回归模型...")
# 注意：使用了SMOTE后，使用默认权重(None)即可，无需class_weight
model = LogisticRegression(max_iter=1000)
model.fit(X_train_resampled, y_train_resampled)

# ===================== 8. 三集评估与ROC/AUC计算 =====================
train_acc = accuracy_score(y_train_resampled, model.predict(X_train_resampled))
val_acc = accuracy_score(y_val, model.predict(X_val_selected))
test_acc = accuracy_score(y_test, model.predict(X_test_selected))

y_pred = model.predict(X_test_selected)
prob_test = model.predict_proba(X_test_selected)[:, 1]

test_report = classification_report(y_test, y_pred, target_names=["负面", "正面"])
fpr, tpr, _ = roc_curve(y_test, prob_test)
auc_val = auc(fpr, tpr)
test_f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\n📊 组合优化模型测试集准确率: {test_acc:.2%}")
print(f"📊 组合优化模型加权 F1: {test_f1:.4f}")
print(f"📊 组合优化模型 AUC 值: {auc_val:.3f}")
print("\n" + test_report)

# ===================== 9. 保存评估报告与可视化图表 =====================
report_path = os.path.join(SAVE_DIR, "组合优化模型评估结果.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("========== 组合优化策略（特征选择 + Borderline-SMOTE）评估结果 ==========\n")
    f.write(f"原始特征维度：10000 | 筛选后特征维度：{K_BEST}\n")
    f.write(f"SMOTE前训练集样本数：{X_train_selected.shape}\n")
    f.write(f"SMOTE后训练集样本数：{X_train_resampled.shape}\n\n")
    f.write(f"训练集准确率：{train_acc:.2%}\n")
    f.write(f"验证集准确率：{val_acc:.2%}\n")
    f.write(f"测试集准确率：{test_acc:.2%}\n")
    f.write(f"测试集加权F1：{test_f1:.4f}\n")
    f.write(f"AUC值：{auc_val:.3f}\n\n")
    f.write("========== 测试集详细分类报告 ==========\n")
    f.write(test_report + "\n\n")
    f.write("========== 论文图表引用提示 ==========\n")
    f.write(f"- 如图5-X所示（{FIG_PREFIX}roc.png）：组合优化模型ROC曲线，AUC={auc_val:.3f}\n")
    f.write(f"- 如图5-X所示（{FIG_PREFIX}confusion_matrix.png）：组合优化模型测试集混淆矩阵\n")

# 绘制ROC曲线
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, 'm-', linewidth=2, label=f'组合优化模型 (AUC = {auc_val:.3f})')
plt.plot([0, 1], [0, 1], 'k:', alpha=0.5, label='随机基线')
plt.xlabel('假正率 (False Positive Rate)', fontsize=12)
plt.ylabel('真正率 (True Positive Rate)', fontsize=12)
plt.title('组合优化策略（特征选择+SMOTE） ROC曲线', fontsize=14)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(SAVE_DIR, f"{FIG_PREFIX}roc.png"), dpi=300, bbox_inches='tight')
plt.close()

# 绘制混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=["负面", "正面"], yticklabels=["负面", "正面"])
plt.title("组合优化模型 测试集混淆矩阵")
plt.xlabel("预测标签")
plt.ylabel("真实标签")
plt.savefig(os.path.join(SAVE_DIR, f"{FIG_PREFIX}confusion_matrix.png"), dpi=300)
plt.close()

print(f"\n🎉 组合优化策略所有结果已保存至: {SAVE_DIR}")