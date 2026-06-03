# ======================== 情感分析模型（含训练集+验证集+测试集 三划分完整版） ========================
import pandas as pd
import jieba
import re
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# 引入样本平衡工具
from imblearn.over_sampling import SMOTE

# ===================== 保存路径：D盘 → 运行结果文件夹 =====================
SAVE_DIR = r"D:\运行结果\\"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===================== 读取OCEMOTION口语数据集 =====================
print("正在读取 OCEMOTION.csv 数据集...")
df = pd.read_csv(
    r"C:\Users\1\y3PuHapW\shuju\OCEMOTION.csv",
    encoding="utf-8",
    sep="\t",
    on_bad_lines="skip",
    engine="python"
)
# 筛选有效三列：序号、文本、情感标签
df = df.iloc[:, :3]
df.columns = ["id", "text", "label"]
# 去除空文本数据
df = df.dropna(subset=["text"])
print(f"数据集有效样本总量：{len(df)} 条")


# ===================== 文本预处理函数 =====================
def clean_text(text):
    text = str(text).lower()
    # 仅保留中文字符
    text = re.sub(r"[^\u4e00-\u9fa5]", "", text)
    # 结巴分词
    return " ".join(jieba.lcut(text))


df["clean_text"] = df["text"].apply(clean_text)

# 过滤清洗后为空的无效文本（小幅降噪）
df = df[df["clean_text"] != ""].copy()

# ===================== 情感标签二分类转换 =====================
# 正面：快乐、喜爱、惊喜  负面：悲伤、愤怒、恐惧等
def map_label(lb):
    positive_labels = ["happiness", "love", "surprise"]
    return 1 if lb in positive_labels else 0


df["y"] = df["label"].apply(map_label)

# ===================== 【核心升级：数据集三层划分】 =====================
# 第一步：先拆分 训练集(80%) + 剩余数据集(20%)
X_train_all, X_temp, y_train_all, y_temp = train_test_split(
    df["clean_text"], df["y"], test_size=0.2, random_state=42, stratify=df["y"]
)
# 第二步：剩余数据集对半拆分 验证集(10%) + 测试集(10%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# 训练集最终数据
X_train = X_train_all
y_train = y_train_all

print(f"训练集样本数：{len(X_train)} 条")
print(f"验证集样本数：{len(X_val)} 条")
print(f"测试集样本数：{len(X_test)} 条")

# ===================== TF-IDF特征提取（小幅增强特征表达） =====================
# 增加二元语法、过滤低频词，提升语义区分能力
tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=2)
# 仅用训练集拟合词典（防止数据泄露）
X_train_vec = tfidf.fit_transform(X_train)
# 验证集、测试集仅转换特征
X_val_vec = tfidf.transform(X_val)
X_test_vec = tfidf.transform(X_test)

# ===================== 核心：SMOTE 合成采样 平衡训练集正负样本 =====================
smote = SMOTE(random_state=42)
X_train_vec, y_train = smote.fit_resample(X_train_vec, y_train)

# ===================== 模型训练 + 验证集调优评估 =====================
# 调大迭代次数，保留类别权重，进一步平衡分类偏向
model = LogisticRegression(max_iter=3000, class_weight='balanced', random_state=42)
model.fit(X_train_vec, y_train)

# 分别获取三集准确率
train_acc = accuracy_score(y_train, model.predict(X_train_vec))
val_acc = accuracy_score(y_val, model.predict(X_val_vec))
test_acc = accuracy_score(y_test, model.predict(X_test_vec))

# 测试集最终分类报告
y_pred = model.predict(X_test_vec)
test_report = classification_report(y_test, y_pred, target_names=["负面", "正面"])

# ===================== 保存全套结果到D盘 =====================
with open(SAVE_DIR + "三集模型评估结果.txt", "w", encoding="utf-8") as f:
    f.write("========== 情感分析模型三数据集评估结果 ==========\n")
    f.write(f"数据集总样本量：{len(df)} 条\n")
    f.write(f"训练集样本数：{len(X_train)} 条\n")
    f.write(f"验证集样本数：{len(X_val)} 条\n")
    f.write(f"测试集样本数：{len(X_test)} 条\n\n")
    f.write(f"训练集准确率：{train_acc:.2%}\n")
    f.write(f"验证集准确率：{val_acc:.2%}\n")
    f.write(f"测试集准确率：{test_acc:.2%}\n\n")
    f.write("========== 测试集详细分类报告 ==========\n")
    f.write(test_report)

# ===================== 绘制专业可视化图表 =====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 三数据集准确率对比图
plt.figure(figsize=(7, 5))
sets_name = ["训练集", "验证集", "测试集"]
acc_list = [train_acc, val_acc, test_acc]
plt.bar(sets_name, acc_list, color=["#409EFF", "#67C23A", "#E6A23C"])
plt.title("训练集/验证集/测试集 准确率对比")
plt.ylim(0, 1)
plt.ylabel("准确率")
for i, acc in enumerate(acc_list):
    plt.text(i, acc + 0.01, f"{acc:.2%}", ha="center")
plt.savefig(SAVE_DIR + "三集准确率对比.png", dpi=300)
plt.close()

# 2. 测试集混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["负面", "正面"], yticklabels=["负面", "正面"])
plt.title("测试集混淆矩阵")
plt.xlabel("预测标签")
plt.ylabel("真实标签")
plt.savefig(SAVE_DIR + "测试集混淆矩阵.png", dpi=300)
plt.close()

# 3. 整体样本正负分布
plt.figure(figsize=(6, 5))
df["y"].value_counts().plot(kind="bar", color=["salmon", "skyblue"])
plt.title("数据集正负情感样本分布")
plt.xticks([0, 1], ["负面", "正面"], rotation=0)
plt.savefig(SAVE_DIR + "样本分布统计图.png", dpi=300)
plt.close()

print("\n✅ 所有结果已保存至D盘运行结果文件夹！")
print("✅ 包含：三集评估报告、准确率对比图、混淆矩阵、样本分布图")
print(f"✅ 最终模型泛化准确率（测试集）：{test_acc:.2%}")

# ===================== 实时情感预测模块（带百分比 + 自动保存记录） =====================
print("\n==================== 实时情感预测系统（输入q退出）====================")
while True:
    sentence = input("\n请输入待分析文本：")
    if sentence.lower() == "q":
        break

    clean_sent = clean_text(sentence)
    sent_vec = tfidf.transform([clean_sent])

    # 获取概率
    prob = model.predict_proba(sent_vec)[0]
    neg_p = prob[0]
    pos_p = prob[1]

    # 输出情感 + 百分比
    if pos_p > neg_p:
        result_str = f"→ 正面情感，概率：{pos_p:.2%}"
    else:
        result_str = f"→ 负面情感，概率：{neg_p:.2%}"

    print(result_str)

    # ===================== 【自动保存预测记录到D盘】 =====================
    with open(SAVE_DIR + "实时预测记录.txt", "a", encoding="utf-8") as f:
        f.write(f"输入：{sentence} | {result_str}\n")

print("\n🎉 所有预测记录已保存到 D盘运行结果 → 实时预测记录.txt")
print("🎉 程序运行完毕！")
