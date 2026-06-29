# ======================== 情感分析模型（全功能版：评估可视化 + R键重录 + AI纠错 + 多轮记忆） ========================
import pandas as pd
import jieba
import re
import os
import warnings
import cv2
import numpy as np
import threading
import wave
import tempfile
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc)

import dashscope
from dashscope import Generation
from aip import AipSpeech

warnings.filterwarnings("ignore")

# 设置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 【配置区】 =====================
dashscope.api_key = "sk-ws-H.RPMPDLX.Li92.MEQCIHNcQM4ASgFh15-1W5KEpkXxmNYhkhwI8pbwUP2YJh-qAiAFjm_OE0v1jyMeECgIBUUsPGeWMFD73O8HXQ3x0005-Q"

BAIDU_APP_ID = '123745318'
BAIDU_API_KEY = 'nsURm1FNv9iz02v7bkvuWPed'
BAIDU_SECRET_KEY = 'HtZCZIKgerbdImOQ5Zc1cQj741jYOgbx'
baidu_client = AipSpeech(BAIDU_APP_ID, BAIDU_API_KEY, BAIDU_SECRET_KEY)

SAVE_DIR = r"D:\运行结果"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===================== 1. 数据集处理与模型训练 =====================
print("=" * 50)
print("📊 正在读取并统计 OCEMOTION.csv 数据集...")
df = pd.read_csv(r"D:\情感分析项目1\OCEMOTION.csv", encoding="utf-8", sep="\t", on_bad_lines="skip", engine="python")
df = df.iloc[:, :3]
df.columns = ["id", "text", "label"]
df = df.dropna(subset=["text"])

total_count = len(df)
positive_labels = ["happiness", "love", "surprise"]
df["y"] = df["label"].apply(lambda lb: 1 if lb in positive_labels else 0)
pos_count = int(df["y"].sum())
neg_count = total_count - pos_count

print(f"\n【原始数据集统计】")
print(f"  • 有效总数据量: {total_count} 条")
print(f"  • 正面情感样本: {pos_count} 条 ({pos_count / total_count:.2%})")
print(f"  • 负面情感样本: {neg_count} 条 ({neg_count / total_count:.2%})")


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\u4e00-\u9fa5]", "", text)
    return " ".join(jieba.lcut(text))


df["clean_text"] = df["text"].apply(clean_text)

X_train_all, X_temp, y_train_all, y_temp = train_test_split(
    df["clean_text"], df["y"], test_size=0.2, random_state=42, stratify=df["y"]
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)
X_train, y_train = X_train_all, y_train_all

tfidf = TfidfVectorizer()
X_train_vec = tfidf.fit_transform(X_train)
X_val_vec = tfidf.transform(X_val)
X_test_vec = tfidf.transform(X_test)

model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_vec, y_train)
print("✅ 模型训练完毕！\n")

# ===================== 2. 模型评估与可视化保存 =====================
print("📈 正在生成评估指标并保存图片...")

# --- 计算三集准确率 ---
y_train_pred = model.predict(X_train_vec)
y_val_pred = model.predict(X_val_vec)
y_test_pred = model.predict(X_test_vec)
acc_train = accuracy_score(y_train, y_train_pred)
acc_val = accuracy_score(y_val, y_val_pred)
acc_test = accuracy_score(y_test, y_test_pred)

# 测试集详细指标
y_prob = model.predict_proba(X_test_vec)[:, 1]
prec = precision_score(y_test, y_test_pred)
rec = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
cm = confusion_matrix(y_test, y_test_pred)

print(f"  • 训练集准确率: {acc_train:.4f}")
print(f"  • 验证集准确率: {acc_val:.4f}")
print(f"  • 测试集准确率: {acc_test:.4f}")
print(f"  • 测试集 Precision/Recall/F1: {prec:.4f} / {rec:.4f} / {f1:.4f}")

# --- 图1: 三集准确率对比柱状图 ---
plt.figure(figsize=(8, 6))
datasets = ['训练集', '验证集', '测试集']
accs = [acc_train, acc_val, acc_test]
colors_bar = ['#3498DB', '#2ECC71', '#F39C12']
bars = plt.bar(datasets, accs, color=colors_bar, width=0.6, edgecolor='white', linewidth=1.2)
for bar, acc in zip(bars, accs):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f'{acc * 100:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.ylim(0, 1.05)
plt.ylabel('准确率', fontsize=12)
plt.title('训练集 / 验证集 / 测试集 准确率对比', fontsize=14, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "Accuracy_Comparison.png"), dpi=150)
plt.close()

# --- 图2: 全局积极/消极占比饼图 ---
plt.figure(figsize=(6, 6))
plt.pie([pos_count, neg_count], explode=(0.05, 0), labels=['积极情感', '消极情感'],
        colors=['#2ECC71', '#E74C3C'], autopct='%1.1f%%', startangle=90, shadow=True, textprops={'fontsize': 12})
plt.title('全数据集情感分布', fontsize=14, fontweight='bold')
plt.axis('equal')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "Dataset_Class_Distribution.png"), dpi=150)
plt.close()


# --- 图3: 各子集积极/消极占比饼图 ---
def plot_subset_pie(ax, y_true, title):
    pos = int((y_true == 1).sum())
    neg = len(y_true) - pos
    ax.pie([pos, neg], labels=['积极', '消极'], colors=['#2ECC71', '#E74C3C'],
           autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
    ax.set_title(title, fontsize=12, fontweight='bold')


fig, axes = plt.subplots(1, 3, figsize=(12, 4))
plot_subset_pie(axes[0], y_train, '训练集')
plot_subset_pie(axes[1], y_val, '验证集')
plot_subset_pie(axes[2], y_test, '测试集')
plt.suptitle('各子集情感分布对比', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "Subset_Class_Distribution.png"), dpi=150)
plt.close()

# --- 图4: Sigmoid / ROC 曲线 ---
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='#E74C3C', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.fill_between(fpr, tpr, alpha=0.1, color='#E74C3C')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Sigmoid Output / ROC Curve', fontsize=14)
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "Sigmoid_ROC_Curve.png"), dpi=150)
plt.close()

# --- 图5: 混淆矩阵热力图 ---
plt.figure(figsize=(7, 6))
im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xticks([0, 1], ['Negative', 'Positive'], fontsize=11)
plt.yticks([0, 1], ['Negative', 'Positive'], fontsize=11)
thresh = cm.max() / 2.
for i in range(2):
    for j in range(2):
        plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center",
                 fontsize=16, fontweight='bold', color="white" if cm[i, j] > thresh else "black")
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.title('Confusion Matrix', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "Confusion_Matrix.png"), dpi=150)
plt.close()

# --- 图6: 综合指标柱状图 ---
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metrics_values = [acc_test, prec, rec, f1]
colors_m = ['#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
plt.figure(figsize=(9, 6))
bars = plt.bar(metrics_names, metrics_values, color=colors_m, width=0.5, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, metrics_values):
    plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
             f'{val:.4f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
plt.ylim(0, 1.15)
plt.title('Test Set Metrics Summary', fontsize=14)
plt.ylabel('Score', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "Metrics_Summary.png"), dpi=150)
plt.close()

print(f"✅ 所有评估图片已保存至: {SAVE_DIR}\n")
print("=" * 50)


# ===================== 3. 中文绘制辅助函数 =====================
def put_chinese_text(img, text, position, font_path="simhei.ttf", font_size=24, color=(255, 255, 255)):
    pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


# ===================== 4. 百度语音录音与识别（支持R键重录） =====================
def record_audio_to_wav(duration=10):
    import pyaudio
    RATE, CHANNELS, FORMAT = 16000, 1, pyaudio.paInt16
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=1024)
    frames = [stream.read(1024, exception_on_overflow=False) for _ in range(int(RATE / 1024 * duration))]
    stream.stop_stream();
    stream.close();
    p.terminate()
    tmp_path = os.path.join(tempfile.gettempdir(), "baidu_asr_temp.wav")
    with wave.open(tmp_path, 'wb') as wf:
        wf.setnchannels(CHANNELS);
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE);
        wf.writeframes(b''.join(frames))
    return tmp_path


def camera_voice_input():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("⚠️ 无法打开摄像头");
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    state = {"status": "🎙️ 正在聆听(最长10秒)...", "text": "", "done": False}

    def voice_thread_func():
        try:
            wav_path = record_audio_to_wav(duration=10)
            state["status"] = "⏳ 正在识别中..."
            with open(wav_path, 'rb') as f:
                audio_data = f.read()
            result = baidu_client.asr(audio_data, 'wav', 16000, {'dev_pid': 1537})
            if result.get('err_no') == 0:
                state["text"] = result['result'][0]
                state["status"] = "✅ 按 Enter 确认，按 R 重新录音"
            else:
                state["status"] = f"⚠️ 识别失败，按 R 重试"
        except Exception as e:
            state["status"] = f"⚠️ 语音异常，按 R 重试"
        finally:
            state["done"] = True

    def start_voice_recognition():
        state.update({"done": False, "text": "", "status": "🎙️ 正在聆听(最长10秒)..."})
        threading.Thread(target=voice_thread_func, daemon=True).start()

    start_voice_recognition()
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        panel_width = 350
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - panel_width, 0), (w, h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        frame = put_chinese_text(frame, "状态:", (w - panel_width + 20, 30), font_size=20)
        frame = put_chinese_text(frame, state["status"], (w - panel_width + 20, 60), font_size=18, color=(0, 255, 0))
        frame = put_chinese_text(frame, "您说的话:", (w - panel_width + 20, 120), font_size=20)
        display_text = state["text"][:30] + "..." if len(state["text"]) > 30 else state["text"]
        frame = put_chinese_text(frame, display_text, (w - panel_width + 20, 150), font_size=22, color=(0, 255, 255))

        # ✅ 固定提示语
        frame = put_chinese_text(frame, "💡 如果语音识别错误，", (w - panel_width + 20, 220), font_size=16,
                                 color=(200, 200, 200))
        frame = put_chinese_text(frame, "   请按“R”键重新识别", (w - panel_width + 20, 245), font_size=16,
                                 color=(200, 200, 200))
        frame = put_chinese_text(frame, "操作: [Enter]确认 [Q]退出", (w - panel_width + 20, h - 40), font_size=16,
                                 color=(200, 200, 200))

        cv2.imshow("Smart Emotion Assistant", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): cap.release(); cv2.destroyAllWindows(); return None
        if key == ord('r'): start_voice_recognition()
        if key == 13 and state["done"] and state["text"]:
            cap.release();
            cv2.destroyAllWindows();
            return state["text"]


# ===================== 5. 主循环：多轮记忆 + AI纠错 =====================
history = [{"role": "system", "content": "你是一个高情商、温暖体贴的心理助手。请结合用户的情绪状态进行共情和回复。"}]
print("\n🎉 欢迎使用智能情感交互系统！")

while True:
    mode = input("\n请选择输入模式 (1.文本 / 2.语音 / q.退出)：")
    if mode.lower() == "q": break
    sentence = None
    if mode == "1":
        sentence = input("📝 请输入文本：")
    elif mode == "2":
        sentence = camera_voice_input()
    else:
        print("⚠️ 无效选择"); continue
    if not sentence or not sentence.strip(): continue

    print(f"🗣️ 原始识别语句：{sentence}")

    # ✅ AI智能纠错
    try:
        fix_resp = Generation.call(model="qwen-turbo", messages=[
            {"role": "user",
             "content": f"你是语音纠错助手。以下句子可能有同音错字，请修正后仅返回正确句子，无错则原样返回，不加任何解释：{sentence}"}
        ], result_format="message")
        if fix_resp.status_code == 200:
            corrected = fix_resp.output.choices[0].message.content.strip()
            if corrected != sentence:
                print(f"✨ 智能纠错结果：{corrected}")
                sentence = corrected
    except Exception as e:
        print(f"⚠️ 纠错异常，使用原句: {e}")

    # 情感预测
    clean_sent = clean_text(sentence)
    prob = model.predict_proba(tfidf.transform([clean_sent]))[0]
    emotion = "正面情感" if prob[1] > prob[0] else "负面情感"
    confidence = f"{max(prob):.2%}"

    prompt = f"""【后台数据】：用户输入"{sentence}"，情感为【{emotion}】。
【任务】：作为温暖体贴的倾听者，结合历史对话和当前情感给出自然得体的回复。
【要求】：自然衔接上下文，保持真人朋友口吻，绝不提及"情感分析"等技术词汇。"""

    history.append({"role": "user", "content": prompt})
    if len(history) > 21: history = [history[0]] + history[-20:]

    try:
        resp = Generation.call(model="qwen-turbo", messages=history, result_format="message")
        ai_reply = resp.output.choices[0].message.content if resp.status_code == 200 else f"[生成失败: {resp.message}]"
    except Exception as e:
        ai_reply = f"[回复失败: {e}]"

    print(f"\n📊 情感判定：{emotion} (概率：{confidence})")
    print(f"🤖 AI 回复：{ai_reply}\n")
    history.append({"role": "assistant", "content": ai_reply})
    # 替换原代码中 with open(...) as f: f.write(...) 这一段
    log_path = os.path.join(SAVE_DIR, "实时预测记录.txt")
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            log_line = f"输入：{sentence} | 情感：{emotion}({confidence}) | AI回复：{ai_reply}\n"
            f.write(log_line)
            f.flush()  # ✅ 强制立即写入磁盘，避免缓冲导致内容为空
        print(f"💾 预测记录已保存至: {log_path}")  # ✅ 控制台可见确认
    except Exception as e:
        print(f"❌ 写入预测记录失败: {e}")  # ✅ 若有权限/路径问题会直接报错

print("🎉 程序运行完毕！")