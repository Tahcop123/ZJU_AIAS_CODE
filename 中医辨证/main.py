import csv
import json
import time
import re
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os

# 配置
client = OpenAI(
    api_key="",
    base_url=""
)

EMBEDDING_API = ""
RERANK_API = ""
EMBEDDING_KEY = ""
RERANK_KEY = EMBEDDING_KEY

CSV_PATH = './datasets/68f201a04e0f8ad44a62069b-momodel/train_data.csv'

TOP_K = 20         # coarse retrieval
FINAL_K = 3        # rerank后选择给LLM的示例


# 缓存
predict_cache = {}
embedding_cache = {}
TRAIN_DB = {}
EMBEDDING_DB = {}
KNOWN_ZX_LIST = []
DATA_LOADED = False


# JSON 提取
def safe_json_extract(content):
    m = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
    if m:
        try: return json.loads(m.group(1).strip())
        except: pass
    try: return json.loads(content)
    except: return None


# 输入规范化
def rewrite_symptom(symptom):
    """ 使用 LLM 对症状进行轻度标准化 """
    prompt = f"""你是医学术语规范化专家，请将症状进行归一化：
    输入：{symptom}
    要求：
    - 输出简化后的症状（不解释）
    - 保留医学语义
    """
    try:
        resp = client.chat.completions.create(
            model="ernie-3.5-8k",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        cleaned = resp.choices[0].message.content.strip()
        return cleaned if cleaned else symptom
    except:
        return symptom


# 加载数据
def _ensure_data_loaded():
    global TRAIN_DB, EMBEDDING_DB, KNOWN_ZX_LIST, DATA_LOADED

    if DATA_LOADED:
        return

    target_path = CSV_PATH
    if not os.path.exists(target_path):
        print(f"训练数据不存在：{CSV_PATH}")
        DATA_LOADED = True
        return

    print("加载训练集...")
    temp_zx = set()

    with open(target_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 4: continue
            s, zx, zf = row[1].strip(), row[2].strip(), row[3].strip()
            TRAIN_DB[s] = {"zx": zx, "zf": zf}
            temp_zx.add(zx)

    KNOWN_ZX_LIST = list(temp_zx)

    print("预计算 embeddings...")
    for s in TRAIN_DB:
        EMBEDDING_DB[s] = get_embedding(s)

    print(f"加载完成：{len(TRAIN_DB)} 条病例，{len(KNOWN_ZX_LIST)} 个证型")
    DATA_LOADED = True


# 获取embedding
def get_embedding(text):
    text = text.strip()
    if not text:
        return np.zeros(1024).tolist()

    if text in embedding_cache:
        return embedding_cache[text]

    payload = {"model": "bge-large-zh", "input": [text]}
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {EMBEDDING_KEY}'
    }

    try:
        r = requests.post(EMBEDDING_API, headers=headers, json=payload, timeout=6)
        if r.status_code == 200:
            emb = np.array(r.json()["data"][0]["embedding"])
            emb = emb / (np.linalg.norm(emb) + 1e-9)
            emb = emb.tolist()
            embedding_cache[text] = emb
            return emb
    except:
        pass

    return np.zeros(1024).tolist()


# Rerank
def rerank(symptom, candidates):
    """ 使用 bge-reranker-large 对 coarse 召回结果重排序 """
    payload = {
        "model": "bge-reranker-large",
        "input": [
            {"query": symptom, "documents": candidates}
        ]
    }
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {RERANK_KEY}'
    }

    try:
        r = requests.post(RERANK_API, json=payload, headers=headers, timeout=8)
        if r.status_code == 200:
            scores = r.json()["data"][0]["scores"]
            pairs = list(zip(candidates, scores))
            pairs.sort(key=lambda x: x[1], reverse=True)
            return [p[0] for p in pairs]
    except:
        pass

    return candidates  # fallback


# 检索
def retrieve_similar_cases(symptom, top_k=TOP_K, final_k=FINAL_K):
    if not EMBEDDING_DB:
        return []

    # 
    emb_in = get_embedding(symptom)
    sims = [(s, float(cosine_similarity([emb_in], [emb])[0][0]))
            for s, emb in EMBEDDING_DB.items()]

    sims.sort(key=lambda x: x[1], reverse=True)
    coarse = [s for s, _ in sims[:top_k]]

    # rerank 
    reranked = rerank(symptom, coarse)

    
    selected = reranked[:final_k]

    examples = []
    for s in selected:
        zx, zf = TRAIN_DB[s]["zx"], TRAIN_DB[s]["zf"]
        examples.append(
            f"症状：{s}\n分析：根据症状判断为 {zx}\n证型：{zx}\n治法：{zf}"
        )

    return examples


# 调LLM
def call_large_model(symptom, examples):
    refs = ", ".join([f"‘{x}’" for x in KNOWN_ZX_LIST])

    system_prompt = f"""
你是中医专家，请根据症状推断证型与治法。

【参考病例示例（含推理）】
{chr(10).join(examples)}

【证型候选库】
{refs}

输出严格 JSON：
{{
  "证型": "<从候选库中选择>",
  "治法": "<对应治法>"
}}
禁止输出 JSON 以外的内容。
"""

    try:
        resp = client.chat.completions.create(
            model="ernie-4.5-0.3b",
            temperature=0.01,
            max_completion_tokens=128,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"症状：{symptom}"}
            ]
        )
        return safe_json_extract(resp.choices[0].message.content)
    except:
        return None


# 输出纠正
def correct_output(zx, zf):
    if zx not in KNOWN_ZX_LIST:
        # fallback 到最相近的词
        best = max(KNOWN_ZX_LIST, key=lambda x: similarity(x, zx))
        return best, zf
    return zx, zf


# 核心 predict 函数
def predict(symptom: str):
    symptom = symptom.strip()

    _ensure_data_loaded()

    if symptom in TRAIN_DB:
        return TRAIN_DB[symptom]["zx"], TRAIN_DB[symptom]["zf"]

    if symptom in predict_cache:
        return predict_cache[symptom]

    clean_symptom = rewrite_symptom(symptom)
    examples = retrieve_similar_cases(clean_symptom)

    raw = call_large_model(clean_symptom, examples) or {}
    zx = raw.get("证型", "未知").strip()
    zf = raw.get("治法", "对症治疗").strip()

    zx, zf = correct_output(zx, zf)
    predict_cache[symptom] = (zx, zf)

    return zx, zf


# scoring
def similarity(a, b):
    emb1 = get_embedding(a)
    emb2 = get_embedding(b)
    return float(cosine_similarity([emb1], [emb2])[0][0])


def score(pred_zx, pred_zf, gt_zx, gt_zf):
    s1 = [similarity(p, g) for p, g in zip(pred_zx, gt_zx)]
    s2 = [similarity(p, g) for p, g in zip(pred_zf, gt_zf)]
    return (np.mean(s1) + np.mean(s2)) / 2 * 100



# 运行测试
if __name__ == "__main__":
    _ensure_data_loaded()

    test_symptoms = []
    gt_zx = []
    gt_zf = []

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for i, row in enumerate(reader):
            if i >= 50: break
            test_symptoms.append(row[1])
            gt_zx.append(row[2])
            gt_zf.append(row[3])

    pred_zx, pred_zf = [], []

    for s in test_symptoms:
        z1, z2 = predict(s)
        pred_zx.append(z1)
        pred_zf.append(z2)

    print("分数：", score(pred_zx, pred_zf, gt_zx, gt_zf))
