# -*- coding: utf-8 -*-
"""
Streamlit: Transformer風 自己注意ヒートマップ（改良・完全版）
- 位置エンコーディング（sin/cos）
- W_Q, W_K, W_V による線形射影
- マルチヘッド対応（平均 or 指定ヘッドを表示）
- L2正規化 / 対角マスク / 温度（temperature）
- 再現性のための乱数シード

※ これは「Transformerの自己注意に近づけた可視化」であり、学習済みTransformerの attention そのものではありません。
   より厳密に行うには、日本語BERT等から attention を直接取得してください。
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
from janome.tokenizer import Tokenizer
from gensim.models import KeyedVectors
import requests
import os

# =========================
# 設定
# =========================
DROPBOX_LINK = "https://www.dropbox.com/scl/fi/89zfk7npuo5suivpkox97/jawiki.word_vectors.300d.bin?rlkey=4hi0dkpr16plbsdb2w37v3u1r&st=3miejyz1&dl=1"
EMBED_DIM = 300         # word2vec の次元
MAX_TOKENS = 20         # 可視化の最大トークン数（多すぎると見づらい）
DEFAULT_SENTENCE = "昨日インスタに写真をアップした。"

# =========================
# ユーティリティ
# =========================

def download_file(url: str, destination: str):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

def row_softmax(x: np.ndarray) -> np.ndarray:
    # x: (..., N)
    x = x - np.max(x, axis=-1, keepdims=True)
    np.exp(x, out=x)
    denom = np.sum(x, axis=-1, keepdims=True)
    return x / (denom + 1e-12)

@st.cache_resource(show_spinner=False)
def load_word_vectors() -> KeyedVectors:
    temp_path = "/tmp/jawiki.word_vectors.300d.bin"
    if not os.path.exists(temp_path):
        download_file(DROPBOX_LINK, temp_path)
    model = KeyedVectors.load_word2vec_format(temp_path, binary=True)
    return model

# =========================
# 前処理
# =========================

def tokenize_japanese(text: str):
    tokenizer = Tokenizer()
    tokens = [t.surface for t in tokenizer.tokenize(text)]
    # 長すぎる場合は先頭MAX_TOKENSまで
    if len(tokens) > MAX_TOKENS:
        st.warning(f"単語数が多いため、先頭{MAX_TOKENS}個に制限しました。")
        tokens = tokens[:MAX_TOKENS]
    return tokens

# 先頭出現順でユニーク化

def unique_in_order(seq):
    return list(dict.fromkeys(seq))

# =========================
# 位置エンコーディング（sinusoidal）
# =========================

def sinusoidal_positional_encoding(n_tokens: int, d_model: int) -> np.ndarray:
    # Vaswani et al. (2017)
    position = np.arange(n_tokens)[:, None]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe = np.zeros((n_tokens, d_model), dtype=np.float32)
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe

# =========================
# 重み行列の用意（再現性のためseedで固定）
# =========================

@st.cache_resource(show_spinner=False)
def init_projection_matrices(seed: int, d_model: int, num_heads: int):
    assert d_model % num_heads == 0, "d_model は num_heads で割り切れる必要があります"
    rng = np.random.default_rng(seed)
    d_k = d_model // num_heads
    scale = 1.0 / np.sqrt(d_model)
    W_Q = rng.normal(0.0, scale, size=(d_model, num_heads * d_k)).astype(np.float32)
    W_K = rng.normal(0.0, scale, size=(d_model, num_heads * d_k)).astype(np.float32)
    W_V = rng.normal(0.0, scale, size=(d_model, num_heads * d_k)).astype(np.float32)
    return W_Q, W_K, W_V

# =========================
# マルチヘッド自己注意
# =========================

def multihead_self_attention(X: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray,
                             num_heads: int, temperature: float = 1.0, mask_diagonal: bool = False):
    """
    X: (N, d_model)
    W_*: (d_model, num_heads * d_k)
    戻り値:
      out: (N, d_model)
      attn: (num_heads, N, N)  # 各ヘッドの注意重み
    """
    N, d_model = X.shape
    d_k = d_model // num_heads

    Q = X @ W_Q  # (N, h*d_k)
    K = X @ W_K
    V = X @ W_V

    # (h, N, d_k)
    Q = Q.reshape(N, num_heads, d_k).transpose(1, 0, 2)
    K = K.reshape(N, num_heads, d_k).transpose(1, 0, 2)
    V = V.reshape(N, num_heads, d_k).transpose(1, 0, 2)

    # スコア: (h, N, N)
    # scores[h, i, j] = Q[h,i]·K[h,j] / sqrt(d_k)
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)

    if temperature != 1.0:
        scores = scores / max(temperature, 1e-6)

    if mask_diagonal:
        for h in range(num_heads):
            np.fill_diagonal(scores[h], -1e9)

    attn = row_softmax(scores)  # (h, N, N)

    # 出力（使わないが計算はしておく）
    out_heads = np.matmul(attn, V)  # (h, N, d_k)
    out = out_heads.transpose(1, 0, 2).reshape(N, num_heads * d_k)  # (N, d_model)

    return out, attn

# =========================
# プロット
# =========================

def plot_attention(tokens, attn: np.ndarray, head_mode: str, head_index: int, annotate: bool, cmap: str):
    """
    attn: (h, N, N)
    head_mode: "平均" or "ヘッド指定"
    """
    if head_mode == "平均":
        A = attn.mean(axis=0)
        title = "Self-Attention Heatmap（全ヘッド平均）"
    else:
        H = attn.shape[0]
        head_index = max(0, min(head_index, H-1))
        A = attn[head_index]
        title = f"Self-Attention Heatmap（ヘッド {head_index}）"

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(A, xticklabels=tokens, yticklabels=tokens, ax=ax,
                annot=annotate, fmt=".2f" if annotate else "",
                cmap=cmap, vmin=0.0, vmax=1.0)
    ax.xaxis.tick_top()
    ax.set_yticklabels(tokens, rotation=0)
    ax.set_ylabel('query token')
    ax.set_title(title + "\n\nkey token")
    plt.tight_layout()
    return fig

# =========================
# メイン（UI）
# =========================

def main():
    st.title("Transformer風 自己注意ヒートマップ（改良版）")

    # サイドバー設定
    st.sidebar.header("設定")
    seed = st.sidebar.number_input("乱数シード", min_value=0, max_value=10_000, value=0, step=1)

    # 300を割り切るヘッド数のみ選択肢に
    valid_heads = [1,2,3,4,5,6,10,12,15,20,25,30,50,60,75,100,150]
    num_heads = st.sidebar.selectbox("ヘッド数", valid_heads, index=3)  # デフォルト4

    l2norm = st.sidebar.checkbox("埋め込みをL2正規化", value=True)
    use_posenc = st.sidebar.checkbox("位置エンコーディングを加算", value=True)
    mask_diag = st.sidebar.checkbox("対角マスク（自己参照を抑制）", value=False)

    temperature = st.sidebar.slider("temperature（低いほど尖る）", min_value=0.3, max_value=2.0, value=1.0, step=0.1)

    annotate = st.sidebar.checkbox("セルに数値を表示", value=True)
    cmap = st.sidebar.selectbox("カラーマップ", ["viridis","magma","plasma","cividis","YlGnBu","rocket"], index=0)

    oov_mode = st.sidebar.selectbox("未知語の埋め方", ["ゼロ","ランダム(-0.25~0.25)"], index=0)

    st.markdown("""
    **注意**: これは word2vec の静的埋め込み + 位置エンコーディング + ランダムな W_Q/W_K/W_V による **擬似的な自己注意** です。
    厳密なTransformerの注意ではありません。
    """)

    # モデルロード
    with st.spinner('日本語Wikipediaベクトル（約800MB）をロード中...'):
        wv = load_word_vectors()

    sentence = st.text_area("文章を入力してください。", DEFAULT_SENTENCE)

    if st.button("Visualize"):
        np.random.seed(seed)
        rng = np.random.default_rng(seed)

        tokens = tokenize_japanese(sentence)
        tokens_unique = unique_in_order(tokens)
        vocab = {w:i for i, w in enumerate(tokens_unique)}
        token_ids = np.array([vocab[w] for w in tokens], dtype=int)

        # 語彙ごとの埋め込み行列（ユニーク語彙単位）
        E = np.zeros((len(tokens_unique), EMBED_DIM), dtype=np.float32)
        for w, idx in vocab.items():
            if w in wv:
                E[idx] = wv[w]
            else:
                if oov_mode.startswith("ゼロ"):
                    E[idx] = np.zeros(EMBED_DIM, dtype=np.float32)
                else:
                    E[idx] = rng.uniform(-0.25, 0.25, EMBED_DIM).astype(np.float32)

        # L2正規化（オプション）
        if l2norm:
            norms = np.linalg.norm(E, axis=1, keepdims=True)
            E = E / (norms + 1e-9)

        # トークン列へ（順序付き）
        X = E[token_ids]  # (N, d_model)

        # 位置エンコーディング
        if use_posenc:
            pe = sinusoidal_positional_encoding(len(tokens), EMBED_DIM)
            X = X + pe

        # 射影行列
        try:
            W_Q, W_K, W_V = init_projection_matrices(seed, EMBED_DIM, num_heads)
        except AssertionError:
            st.error(f"EMBED_DIM={EMBED_DIM} は num_heads={num_heads} で割り切れません。")
            return

        # 自己注意
        _, attn = multihead_self_attention(X, W_Q, W_K, W_V, num_heads=num_heads,
                                           temperature=temperature, mask_diagonal=mask_diag)

        # 表示モード
        head_mode = st.radio("表示モード", ["平均","ヘッド指定"], horizontal=True)
        head_index = 0
        if head_mode == "ヘッド指定":
            head_index = st.slider("ヘッド番号", min_value=0, max_value=num_heads-1, value=0, step=1)

        fig = plot_attention(tokens, attn, head_mode, head_index, annotate, cmap)
        st.pyp
if __name__ == "__main__":
    main()
