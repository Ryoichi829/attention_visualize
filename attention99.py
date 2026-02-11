# 日本語学習済みモデルをdropboxから読み込む
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
from janome.tokenizer import Tokenizer
from gensim.models import KeyedVectors
import requests
import os

dropbox_link = "https://www.dropbox.com/scl/fi/xvpmnduzus1xt4x1oqj09/jawiki.entity_vectors.100d_200_000.bin?rlkey=gwxb0swvg23ekh9mhezjw6f6j&st=p752egym&dl=1"

def download_file(url, destination):
    response = requests.get(url, stream=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

def softmax(x, T=1.0):
    x = x / T
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def self_attention(Q, K, V):
    scores = np.dot(Q, K.T) / np.sqrt(K.shape[-1])
    # 対角マスク（自分自身への注意を禁止）
    np.fill_diagonal(scores, -1e9)
    attention_weights = softmax(scores, T=1.5) # 2.0でマイルドにする
    output = np.dot(attention_weights, V)
    return output, attention_weights

def plot_attention(tokens, attention_weights):
    fig, ax = plt.subplots(figsize=(10, 10))
    n = len(tokens)
    # 通常描画
    sns.heatmap(attention_weights, xticklabels=False, yticklabels=False, ax=ax, annot=True, fmt=".2f", cmap="viridis")
    
    # 対角線だけグレーを上書き
    diag = np.eye(n, dtype=bool)
    overlay = np.full((n, n), np.nan, dtype=float)
    overlay[diag] = 1.0
    sns.heatmap(overlay, ax=ax, cmap=sns.color_palette(["#CCCCCC"]), cbar=False, annot=False, xticklabels=False, yticklabels=False)
    
    # 対角線の数字を消す
    for t in ax.texts:
        x, y = t.get_position()  # (col+0.5, row+0.5)
        col = int(round(x - 0.5))
        row = int(round(y - 0.5))
        if row == col:
            t.set_text("")

    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels(tokens, rotation=0)
    ax.set_yticklabels(tokens, rotation=0)

    ax.xaxis.tick_top()
    ax.set_yticklabels(tokens, rotation=0)
    ax.set_ylabel('query token')
    ax.set_title("Self-Attention Heatmap\n\nkey token")
            
    return fig

def tokenize_japanese(text):
    tokenizer = Tokenizer()
    tokens = [token.surface for token in tokenizer.tokenize(text)]
    if len(tokens) > 15:
        st.warning(f"単語数が多いため、先頭15個に制限しました。")
        tokens = tokens[:15]
    return tokens

# @st.cache(allow_output_mutation=True)
@st.cache_resource
def load_word_vectors():
    temp_path = "/tmp/jawiki.word_vectors.100d.bin"
    if not os.path.exists(temp_path):
        download_file(dropbox_link, temp_path)
    
    model = KeyedVectors.load_word2vec_format(temp_path, binary=True)
    return model

def main():
    st.title("Self-Attention Mechanism Visualization")
    word_vectors = load_word_vectors()
    sentence = st.text_area("文章を入力してください。", "昨日インスタに写真をアップした。")
    if st.button("Visualize"):
        tokens = tokenize_japanese(sentence)
        # st.write('tokens', tokens)
        vocab = {word: idx for idx, word in enumerate(set(tokens))}
        token_ids = np.array([vocab[word] for word in tokens])

        embedding_dim = 100
        embeddings = np.zeros((len(vocab), embedding_dim))
        
        # st.write('vocab:', vocab)
        for word, idx in vocab.items():
            if word in word_vectors:
                embeddings[idx] = word_vectors[word][:embedding_dim]
            else:
                embeddings[idx] = np.random.uniform(-0.25, 0.25, embedding_dim)

        np.random.seed(0)
        
        Wq = np.random.randn(embedding_dim, embedding_dim)
        Wk = np.random.randn(embedding_dim, embedding_dim)
        Wv = np.random.randn(embedding_dim, embedding_dim)

        Q = embeddings[token_ids] @ Wq
        K = embeddings[token_ids] @ Wk
        V = embeddings[token_ids] @ Wv

        
        # Q = embeddings[token_ids]
        # K = embeddings[token_ids]
        # V = embeddings[token_ids]

        # st.write('Q.shape=', Q.shape)
        # st.write('Q=', Q)

        output, attention_weights = self_attention(Q, K, V)
        # st.write("Tokens:", '　'.join(tokens))

        fig = plot_attention(tokens, attention_weights)
        st.pyplot(fig)

        # 著作権表示
        st.write("東北大学乾研究室の日本語Wikipediaエンティティベクトルを使用しています。")

if __name__ == "__main__":
    main()
