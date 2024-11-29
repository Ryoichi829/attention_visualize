import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
import tiktoken
import openai

# OpenAI APIキーの設定
openai.api_key = st.secrets["OPENAI_API_KEY"]

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def self_attention(Q, K, V):
    scores = np.dot(Q, K.T) / np.sqrt(K.shape[-1])
    attention_weights = softmax(scores)
    output = np.dot(attention_weights, V)
    return output, attention_weights

def plot_attention(tokens, attention_weights):
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        attention_weights,
        xticklabels=tokens,
        yticklabels=tokens,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="viridis"
    )
    ax.xaxis.tick_top()
    ax.set_yticklabels(tokens, rotation=0)
    ax.set_ylabel('query token')
    ax.set_title("Self-Attention Heatmap\n\nkey token")
    return fig

def tokenize_with_tiktoken(text):
    """tiktokenを使用してトークン化する"""
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    decoded_tokens = [encoding.decode_single_token_bytes(tok).decode("utf-8") for tok in tokens]
    return tokens, decoded_tokens

def get_embeddings_from_openai(text):
    """OpenAIのtext-embedding-ada-002を使用してテキストの埋め込みを取得"""
    response = openai.Embedding.create(
        model="text-embedding-ada-002",  # 適宜text-embedding-3-smallなどに変更可能
        input=text
    )
    embeddings = response['data'][0]['embedding']
    return np.array(embeddings)

def main():
    st.title("Self-Attention Mechanism Visualization with OpenAI Embeddings")

    sentence = st.text_area("文章を入力してください。", "昨日、友達と映画を見に行った。とても面白かった。")
    if st.button("Visualize"):
        tokens, decoded_tokens = tokenize_with_tiktoken(sentence)
        st.write("トークン化された結果:", decoded_tokens)

        # トークンごとに埋め込みを取得
        embeddings = np.array([get_embeddings_from_openai(tok) for tok in decoded_tokens])
        st.write(f"取得した埋め込みの形状: {embeddings.shape}")

        # Q, K, Vを埋め込みベクトルで構築
        Q = embeddings
        K = embeddings
        V = embeddings

        # Self-Attentionの計算
        output, attention_weights = self_attention(Q, K, V)

        # ヒートマップをプロット
        fig = plot_attention(decoded_tokens, attention_weights)
        st.pyplot(fig)

if __name__ == "__main__":
    main()
