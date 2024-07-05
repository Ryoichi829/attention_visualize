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

dropbox_link = "https://www.dropbox.com/scl/fi/89zfk7npuo5suivpkox97/jawiki.word_vectors.300d.bin?rlkey=4hi0dkpr16plbsdb2w37v3u1r&st=3miejyz1&dl=1"

def download_file(url, destination):
    response = requests.get(url, stream=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

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
    sns.heatmap(attention_weights, xticklabels=tokens, yticklabels=tokens, ax=ax, annot=True, fmt=".2f", cmap="viridis")
    ax.xaxis.tick_top()
    ax.set_yticklabels(tokens, rotation=0)
    ax.set_ylabel('query token')
    ax.set_title("Self-Attention Heatmap\n\nkey token")
    return fig

def tokenize_japanese(text):
    tokenizer = Tokenizer()
    tokens = [token.surface for token in tokenizer.tokenize(text)]
    return tokens

@st.cache(allow_output_mutation=True)
def load_word_vectors():
    temp_path = "/tmp/jawiki.word_vectors.300d.bin"
    if not os.path.exists(temp_path):
        download_file(dropbox_link, temp_path)
    model = KeyedVectors.load_word2vec_format(temp_path, binary=True)
    return model

def main():
    st.title("Self-Attention Mechanism Visualization (Japanese)")

    # モデルのロードをローディングインジケーターで包む
    with st.spinner('モデル(800MB)をロード中...しばらくお待ちください。'):
        # load pre-trained word vectors
        word_vectors = load_word_vectors()

    sentence = st.text_area("Enter a sentence to visualize attention", "昨日、友達と映画を見に行った。とても面白かった。")
    if st.button("Visualize"):
        tokens = tokenize_japanese(sentence)
        # st.write('tokens', tokens)
        vocab = {word: idx for idx, word in enumerate(set(tokens))}
        token_ids = np.array([vocab[word] for word in tokens])

        embedding_dim = 300
        embeddings = np.zeros((len(vocab), embedding_dim))
        
        # st.write('vocab:', vocab)
        for word, idx in vocab.items():
            if word in word_vectors:
                embeddings[idx] = word_vectors[word]
            else:
                embedding[idx] = np.random.uniform(-0.25, 0.25, embedding_dim)

        Q = embeddings[token_ids]
        K = embeddings[token_ids]
        V = embeddings[token_ids]

        # st.write('Q.shape=', Q.shape)
        # st.write('Q=', Q)

        output, attention_weights = self_attention(Q, K, V)
        # st.write("Tokens:", '　'.join(tokens))

        fig = plot_attention(tokens, attention_weights)
        st.pyplot(fig)

if __name__ == "__main__":
    main()
