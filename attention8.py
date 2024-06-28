import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from janome.tokenizer import Tokenizer
from gensim.models import KeyedVectors
import datetime

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def self_attention(Q, K, V):
    scores = np.dot(Q, K.T) / np.sqrt(K.shape[-1])
    attention_weights = softmax(scores)
    output = np.dot(attention_weights, V)
    return output, attention_weights

def plot_attention(tokens, attention_weights):
    plt.rcParams['font.family'] = "MS Gothic"
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(attention_weights, xticklabels=tokens, yticklabels=tokens, ax=ax, annot=True, fmt=".2f", cmap="viridis")
    ax.xaxis.tick_top()
    ax.set_yticklabels(tokens, rotation=0)
    ax.set_title("Self-Attention Heatmap")
    return fig

def tokenize_japanese(text):
    tokenizer = Tokenizer()
    tokens = [token.surface for token in tokenizer.tokenize(text)]
    return tokens

@st.cache_data(ttl=datetime.timedelta(hours=1))
def load_word_vectors(filepath):
    return KeyedVectors.load_word2vec_format(filepath, binary=True)

def main():
    st.title("Self-Attention Mechanism Visualization (Japanese)")

    # モデルのロードをローディングインジケーターで包む
    with st.spinner('モデル(800MB)をロード中...しばらくお待ちください。'):
        # load pre-trained word vectors
        word_vectors = load_word_vectors('C:/Users/ryoic/word2vec/jawiki.word_vectors.300d.bin')

    sentence = st.text_area("Enter a sentence to visualize attention", "猫が餌を食べるのは、それが美味しいからだ。")
    if st.button("Visualize"):
        tokens = tokenize_japanese(sentence)
        st.write('tokens', tokens)
        vocab = {word: idx for idx, word in enumerate(set(tokens))}
        token_ids = np.array([vocab[word] for word in tokens])

        embedding_dim = 300
        embeddings = np.zeros((len(vocab), embedding_dim))
        
        st.write('vocab:', vocab)
        for word, idx in vocab.items():
            if word in word_vectors:
                embeddings[idx] = word_vectors[word]
            else:
                embedding[idx] = np.random.uniform(-0.25, 0.25, embedding_dim)

        Q = embeddings[token_ids]
        K = embeddings[token_ids]
        V = embeddings[token_ids]

        st.write('Q=', Q)

        output, attention_weights = self_attention(Q, K, V)
        st.write("Tokens:", '  '.join(tokens))

        fig = plot_attention(tokens, attention_weights)
        st.pyplot(fig)

        # Highlighting specific relationship
        if "それ" in tokens and "餌" in tokens:
            sore_index = tokens.index("それ")
            esa_index = tokens.index("餌")
            st.write(f"Attention weight from 'それ' to '餌': {attention_weights[sore_index][esa_index]:.2f}")

if __name__ == "__main__":
    main()
