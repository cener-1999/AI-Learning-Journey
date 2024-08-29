import jieba
import numpy as np
import matplotlib.pyplot as plt

corpus = ['我特别特别喜欢看电影',
          '这部电影真的是很好看的电影',
          '今天天气真是难得的好天气',
          '我今天去看了一部电影',
          '电影院的电影都很好看']


corpus_tokenized = [list(jieba.cut(sentence)) for sentence in corpus]

words_dict = {}

for sentence in corpus_tokenized:
    for word in sentence:
        if not word in words_dict:
            words_dict[word] = len(words_dict)  # create index map
print(words_dict)

bow_vectors = []
for sentence in corpus_tokenized:
    sentence_vector = [0] * len(words_dict)
    for word in sentence:
        sentence_vector[words_dict[word]] += 1
    bow_vectors.append(sentence_vector)
print(bow_vectors)

def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_1 = np.linalg.norm(vector1)
    norm_2 = np.linalg.norm(vector2)
    cosine_similarity = dot_product / (norm_1 * norm_2)
    return cosine_similarity

similarities_matrix = np.zeros((len(corpus), len(corpus)))

for i in range(len(corpus)):
    similarities_matrix[i][i] = 1
    for j in range(i+1, len(corpus)):
        similarities_matrix[i][j] = cosine_similarity(bow_vectors[i], bow_vectors[j])
        similarities_matrix[j][i] = similarities_matrix[i][j]

plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['font.family'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots()

cax = ax.matshow(similarities_matrix, cmap=plt.colormaps['Blues'])
fig.colorbar(cax)
ax.set_xticks(range(len(corpus)))
ax.set_yticks(range(len(corpus)))
ax.set_xticklabels(corpus, rotation=45, ha='left')
ax.set_yticklabels(corpus)
plt.show()
