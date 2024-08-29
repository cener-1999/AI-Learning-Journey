import torch
import torch.nn as nn

sentences = ['Amy is a teacher', 'Priska is a Boss', 'Cindy is a Boss', 'Bob is a Student', 'Jack is a Student']
words = ' '.join(sentences).split()
words_set = set(words)
word_to_index = dict(zip(words_set, range(len(words_set))))
index_to_word = {index: word for word, index in word_to_index.items()}
vocab_size = len(words_set)


def create_similarity_matrix(sentence, window_size=2):
    data = []
    for sentence in sentence:
        words = sentence.split()
        for index, word in enumerate(words):
            for neighbour in words[max(0, index - window_size): min(vocab_size-1, index + window_size)]:
                if neighbour != word:
                    data.append((word, neighbour))
    return data


def one_hot_encode(word, word_to_index):
    tensor = torch.zeros(len(word_to_index))
    tensor[word_to_index[word]] = 1
    return tensor

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.input_to_hidden = nn.Linear(vocab_size, embedding_size, bias=False)
        self.hidden_to_output = nn.Linear(embedding_size, vocab_size, bias=False)

    def forward(self, X):
        hidden = self.input_to_hidden(X)
        output = self.hidden_to_output(hidden)
        return output

skipgram_model = SkipGram(vocab_size, embedding_size=2)
print(skipgram_model)

