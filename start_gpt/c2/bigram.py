from collections import defaultdict, Counter

def tokenize(text):
    return [char for char in text]


def count_ngrams(corpus, n):
    ngrams_count = defaultdict(Counter)
    for text in corpus:
        tokens = tokenize(text)
        for i in range(len(tokens) - n + 1): #代表可能分成的组数，就是滑动窗口可以滑动的次数，最后n个词语是无法滑动的
            ngrams = tuple(tokens[i: i+n])
            prefix = ngrams[:-1]
            token = ngrams[-1]
            ngrams_count[prefix][token] += 1
    return ngrams_count


def ngram_probs(ngrams_counts):
    ngram_probs = defaultdict(Counter)
    for prefix, counts in ngrams_counts.items():
        total_count = sum(counts.values())
        for token, count in counts.items():
            ngram_probs[prefix][token] = count / total_count
    return ngram_probs

def generate_next_token(ngram_probs, prefix: tuple):
    if prefix not in ngram_probs:
        return None
    next_token_probs = ngram_probs[prefix]
    next_token = max(next_token_probs, key=next_token_probs.get)
    return next_token

def generate_sentence(ngram_probs, prefix, n, length=6):
    tokens = list(prefix)
    for _ in range(length - len(prefix)):
        next_token = generate_next_token(ngram_probs, tuple(tokens[-(n-1):]))
        if not next_token:
            break
        tokens.append(next_token)
    return "".join(tokens)


corpus = ["我喜欢吃苹果",
          "我喜欢吃香蕉",
          "她喜欢吃葡萄",
          "他不喜欢吃香蕉",
          "他喜欢吃苹果",
          "她喜欢吃草莓",]

bigrams = count_ngrams(corpus, 2)
# print('bigrams 词频')
# for prefix, counts in bigrams.items():
#     print(f"{prefix}: {dict(counts)}")≠≠
bigrams_probs = ngram_probs(bigrams)
# print('\n bigram 出现的概率')
# for prefix, probs in bigrams_probs.items():
#     print(f"{prefix}: {dict(probs)}")
generated_text = generate_sentence(bigrams,'我', 2)
print(generated_text)


