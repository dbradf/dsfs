from collections import defaultdict, Counter
import json
import math
import random
import re
from typing import List, Dict, Iterable, Tuple

import tqdm

from dsfs.linalg.vector import dot, Vector
from dsfs.deep_learning import Tensor, Layer, random_tensor, zeros_like, tensor_apply, tanh


def generate_bigrams(document):
    transitions = defaultdict(list)
    for prev, current in zip(document, document[1:]):
        transitions[prev].append(current)

    return transitions


def generate_using_bigrams(bigrams) -> str:
    current = "."
    result = []
    while True:
        next_word_candidates = bigrams[current]
        current = random.choice(next_word_candidates)
        result.append(current)
        if current == ".":
            return " ".join(result)


def generate_trigrams(document):
    trigrams = defaultdict(list)
    starts = []

    for prev, current, next in zip(document, document[1:], document[2:]):
        if prev == ".":
            starts.append(current)

        trigrams[(prev, current)].append(next)

    return trigrams, starts


def generating_using_trigrams(trigrams, starts) -> str:
    current = random.choice(starts)
    prev = "."
    result = [current]
    while True:
        next_word_candidates = trigrams[(prev, current)]
        next_word = random.choice(next_word_candidates)

        prev, current = current, next_word
        result.append(current)

        if current == ".":
            return " ".join(result)


Grammar = Dict[str, List[str]]

base_grammar = {
    "_S": ["_NP _VP"],
    "_NP": ["_N", "_A _NP _P _A _N"],
    "_VP": ["_V", "_V _NP"],
    "_N": ["data science", "Python", "regression"],
    "_A": ["big", "linear", "logistic"],
    "_P": ["about", "near"],
    "_V": ["learns", "trains", "tests", "is"],
}


def is_terminal(token: str) -> bool:
    return token[0] != "_"


def expand(grammar: Grammar, tokens: List[str]) -> List[str]:
    for i, token in enumerate(tokens):
        if is_terminal(token):
            continue

        replacement = random.choice(grammar[token])

        if is_terminal(replacement):
            tokens[i] = replacement
        else:
            tokens = tokens[:i] + replacement.split() + tokens[(i + 1) :]

        return expand(grammar, tokens)

    return tokens


def generate_sentence(grammar: Grammar) -> List[str]:
    return expand(grammar, ["_S"])


def sample_from(weights: List[float]) -> int:
    total = sum(weights)
    rnd = total * random.random()
    for i, w in enumerate(weights):
        rnd -= w
        if rnd <= 0:
            return i


class LDA:
    def __init__(self, documents, n_topics):
        self.n_topics = n_topics
        self.document_topic_counts = [Counter() for _ in documents]
        self.topic_word_counts = [Counter() for _ in range(n_topics)]
        self.topic_counts = [0 for _ in range(n_topics)]
        self.document_lengths = [len(document) for document in documents]
        self.distinct_words = set(word for document in documents for word in document)
        self.W = len(self.distinct_words)
        self.D = len(documents)

        self.train(documents)

    def p_topic_given_document(self, topic: int, d: int, alpha: float = 0.1) -> float:
        return (self.document_topic_counts[d][topic] + alpha) / (
            self.document_lengths[d] + self.n_topics * alpha
        )

    def p_word_given_topic(self, word: str, topic: int, beta: float = 0.1) -> float:
        return (self.topic_word_counts[topic][word] + beta) / (
            self.topic_counts[topic] + self.W * beta
        )

    def topic_weight(self, d: int, word: str, k: int) -> float:
        return self.p_word_given_topic(word, k) * self.p_topic_given_document(k, d)

    def choose_new_topic(self, d: int, word: str) -> int:
        return sample_from([self.topic_weight(d, word, k) for k in range(self.n_topics)])

    def train(self, documents):
        document_topics = [
            [random.randrange(self.n_topics) for word in document] for document in documents
        ]

        for d in range(self.D):
            for word, topic in zip(documents[d], document_topics[d]):
                self.document_topic_counts[d][topic] += 1
                self.topic_word_counts[topic][word] += 1
                self.topic_counts[topic] += 1

        for iter in tqdm.trange(1000):
            for d in range(self.D):
                for i, (word, topic) in enumerate(zip(documents[d], document_topics[d])):
                    self.document_topic_counts[d][topic] -= 1
                    self.topic_word_counts[topic][word] -= 1
                    self.topic_counts[topic] -= 1
                    self.document_lengths[d] -= 1

                    new_topic = self.choose_new_topic(d, word)
                    document_topics[d][i] = new_topic

                    self.document_topic_counts[d][new_topic] += 1
                    self.topic_word_counts[new_topic][word] += 1
                    self.topic_counts[new_topic] += 1
                    self.document_lengths[d] += 1

    def topics(self):
        for k, word_counts in enumerate(self.topic_word_counts):
            for word, count in word_counts.most_common():
                if count > 0:
                    print(k, word, count)


def cosine_similarity(v1: Vector, v2: Vector) -> float:
    return dot(v1, v2) / math.sqrt(dot(v1, v1) * dot(v2, v2))


colors = ["red", "green", "blue", "yellow", "black", ""]
nouns = ["bed", "car", "boat", "cat"]
verbs = ["is", "was", "seems"]
adberbs = ["very", "quite", "extremely", ""]
adjectives = ["slow", "fast", "soft", "hard"]


def make_sentence() -> str:
    return " ".join(
        [
            "The",
            random.choice(colors),
            random.choice(nouns),
            random.choice(verbs),
            random.choice(adberbs),
            random.choice(adjectives),
            ".",
        ]
    )


NUM_SENTENCES = 50

sentences = [make_sentence() for _ in range(NUM_SENTENCES)]


class Vocabulary:
    def __init__(self, words: List[str] = None) -> None:
        self.w2i: Dict[str, int] = {}
        self.i2w: Dict[int, str] = {}

        for word in words or []:
            self.add(word)

    @property
    def size(self) -> int:
        return len(self.w2i)

    def add(self, word: str) -> None:
        if word not in self.w2i:
            word_id = len(self.w2i)
            self.w2i[word] = word_id
            self.i2w[word_id] = word

    def get_id(self, word: str) -> int:
        return self.w2i.get(word)

    def get_word(self, word_id: int) -> str:
        return self.i2w.get(word_id)

    def one_hot_encode(self, word: str) -> Tensor:
        word_id = self.get_id(word)
        assert word_id is not None

        return [1.0 if i == word_id else 0.0 for i in range(self.size)]

    def save(self, filename: str):
        with open(filename, "w") as f:
            json.dump(self.w2i, f)

    @classmethod
    def load(cls, filename: str):
        vocab = cls()
        with open(filename) as f:
            vocab.w2i = json.load(f)
            vocab.i2w = {id: word for word, id in vocab.w2i.items()}

        return vocab


class Embedding(Layer):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.embeddings = random_tensor(num_embeddings, embedding_dim)
        self.grad = zeros_like(self.embeddings)

        self.last_input_id = None

    def forward(self, input_id: int) -> Tensor:
        self.input_id = input_id
        return self.embeddings[input_id]

    def backward(self, gradient: Tensor) -> None:
        if self.last_input_id is not None:
            zero_row = [0 for _ in range(self.embedding_dim)]
            self.grad[self.last_input_id] = zero_row

        self.last_input_id = self.input_id
        self.grad[self.input_id] = gradient

    def params(self) -> Iterable[Tensor]:
        return [self.embeddings]

    def grads(self) -> Iterable[Tensor]:
        return [self.grad]


class TextEmbedding(Embedding):
    def __init__(self, vocab: Vocabulary, embedding_dim: int) -> None:
        super().__init__(vocab.size, embedding_dim)

        self.vocab = vocab

    def __getitem__(self, word: str) -> Tensor:
        word_id = self.vocab.get_id(word)
        if word_id is not None:
            return self.embeddings[word_id]
        else:
            return None

    def closest(self, word: str, n: int = 5) -> List[Tuple[float, str]]:
        vector = self[word]
        scores = [
            (cosine_similarity(vector, self.embeddings[i]), other_word)
            for other_word, i in self.vocab.w2i.items()
        ]
        scores.sort(reverse=True)

        return scores[:n]


def create_training_data(sentences):
    inputs: List[int] = []
    targets: List[Tensor] = []

    tokenized_sentences = [re.findall("[a-z]+|[.]", sentence.lower()) for sentence in sentences]
    vocab = Vocabulary(word for sentence_words in tokenized_sentences for word in sentence_words)

    for sentence in tokenized_sentences:
        for i, word in enumerate(sentence):
            for j in [i - 2, i - 1, i + 1, i + 2]:
                if 0 <= j < len(sentence):
                    nearby_word = sentence[j]
                    inputs.append(vocab.get_id(word))
                    targets.append(vocab.one_hot_encode(nearby_word))

    return inputs, targets, vocab


class SimpleRnn(Layer):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.w = random_tensor(hidden_dim, input_dim, init="xavier")
        self.u = random_tensor(hidden_dim, hidden_dim, init="xavier")
        self.b = random_tensor(hidden_dim)

        self.reset_hidden_state()

    def reset_hidden_state(self) -> None:
        self.hidden = [0 for _ in range(self.hidden_dim)]

    def forward(self, inp: Tensor) -> Tensor:
        self.input = inp
        self.prev_hidden = self.hidden

        a = [
            (dot(self.w[h], inp) + dot(self.u[h], self.hidden) + self.b[h])
            for h in range(self.hidden_dim)
        ]

        self.hidden = tensor_apply(tanh, a)
        return self.hidden

    def backward(self, gradient: Tensor):
        a_grad = [gradient[h] * (1 - self.hidden[h] ** 2) for h in range(self.hidden_dim)]
        self.b_grad = a_grad
        self.w_grad = [
            [a_grad[h] * self.input[i] for i in range(self.input_dim)]
            for h in range(self.hidden_dim)
        ]
        self.u_grad = [
            [a_grad[h] * self.prev_hidden[h2] for h2 in range(self.hidden_dim)]
            for h in range(self.hidden_dim)
        ]
        return [
            sum(a_grad[h] * self.w[h][i] for h in range(self.hidden_dim))
            for i in range(self.input_dim)
        ]

    def params(self) -> Iterable[Tensor]:
        return [self.w, self.u, self.b]

    def grads(self) -> Iterable[Tensor]:
        return [self.w_grad, self.u_grad, self.b_grad]
