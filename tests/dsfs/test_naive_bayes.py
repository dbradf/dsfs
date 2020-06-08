import math

import dsfs.naive_bayes as under_test

messages = [
    under_test.Message("spam rules", is_spam=True),
    under_test.Message("ham rules", is_spam=False),
    under_test.Message("hello ham", is_spam=False),
]


def test_training():
    model = under_test.NaiveBayesClassifier(k=0.5)
    model.train(messages)

    assert model.tokens == {"spam", "ham", "rules", "hello"}
    assert model.spam_messages == 1
    assert model.ham_messages == 2
    assert model.token_spam_counts == {"spam": 1, "rules": 1}
    assert model.token_ham_counts == {"ham": 2, "rules": 1, "hello": 1}


def test_prediction():
    model = under_test.NaiveBayesClassifier(k=0.5)
    model.train(messages)

    text = "hello spam"

    probs_if_spam = [
        (1 + 0.5) / (1 + 2 * 0.5),
        1 - (0 + 0.5) / (1 + 2 * 0.5),
        1 - (1 + 0.5) / (1 + 2 * 0.5),
        (0 + 0.5) / (1 + 2 * 0.5),
    ]

    probs_if_ham = [
        (0 + 0.5) / (2 + 2 * 0.5),
        1 - (2 + 0.5) / (2 + 2 * 0.5),
        1 - (1 + 0.5) / (2 + 2 * 0.5),
        (1 + 0.5) / (2 + 2 * 0.5),
    ]

    p_if_spam = math.exp(sum(math.log(p) for p in probs_if_spam))
    p_if_ham = math.exp(sum(math.log(p) for p in probs_if_ham))

    assert model.predict(text) == (p_if_spam / (p_if_spam + p_if_ham))
