from collections import Counter

import dsfs.nlp as under_test


def test_sample_from():
    draws = Counter(under_test.sample_from([0.1, 0.1, 0.8]) for _ in range(1000))
    assert 10 < draws[0] < 190
    assert 10 < draws[1] < 190
    assert 650 < draws[2] < 950
    assert draws[0] + draws[1] + draws[2] == 1000


def test_cosine_similarity():
    assert under_test.cosine_similarity([1.0, 1, 1], [2.0, 2, 2]) == 1
    assert under_test.cosine_similarity([-1.0, -1], [2.0, 2]) == -1
    assert under_test.cosine_similarity([1.0, 0], [0.0, 1]) == 0


def test_vocabulary():
    vocab = under_test.Vocabulary(["a", "b", "c"])
    assert vocab.size == 3
    assert vocab.get_id("b") == 1
    assert vocab.one_hot_encode("b") == [0, 1, 0]
    assert vocab.get_id("z") is None
    assert vocab.get_word(2) == "c"

    vocab.add("z")
    assert vocab.size == 4
    assert vocab.get_id("z") == 3
    assert vocab.one_hot_encode("z") == [0, 0, 0, 1]
