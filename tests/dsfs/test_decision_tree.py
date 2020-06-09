import dsfs.decision_tree as under_test

from typing import NamedTuple, Optional


class Candidate(NamedTuple):
    level: str
    lang: str
    tweets: bool
    phd: bool
    did_well: Optional[bool] = None  # allow unlabeled data


inputs = [
    Candidate("Senior", "Java", False, False, False),
    Candidate("Senior", "Java", False, True, False),
    Candidate("Mid", "Python", False, False, True),
    Candidate("Junior", "Python", False, False, True),
    Candidate("Junior", "R", True, False, True),
    Candidate("Junior", "R", True, True, False),
    Candidate("Mid", "R", True, True, True),
    Candidate("Senior", "Python", False, False, False),
    Candidate("Senior", "R", True, False, True),
    Candidate("Junior", "Python", True, False, True),
    Candidate("Senior", "Python", True, True, True),
    Candidate("Mid", "Python", False, True, True),
    Candidate("Mid", "Java", True, False, True),
    Candidate("Junior", "Python", False, True, False),
]


def test_entropy():
    assert under_test.entropy([1.0]) == 0
    assert under_test.entropy([0.5, 0.5]) == 1
    assert 0.81 < under_test.entropy([0.25, 0.75]) < 0.82


def test_data_entropy():
    assert under_test.data_entropy(["a"]) == 0
    assert under_test.data_entropy([True, False]) == 1
    assert under_test.data_entropy([3, 4, 4, 4]) == under_test.entropy([0.25, 0.75])


def test_build_tree_id3():
    tree = under_test.build_tree_id3(inputs, ["level", "lang", "tweets", "phd"], "did_well")
    assert under_test.classify(tree, Candidate("Junior", "Java", True, False))
    assert not under_test.classify(tree, Candidate("Junior", "Java", True, True))
