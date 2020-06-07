import dsfs.k_nearest_neighbors as under_test


def test_raw_majority_vote():
    assert under_test.raw_majority_vote(["a", "b", "c", "b"]) == "b"


def test_majority_vote():
    assert under_test.majority_vote(["a", "b", "c", "b", "a"]) == "b"
