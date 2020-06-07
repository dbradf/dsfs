import dsfs.linalg.matrix as under_test


def test_shape():
    assert under_test.shape([[1, 2, 3], [4, 5, 6]]) == (2, 3)


def test_identity():
    assert under_test.identity_matrix(5) == [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ]
