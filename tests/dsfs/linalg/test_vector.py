import dsfs.linalg.vector as under_test


def test_add():
    assert under_test.add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]


def test_subtract():
    assert under_test.subtract([5, 7, 9], [4, 5, 6]) == [1, 2, 3]


def test_sum():
    assert under_test.vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]]) == [16, 20]


def test_scalar_multiply():
    assert under_test.scalar_multiply(2, [1, 2, 3]) == [2, 4, 6]


def test_vector_mean():
    assert under_test.vector_mean([[1, 2], [3, 4], [5, 6]]) == [3, 4]


def test_dot():
    assert under_test.dot([1, 2, 3], [4, 5, 6]) == 32


def test_sum_of_squares():
    assert under_test.sum_of_squares([1, 2, 3]) == 14


def test_magnitude():
    assert under_test.magnitude([3, 4]) == 5
