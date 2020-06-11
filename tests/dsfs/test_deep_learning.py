import operator

import dsfs.deep_learning as under_test


def test_shape():
    assert under_test.shape([1, 2, 3]) == [3]
    assert under_test.shape([[1, 2], [3, 4], [5, 6]]) == [3, 2]


def test_is_1d():
    assert under_test.is_1d([1, 2, 3])
    assert not under_test.is_1d([[1, 2], [3, 4]])


def test_tensor_sum():
    assert under_test.tensor_sum([1, 2, 3]) == 6
    assert under_test.tensor_sum([[1, 2], [3, 4]]) == 10


def test_tensor_apply():
    assert under_test.tensor_apply(lambda x: x + 1, [1, 2, 3]) == [2, 3, 4]
    assert under_test.tensor_apply(lambda x: 2 * x, [[1, 2], [3, 4]]) == [[2, 4], [6, 8]]


def test_zeros_like():
    assert under_test.zeros_like([1, 2, 3]) == [0, 0, 0]
    assert under_test.zeros_like([[1, 2], [3, 4]]) == [[0, 0], [0, 0]]


def test_tensor_combine():
    assert under_test.tensor_combine(operator.add, [1, 2, 3], [4, 5, 6]) == [5, 7, 9]
    assert under_test.tensor_combine(operator.mul, [1, 2, 3], [4, 5, 6]) == [4, 10, 18]


def test_random_uniform():
    assert under_test.shape(under_test.random_uniform(2, 3, 4)) == [2, 3, 4]


def test_random_normal():
    assert under_test.shape(under_test.random_normal(5, 6, mean=10)) == [5, 6]
