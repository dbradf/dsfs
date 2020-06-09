import dsfs.multiple_regression as under_test

x = [1, 2, 3]
y = 30
beta = [4, 4, 4]


def test_error():
    assert under_test.error(x, y, beta) == -6


def test_squared_error():
    assert under_test.squared_error(x, y, beta) == 36


def test_sqerror_gradiant():
    assert under_test.sqerror_gradient(x, y, beta) == [-12, -24, -36]
