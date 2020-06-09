import dsfs.linear_regression as under_test


def test_least_squares_fit():
    x = [i for i in range(-100, 110, 10)]
    y = [3 * i - 5 for i in x]

    assert under_test.least_squares_fit(x, y) == (-5, 3)
