import dsfs.stats.stats as under_test

sample_list = [6, 1, 0, 34, 2, 6, 1, 3, 21, 5, 8, 13]
sample_list_2 = [1, 2, 1, 3, 2, 2, 1, 3, 2, 3, 8, 9]


def test_mean():
    assert under_test.mean([1, 2, 3, 4, 5, 6]) == 3.5


def test_median():
    assert under_test.median([1, 10, 2, 9, 5]) == 5
    assert under_test.median([1, 9, 2, 10]) == (2 + 9) / 2


def test_quantile():
    assert under_test.quantile(sample_list, 0.1) == 1
    assert under_test.quantile(sample_list, 0.25) == 2
    assert under_test.quantile(sample_list, 0.75) == 13
    assert under_test.quantile(sample_list, 0.90) == 21


def test_mode():
    assert set(under_test.mode(sample_list)) == {1, 6}


def test_data_range():
    assert under_test.data_range(sample_list) == 34


def test_variant():
    assert 100.77 < under_test.variance(sample_list) < 100.79


def test_standard_deviation():
    assert 10.02 < under_test.standard_deviation(sample_list) < 10.04


def test_interquartile_range():
    assert under_test.interquartile_range(sample_list) == 11


def test_covariance():
    assert 5.96 < under_test.covariance(sample_list, sample_list_2) < 5.97


def test_correlation():
    assert 0.22 < under_test.correlation(sample_list, sample_list_2) < 0.23
