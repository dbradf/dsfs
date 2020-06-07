import dsfs.data_exploration as under_test

vectors = [[-3, -1, 1], [-1, 0, 1], [1, 1, 1]]


def test_scale():
    means, stdevs = under_test.scale(vectors)
    assert means == [-1, 0, 1]
    assert stdevs == [2, 1, 0]


def test_rescale():
    means, stdevs = under_test.scale(under_test.rescale(vectors))
    assert means == [0, 0, 1]
    assert stdevs == [1, 1, 0]
