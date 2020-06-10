import dsfs.neural_networks as under_test


def test_xor_network():
    xor_network = [[[20.0, 20, -30], [20.0, 20, -10]], [[-60.0, 60, -30]]]

    assert 0.000 < under_test.feed_forward(xor_network, [0, 0])[-1][0] < 0.001
    assert 0.999 < under_test.feed_forward(xor_network, [1, 0])[-1][0] < 1.000
    assert 0.999 < under_test.feed_forward(xor_network, [0, 1])[-1][0] < 1.000
    assert 0.000 < under_test.feed_forward(xor_network, [1, 1])[-1][0] < 0.001
