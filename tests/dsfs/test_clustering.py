import dsfs.clustering as under_test


leaf1 = under_test.Leaf([10, 20])
leaf2 = under_test.Leaf([30, -15])
merged = under_test.Merged((leaf1, leaf2), order=1)


def test_num_differences():
    assert under_test.num_differences([1, 2, 3], [2, 1, 3]) == 2
    assert under_test.num_differences([1, 2], [1, 2]) == 0


def test_get_values():
    assert under_test.get_values(merged) == [[10, 20], [30, -15]]
