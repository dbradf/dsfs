import dsfs.network as under_test


def test_build_friendships():
    friendships = under_test.build_friendships(under_test.users)
    assert friendships[4] == [3, 5]
    assert friendships[8] == [6, 7, 9]
