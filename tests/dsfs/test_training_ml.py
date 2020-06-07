import dsfs.training_ml as under_test


def test_train_test_split():
    xs = [x for x in range(1000)]
    ys = [2 * x for x in xs]

    x_train, x_test, y_train, y_test = under_test.train_test_split(xs, ys, 0.25)
    assert len(x_train) == len(y_train) == 750
    assert len(x_test) == len(y_test) == 250

    assert all(y == 2 * x for x, y in zip(x_train, y_train))
    assert all(y == 2 * x for x, y in zip(x_test, y_test))


def test_accuracy():
    assert under_test.accuracy(70, 4930, 13930, 981070) == 0.98114


def test_precision():
    assert under_test.precision(70, 4930, 13930, 981070) == 0.014


def test_recall():
    assert under_test.recall(70, 4930, 13930, 981070) == 0.005
