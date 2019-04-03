import numpy as np

from spyky.encoding import kulkarni_rajendran


def test_kulkarni_rajendran():
    inpts = np.array([0, 1, 2, 3, 4, 5])
    encoding = kulkarni_rajendran(inpts)
    expected = [2700.0, 2801.2, 2902.4, 3003.6, 3104.8, 3206.0]
    assert all([a == b for a, b in zip(encoding, expected)])
