import math

import pytest

import cie2000

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

W = (100.0, 0.0, 0.0)
B = (0.0, 0.0, 0.0)


def _c():
    return cie2000.ColorDiff()


def _sample():
    return (50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485)

CASES = [
        ("same", lambda: _c().delta_e(W, W) == 0.0),
        ("black_white", lambda: _c().delta_e(W, B) > 90),
        ("symmetry", lambda: abs(_c().delta_e(W, B) - _c().delta_e(B, W)) < 1e-6),
        ("positive", lambda: _c().delta_e((50, 0, 0), (60, 0, 0)) > 0),
        ("e76_zero", lambda: _c().delta_e76(W, W) == 0.0),
        ("e76_100", lambda: abs(_c().delta_e76(W, B) - 100.0) < 1e-9),
        ("e76_positive", lambda: _c().delta_e76(W, B) > 0),
        ("known_pair", lambda: abs(_c().delta_e(*_sample()) - 2.0425) < 1e-3),
        ("nan", lambda: _raises(lambda: _c().delta_e((float("nan"), 0, 0), W))),
        ("inf", lambda: _raises(lambda: _c().delta_e((float("inf"), 0, 0), W))),
        ("bad_len", lambda: _raises(lambda: _c().delta_e((1, 2), W))),
        ("deterministic", lambda: _c().delta_e(*_sample()) == _c().delta_e(*_sample())),
        ("float", lambda: isinstance(_c().delta_e(W, B), float)),
        ("two_diff", lambda: _c().delta_e((50, 0, 0), (52, 0, 0)) < _c().delta_e((50, 0, 0), (60, 0, 0))),
        ("e76_sym", lambda: abs(_c().delta_e76(W, B) - _c().delta_e76(B, W)) < 1e-9),
    ]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
