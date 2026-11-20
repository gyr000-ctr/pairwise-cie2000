"""CIEDE2000 色差新增验收测试（不改动既有 test_core.py）。

覆盖：同色归零、对称性、Sharma 34 组参考数据（≤1e-4）、可区分的错误
类型、批量 10 万对性能（≤2s）、多线程并发安全与无共享可变状态。
"""

import math
import random
import threading

import pytest

import cie2000
from cie2000 import (
    ColorDiff,
    ColorError,
    LightnessOutOfRangeError,
    NonFiniteColorError,
)

W = (100.0, 0.0, 0.0)
B = (0.0, 0.0, 0.0)

# Sharma, Wu, Dalal (2005) 官方 34 组补充测试数据，
# 转引自 python-colormath 的同名测试表。
C1 = [
    (50.0000, 2.6772, -79.7751), (50.0000, 3.1571, -77.2803),
    (50.0000, 2.8361, -74.0200), (50.0000, -1.3802, -84.2814),
    (50.0000, -1.1848, -84.8006), (50.0000, -0.9009, -85.5211),
    (50.0000, 0.0000, 0.0000), (50.0000, -1.0000, 2.0000),
    (50.0000, 2.4900, -0.0010), (50.0000, 2.4900, -0.0010),
    (50.0000, 2.4900, -0.0010), (50.0000, 2.4900, -0.0010),
    (50.0000, -0.0010, 2.4900), (50.0000, -0.0010, 2.4900),
    (50.0000, -0.0010, 2.4900), (50.0000, 2.5000, 0.0000),
    (50.0000, 2.5000, 0.0000), (50.0000, 2.5000, 0.0000),
    (50.0000, 2.5000, 0.0000), (50.0000, 2.5000, 0.0000),
    (50.0000, 2.5000, 0.0000), (50.0000, 2.5000, 0.0000),
    (50.0000, 2.5000, 0.0000), (50.0000, 2.5000, 0.0000),
    (60.2574, -34.0099, 36.2677), (63.0109, -31.0961, -5.8663),
    (61.2901, 3.7196, -5.3901), (35.0831, -44.1164, 3.7933),
    (22.7233, 20.0904, -46.6940), (36.4612, 47.8580, 18.3852),
    (90.8027, -2.0831, 1.4410), (90.9257, -0.5406, -0.9208),
    (6.7747, -0.2908, -2.4247), (2.0776, 0.0795, -1.1350),
]
C2 = [
    (50.0000, 0.0000, -82.7485), (50.0000, 0.0000, -82.7485),
    (50.0000, 0.0000, -82.7485), (50.0000, 0.0000, -82.7485),
    (50.0000, 0.0000, -82.7485), (50.0000, 0.0000, -82.7485),
    (50.0000, -1.0000, 2.0000), (50.0000, 0.0000, 0.0000),
    (50.0000, -2.4900, 0.0009), (50.0000, -2.4900, 0.0010),
    (50.0000, -2.4900, 0.0011), (50.0000, -2.4900, 0.0012),
    (50.0000, 0.0009, -2.4900), (50.0000, 0.0010, -2.4900),
    (50.0000, 0.0011, -2.4900), (50.0000, 0.0000, -2.5000),
    (73.0000, 25.0000, -18.0000), (61.0000, -5.0000, 29.0000),
    (56.0000, -27.0000, -3.0000), (58.0000, 24.0000, 15.0000),
    (50.0000, 3.1736, 0.5854), (50.0000, 3.2972, 0.0000),
    (50.0000, 1.8634, 0.5757), (50.0000, 3.2592, 0.3350),
    (60.4626, -34.1751, 39.4387), (62.8187, -29.7946, -4.0864),
    (61.4292, 2.2480, -4.9620), (35.0232, -40.0716, 1.5901),
    (23.0331, 14.9730, -42.5619), (36.2715, 50.5065, 21.2231),
    (91.1528, -1.6435, 0.0447), (88.6381, -0.8985, -0.7239),
    (5.8714, -0.0985, -2.2286), (0.9033, -0.0636, -0.5514),
]
EXPECTED = [
    2.0425, 2.8615, 3.4412, 1.0000, 1.0000, 1.0000,
    2.3669, 2.3669, 7.1792, 7.1792, 7.2195, 7.2195,
    4.8045, 4.8045, 4.7461, 4.3065, 27.1492, 22.8977,
    31.9030, 19.4535, 1.0000, 1.0000, 1.0000, 1.0000,
    1.2644, 1.2630, 1.8731, 1.8645, 2.0373, 1.4146,
    1.4441, 1.5381, 0.6377, 0.9082,
]


def _cd():
    return ColorDiff()


def _random_pairs(n, seed=1234):
    rng = random.Random(seed)
    return [
        (
            (rng.uniform(0.0, 100.0), rng.uniform(-128.0, 128.0),
             rng.uniform(-128.0, 128.0)),
            (rng.uniform(0.0, 100.0), rng.uniform(-128.0, 128.0),
             rng.uniform(-128.0, 128.0)),
        )
        for _ in range(n)
    ]


def test_delta_e_same_color_is_exact_zero():
    cd = _cd()
    for lab in (W, B, (50.0, 2.6772, -79.7751), (-0.0, 1e-300, -1e-300)):
        assert cd.delta_e(lab, lab) == 0.0
        assert isinstance(cd.delta_e(lab, lab), float)
    # 不同容器、同值也归零
    assert cd.delta_e((50.0, 2.0, -3.0), [50.0, 2.0, -3.0]) == 0.0


def test_delta_e_symmetry_within_1e_12():
    cd = _cd()
    for c1, c2 in zip(C1, C2):
        assert math.isfinite(cd.delta_e(c1, c2))
        assert abs(cd.delta_e(c1, c2) - cd.delta_e(c2, c1)) <= 1e-12
    for c1, c2 in _random_pairs(256):
        assert abs(cd.delta_e(c1, c2) - cd.delta_e(c2, c1)) <= 1e-12


def test_delta_e_matches_sharma_reference_within_1e_4():
    cd = _cd()
    assert len(C1) == len(C2) == len(EXPECTED) == 34
    worst = 0.0
    for c1, c2, expected in zip(C1, C2, EXPECTED):
        got = cd.delta_e(c1, c2)
        worst = max(worst, abs(got - expected))
        assert abs(got - expected) <= 1e-4
    assert worst < 1e-4


def test_delta_e_far_colors_give_large_values():
    cd = _cd()
    assert cd.delta_e(W, B) > 90.0
    assert cd.delta_e((50.0, 0.0, 0.0), (60.0, 0.0, 0.0)) > 0.0
    assert (cd.delta_e((50.0, 0.0, 0.0), (52.0, 0.0, 0.0))
            < cd.delta_e((50.0, 0.0, 0.0), (60.0, 0.0, 0.0)))


def test_delta_e_deterministic_and_float():
    cd = _cd()
    c1, c2 = C1[0], C2[0]
    first = cd.delta_e(c1, c2)
    for _ in range(20):
        assert cd.delta_e(c1, c2) == first
        assert isinstance(cd.delta_e(c1, c2), float)


def test_non_finite_input_raises_distinct_error_type():
    cd = _cd()
    bad_values = (float("nan"), float("inf"), float("-inf"))
    for bad in bad_values:
        for idx in range(3):
            lab = [50.0, 1.0, -2.0]
            lab[idx] = bad
            with pytest.raises(NonFiniteColorError):
                cd.delta_e(tuple(lab), W)
            with pytest.raises(NonFiniteColorError):
                cd.delta_e(W, tuple(lab))


def test_negative_lightness_raises_distinct_error_type():
    cd = _cd()
    with pytest.raises(LightnessOutOfRangeError):
        cd.delta_e((-0.0001, 0.0, 0.0), W)
    with pytest.raises(LightnessOutOfRangeError):
        cd.delta_e(W, (-1.0, 0.0, 0.0))


def test_error_types_are_distinguishable_value_errors():
    assert issubclass(NonFiniteColorError, ColorError)
    assert issubclass(LightnessOutOfRangeError, ColorError)
    assert issubclass(ColorError, ValueError)
    # 两类错误必须可区分
    assert NonFiniteColorError is not LightnessOutOfRangeError
    cd = _cd()
    nan_raised = light_raised = False
    try:
        cd.delta_e((float("nan"), 0.0, 0.0), W)
    except LightnessOutOfRangeError:
        pass
    except NonFiniteColorError:
        nan_raised = True
    try:
        cd.delta_e((-1.0, 0.0, 0.0), W)
    except NonFiniteColorError:
        pass
    except LightnessOutOfRangeError:
        light_raised = True
    assert nan_raised and light_raised
    # 非有限检查优先于亮度检查
    with pytest.raises(NonFiniteColorError):
        cd.delta_e((float("-inf"), 0.0, 0.0), W)


def test_bad_length_and_type_raise_value_error():
    cd = _cd()
    with pytest.raises(ValueError):
        cd.delta_e((1.0, 2.0), W)
    with pytest.raises((ValueError, TypeError)):
        cd.delta_e(42, W)
    with pytest.raises((ValueError, TypeError)):
        cd.delta_e((1.0, 2.0, 3.0, 4.0), W)


def test_delta_e76_behaviour_preserved():
    cd = _cd()
    assert cd.delta_e76(W, W) == 0.0
    assert abs(cd.delta_e76(W, B) - 100.0) < 1e-9
    assert cd.delta_e76(W, B) > 0.0
    assert abs(cd.delta_e76(W, B) - cd.delta_e76(B, W)) < 1e-9


def test_delta_e_batch_100k_under_2_seconds():
    cd = _cd()
    pairs = _random_pairs(100_000)
    # 预热，排除首次导入/分派抖动
    cd.delta_e_batch(pairs[:256])

    import time
    t0 = time.perf_counter()
    results = cd.delta_e_batch(pairs)
    elapsed = time.perf_counter() - t0

    assert len(results) == 100_000
    assert all(isinstance(v, float) and math.isfinite(v) for v in results)
    # 与逐对调用结果逐位一致
    assert results[0] == cd.delta_e(*pairs[0])
    assert results[-1] == cd.delta_e(*pairs[-1])
    assert elapsed < 2.0, f"批量 10 万对耗时 {elapsed:.3f}s 超过 2s"


def test_delta_e_batch_empty_and_ordering():
    cd = _cd()
    assert cd.delta_e_batch([]) == []
    pairs = list(zip(C1[:5], C2[:5]))
    assert cd.delta_e_batch(pairs) == [cd.delta_e(a, b) for a, b in pairs]


def test_concurrent_calls_share_no_mutable_state():
    cd = ColorDiff()
    pairs = _random_pairs(4_000)
    expected = [cd.delta_e(a, b) for a, b in pairs]
    errors = []
    results_per_thread = []

    def worker(slice_):
        try:
            local_cd = ColorDiff()
            # 混用共享实例与线程内实例
            out = []
            for i, (a, b) in enumerate(slice_):
                out.append(cd.delta_e(a, b) if i % 2 else local_cd.delta_e(a, b))
            results_per_thread.append(out)
        except Exception as exc:  # pragma: no cover - 并发下不应出错
            errors.append(exc)

    n_threads = 8
    chunks = [pairs[i::n_threads] for i in range(n_threads)]
    threads = [threading.Thread(target=worker, args=(chunk,)) for chunk in chunks]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    merged = [0.0] * len(pairs)
    for i in range(n_threads):
        for j, value in enumerate(results_per_thread[i]):
            merged[i + j * n_threads] = value
    assert merged == expected


def test_instance_is_stateless():
    cd = ColorDiff()
    # 空 __slots__：实例不可也不持有任何属性
    assert ColorDiff.__slots__ == ()
    with pytest.raises(AttributeError):
        cd.something = 1
    with pytest.raises(AttributeError):
        cd.__dict__
