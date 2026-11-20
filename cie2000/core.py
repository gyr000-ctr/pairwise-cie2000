"""CIEDE2000 色差核心 API。

只依赖 Python 标准库（``math``）。:class:`ColorDiff` 为无状态对象：
所有常量均为不可变类属性，方法不读写任何可变共享状态，因此可在多线程中
并发调用。
"""

import math

__all__ = [
    "ColorDiff",
    "ColorError",
    "NonFiniteColorError",
    "LightnessOutOfRangeError",
]

# 模块加载时绑定局部名，热路径上避免全局/属性查找开销。
_sqrt = math.sqrt
_cos = math.cos
_sin = math.sin
_atan2 = math.atan2
_hypot = math.hypot
_degrees = math.degrees
_radians = math.radians
_isfinite = math.isfinite
_exp = math.exp

_TAU = 2.0 * math.pi
_25_POW_7 = 6103515625.0  # 25.0 ** 7


class ColorError(ValueError):
    """色差计算输入错误的基类（同时是 :class:`ValueError`）。"""


class NonFiniteColorError(ColorError):
    """Lab 分量中存在 NaN 或无穷大等非有限值。"""


class LightnessOutOfRangeError(ColorError):
    """亮度 L* 为负（物理上不允许 L* < 0）。"""


def _validate_lab(lab):
    """校验单个 Lab 三元组，返回解包后的 (L, a, b)。

    长度/可迭代性问题抛 :class:`ValueError`/:class:`TypeError`；
    非有限值抛 :class:`NonFiniteColorError`；负亮度抛
    :class:`LightnessOutOfRangeError`，两类错误可区分且均为
    :class:`ValueError` 子类。
    """
    try:
        L, a, b = lab
    except ValueError:
        raise ValueError("Lab 颜色必须是含 3 个分量的序列 (L, a, b)")
    except TypeError:
        raise TypeError("Lab 颜色必须是含 3 个分量的序列 (L, a, b)")
    if not (_isfinite(L) and _isfinite(a) and _isfinite(b)):
        raise NonFiniteColorError("Lab 分量必须全部为有限实数，不能为 NaN 或无穷大")
    if L < 0.0:
        raise LightnessOutOfRangeError("亮度 L* 不能为负值")
    return L, a, b


class ColorDiff:
    """CIE Lab 色差计算器（无状态，线程安全）。

    ``__slots__`` 为空，实例不持有任何属性；方法仅使用不可变类常量与
    局部变量，多线程并发调用之间不存在共享可变状态。
    """

    __slots__ = ()

    # ---- CIEDE2000 公式中的不可变常量 ----
    _K_L = 1.0
    _K_C = 1.0
    _K_H = 1.0

    def delta_e(self, lab1, lab2):
        """计算 CIEDE2000 色差（CIE 142-2003 / Sharma 2005 公式）。

        参数为两个 Lab 三元组 ``(L*, a*, b*)``，返回 Python ``float``。
        同色输入精确返回 ``0.0``；结果对称（正反交换误差为 0）。

        异常：
            NonFiniteColorError: 任一分量为 NaN/inf。
            LightnessOutOfRangeError: 任一亮度 L* < 0。
            ValueError: 输入不是三分量序列。
        """
        L1, a1, b1 = _validate_lab(lab1)
        L2, a2, b2 = _validate_lab(lab2)

        # 同色快速路径，保证同色结果精确为 0.0（也省去整段公式计算）。
        if lab1 == lab2:
            return 0.0

        # ---- 第 1 步：计算 C'I、h'I ----
        C1 = _hypot(a1, b1)
        C2 = _hypot(a2, b2)
        C_bar = (C1 + C2) * 0.5
        C_bar7 = C_bar ** 7
        G = 0.5 * (1.0 - _sqrt(C_bar7 / (C_bar7 + _25_POW_7)))
        a1p = (1.0 + G) * a1
        a2p = (1.0 + G) * a2
        C1p = _hypot(a1p, b1)
        C2p = _hypot(a2p, b2)
        if b1 == 0.0 and a1p == 0.0:
            h1p = 0.0
        else:
            h1p = _degrees(_atan2(b1, a1p)) % 360.0
        if b2 == 0.0 and a2p == 0.0:
            h2p = 0.0
        else:
            h2p = _degrees(_atan2(b2, a2p)) % 360.0

        # ---- 第 2 步：计算 ΔL'、ΔC'、ΔH' ----
        dLp = L2 - L1
        dCp = C2p - C1p
        if C1p * C2p == 0.0:
            dhp = 0.0
        else:
            dhp = h2p - h1p
            if dhp > 180.0:
                dhp -= 360.0
            elif dhp < -180.0:
                dhp += 360.0
        dHp = 2.0 * _sqrt(C1p * C2p) * _sin(_radians(dhp) * 0.5)

        # ---- 第 3 步：计算 CIEDE2000 色差 ----
        L_barp = (L1 + L2) * 0.5
        C_barp = (C1p + C2p) * 0.5
        if C1p * C2p == 0.0:
            h_barp = h1p + h2p
        else:
            if abs(h1p - h2p) <= 180.0:
                h_barp = (h1p + h2p) * 0.5
            elif h1p + h2p < 360.0:
                h_barp = (h1p + h2p + 360.0) * 0.5
            else:
                h_barp = (h1p + h2p - 360.0) * 0.5

        T = (
            1.0
            - 0.17 * _cos(_radians(h_barp - 30.0))
            + 0.24 * _cos(_radians(2.0 * h_barp))
            + 0.32 * _cos(_radians(3.0 * h_barp + 6.0))
            - 0.20 * _cos(_radians(4.0 * h_barp - 63.0))
        )
        d_theta = 30.0 * _exp(-(((h_barp - 275.0) / 25.0) ** 2))
        C_barp7 = C_barp ** 7
        R_C = 2.0 * _sqrt(C_barp7 / (C_barp7 + _25_POW_7))
        L_term = L_barp - 50.0
        S_L = 1.0 + (0.015 * L_term ** 2) / _sqrt(20.0 + L_term ** 2)
        S_C = 1.0 + 0.045 * C_barp
        S_H = 1.0 + 0.015 * C_barp * T
        R_T = -_sin(_radians(2.0 * d_theta)) * R_C

        nL = dLp / (self._K_L * S_L)
        nC = dCp / (self._K_C * S_C)
        nH = dHp / (self._K_H * S_H)
        return _sqrt(nL * nL + nC * nC + nH * nH + R_T * nC * nH)

    def delta_e76(self, lab1, lab2):
        """CIE76 欧氏色差（Lab 空间中的欧氏距离）。"""
        L1, a1, b1 = _validate_lab(lab1)
        L2, a2, b2 = _validate_lab(lab2)
        dL = L2 - L1
        da = a2 - a1
        db = b2 - b1
        return _sqrt(dL * dL + da * da + db * db)

    def delta_e_batch(self, pairs):
        """批量计算 CIEDE2000 色差。

        ``pairs`` 为可迭代的 ``(lab1, lab2)`` 序列；逐对独立校验与计算，
        返回与输入等长、同序的 ``list[float]``。本方法同样无状态，可在
        多线程中对同一实例并发调用。
        """
        delta_e = self.delta_e
        return [delta_e(lab1, lab2) for lab1, lab2 in pairs]
