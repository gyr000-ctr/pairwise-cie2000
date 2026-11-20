"""cie2000 核心 API（初始未实现）。"""


class ColorDiff:
    def delta_e(self, lab1, lab2):
        """计算 CIEDE2000 色差"""
        raise NotImplementedError

    def delta_e76(self, lab1, lab2):
        """欧氏色差"""
        raise NotImplementedError
