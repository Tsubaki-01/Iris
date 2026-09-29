"""随示例附带的小型离线依赖；不从网络安装。"""


def total(values: list[int]) -> int:
    """计算整数总和，供两个 session 的共享服务复用。"""
    return sum(values)
