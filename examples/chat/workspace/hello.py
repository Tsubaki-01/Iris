"""提供 chat 示例可读取的最小 Python 文件。

Example:
    >>> greet("Iris 使用者")
    '你好，Iris 使用者！'
"""


def greet(name: str) -> str:
    """返回面向使用者的中文问候语。

    Args:
        name (str): 要写入问候语的使用者名称。

    Returns:
        str: 包含使用者名称的中文问候语。
    """
    return f"你好，{name}！"


def main() -> None:
    """打印 chat 示例默认使用者的问候语。

    Returns:
        None: 此示例仅写入标准输出。
    """
    print(greet("Iris 使用者"))


if __name__ == "__main__":
    main()
