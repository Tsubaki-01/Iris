# Performance Observations

本目录保存 Iris 各性能阶段可重复运行的测试侧观测。默认 pytest 只运行结构和语义测试；
带本机计时的场景必须显式传入：

~~~powershell
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"
uv run pytest -p no:cacheprovider --basetemp="$PWD\tmp\pytest-tmp" tests/performance --run-performance-timing -s
~~~

计时结果只用于同一机器、同一 fixture 的前后比较，不是跨机器 CI 阈值。每条
<code>PERF_OBSERVATION</code> 包含实际 Git HEAD、dirty 状态、Python、平台、fixture 规模、
原始毫秒样本和确定性结构计数；不得记录工具参数、文件内容、provider payload 或 secret。
