"""延迟工具发现与本地搜索工具。

提供了对注册表中 deferred 工具的本地索引和基于 BM25 等机制的综合搜索支持。
"""

# region imports
from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel

from ..exceptions import IrisToolValidationError
from ..message import TextBlock
from .base import BaseTool, ToolDefinition, ToolExecutionContext, ToolResult

# endregion

# ==========================================
#                 Constants
# ==========================================
# region constants
_BM25_K1 = 1.5
_BM25_B = 0.75
_CJK_SINGLE_WEIGHT = 0.2
_SUBSTRING_MATCH_WEIGHT = 0.03
_QUERY_COVERAGE_WEIGHT = 0.35
_FIELD_WEIGHTS: dict[str, float] = {
    "name": 3.0,
    "tags": 2.0,
    "group": 1.5,
    "description": 1.0,
}
# endregion


class ToolSearchInput(BaseModel):
    """tool_search 的输入参数模型。"""

    query: str
    include_groups: list[str] | None = None
    limit: int = 10

    def normalized_groups(self) -> set[str] | None:
        """返回去重后的组过滤集合。

        Returns:
            set[str] | None: 去重后的组名集合，如果没有提供过滤则返回 None。
        """
        if self.include_groups is None:
            return None
        return set(self.include_groups)


class DeferredToolIndex:
    """为 deferred 工具提供 BM25-like 本地搜索索引。

    在不进行昂贵语义检索的情况下，基于本地工具元数据实现较高召回率和准确率的匹配排序。
    """

    # ==========================================
    #               Attribute Initialization
    # ==========================================
    # region
    def __init__(self) -> None:
        """初始化空索引。"""
        self._definitions: list[ToolDefinition] = []
        self._documents: list[tuple[ToolDefinition, dict[str, float], float, str]] = (
            []
        )  # (definition, weighted_terms, weighted_document_length, haystack)

    # endregion

    # ==========================================
    #               Index Management
    # ==========================================
    # region
    def build(self, tools: Iterable[BaseTool]) -> None:
        """用工具集合重建索引。

        Args:
            tools (Iterable[BaseTool]): 注册表中的工具对象集合。
        """
        self._definitions = [
            tool.definition for tool in tools if tool.definition.deferred
        ]
        self._documents = []
        for definition in self._definitions:
            weighted_terms = _weighted_terms(definition)
            document_length = sum(weighted_terms.values())
            if document_length > 0:
                self._documents.append(
                    (
                        definition,
                        weighted_terms,
                        document_length,
                        _search_text(definition),
                    )
                )

    # endregion

    # ==========================================
    #               Search Methods
    # ==========================================
    # region
    def search(
        self,
        query: str,
        *,
        include_groups: set[str] | None = None,
        limit: int = 10,
    ) -> list[ToolDefinition]:
        """用 BM25-like 相关性搜索 deferred 工具。

        算法基于本地工具元数据进行评分排序。使用了 BM25 的 IDF、词频饱和与文档长度归一化，
        避免长描述不成比例地抬高排名。

        Args:
            query (str): 非空搜索词。
            include_groups (set[str] | None): 可选的筛选组集合，仅在给定的组中搜索。
            limit (int): 最大返回的工具数量。

        Returns:
            list[ToolDefinition]: 按相关性从高到低排序后的候选工具定义列表。

        Raises:
            IrisToolValidationError: 当查询为空或无有效关键词时抛出。
        """
        # --- 1. 验证查询语句并筛选文档 ---
        normalized_query, query_terms, capped_limit = _validated_query(query, limit)
        filtered_docs = [
            doc
            for doc in self._documents
            if include_groups is None or doc[0].group in include_groups
        ]

        if not filtered_docs:
            return []

        # --- 2. 计算平均长度与文档频率 ---
        document_count = len(filtered_docs)
        average_length = sum(doc[2] for doc in filtered_docs) / document_count
        document_frequencies = {
            term: sum(1 for doc in filtered_docs if term in doc[1])
            for term in query_terms
        }

        # --- 3. 对所有筛选后的文档进行评分 ---
        scored: list[tuple[float, ToolDefinition]] = []
        for definition, weighted_terms, document_length, haystack in filtered_docs:
            score = _bm25_score(
                query_terms,
                weighted_terms,
                document_length=document_length,
                average_length=average_length,
                document_count=document_count,
                document_frequencies=document_frequencies,
            )
            # 完整子串命中时赋予固定权重补偿。
            if normalized_query in haystack:
                score += 0.25
            score += _substring_score(query_terms, haystack)
            score += _query_coverage_score(query_terms, weighted_terms, haystack)
            if score > 0:
                scored.append((score, definition))

        # --- 4. 排序并返回上限值 ---
        scored.sort(key=lambda item: (-item[0], item[1].name))
        return [definition for _, definition in scored[:capped_limit]]

    def naive_search(
        self,
        query: str,
        *,
        include_groups: set[str] | None = None,
        limit: int = 10,
    ) -> list[ToolDefinition]:
        """按字符子串硬匹配搜索 deferred 工具。

        这是 stage 4 最初的轻量搜索实现，保留给调试和对照测试使用。
        实际生产环境应当使用 search 方法。

        Args:
            query (str): 搜索词。
            include_groups (set[str] | None): 可选组过滤。
            limit (int): 最大返回数量。

        Returns:
            list[ToolDefinition]: 相关查询的候选定义集合。
        """
        normalized_query, query_terms, capped_limit = _validated_query(query, limit)
        scored: list[tuple[int, ToolDefinition]] = []
        for definition, _, _, haystack in self._documents:
            if include_groups is not None and definition.group not in include_groups:
                continue
            score = sum(1 for term in query_terms if term in haystack)
            if normalized_query in haystack:
                score += 2
            if score > 0:
                scored.append((score, definition))
        scored.sort(key=lambda item: (-item[0], item[1].name))
        return [definition for _, definition in scored[:capped_limit]]

    # endregion


class ToolSearchTool(BaseTool):
    """搜索注册表中默认隐藏的 deferred 工具的系统内置工具。"""

    # ==========================================
    #               Initialization
    # ==========================================
    # region
    def __init__(self, registry: Any) -> None:
        """初始化 tool_search。

        Args:
            registry (Any): 提供 search_deferred 方法的工具注册表实例。
        """
        self.registry = registry
        self.definition = ToolDefinition(
            name="tool_search",
            description="搜索可按需激活的延迟工具。",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词。"},
                    "include_groups": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "可选工具组过滤。",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 10,
                        "description": "最大返回数量。",
                    },
                },
                "required": ["query"],
            },
            group="core",
            metadata={
                "examples": [{"input": {"query": "file search"}}],
                "tags": ["discovery"],
                "version": "1.0",
                "deprecated": False,
                "deprecation_message": None,
            },
        )

    # endregion

    # ==========================================
    #               Properties
    # ==========================================
    # region
    @property
    def input_model(self) -> type[BaseModel] | None:
        """返回搜索输入模型类引用。

        Returns:
            type[BaseModel] | None: 指定的 ToolSearchInput 校验模型。
        """
        return ToolSearchInput

    # endregion

    # ==========================================
    #               Core Methods
    # ==========================================
    # region
    def validate_input(self, params: dict[str, Any]) -> BaseModel | dict[str, Any]:
        """校验 tool_search 命令输入参数。

        Args:
            params (dict[str, Any]): 用户或模型请求的原始参数。

        Returns:
            BaseModel | dict[str, Any]: 校验通过的参数模型或字典对象。

        Raises:
            IrisToolValidationError: 若基础类型校验不通过，或查询字段为空、
                限量非法等原因导致无法查询。
        """
        try:
            value = ToolSearchInput.model_validate(params)
        except ValueError as exc:
            raise IrisToolValidationError(
                "tool_search 参数校验失败", error=str(exc)
            ) from exc

        if not value.query.strip():
            raise IrisToolValidationError("tool_search query 不能为空")
        if value.limit < 1:
            raise IrisToolValidationError("tool_search limit 必须大于 0")

        return value

    async def arun(
        self,
        params: BaseModel | dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        """执行 deferred 工具的实际检索。

        Args:
            params (BaseModel | dict[str, Any]): 执行参数。
            context (ToolExecutionContext): 环境执行上下文。

        Returns:
            ToolResult: 包含检索出的候选工具摘要组成的成功负载结构。
        """
        del context
        if isinstance(params, ToolSearchInput):
            search_input = params
        else:
            search_input = ToolSearchInput.model_validate(params)

        matches = self.registry.search_deferred(
            search_input.query,
            include_groups=search_input.normalized_groups(),
            limit=search_input.limit,
        )
        tools = [_definition_summary(definition) for definition in matches]
        text = json.dumps({"tools": tools}, ensure_ascii=False, separators=(",", ":"))

        return ToolResult(
            tool_use_id="",
            tool_name=self.name,
            content=[TextBlock(text=text)],
            data={"tools": tools},
        )

    # endregion


def _definition_summary(definition: ToolDefinition) -> dict[str, Any]:
    """生成适合返回给大语言模型的简化工具摘要。

    Args:
        definition (ToolDefinition): 原始工具元定义。

    Returns:
        dict[str, Any]: 滤除无关细节后的紧凑 JSON 字典。
    """
    metadata = definition.metadata
    return {
        "name": definition.name,
        "description": definition.description,
        "group": definition.group,
        "tags": list(metadata.get("tags", [])),
        "deprecated": bool(metadata.get("deprecated", False)),
    }


def _validated_query(query: str, limit: int) -> tuple[str, dict[str, float], int]:
    """标准化并校验搜索参数。

    Args:
        query (str): 用户提供的原始搜索词。
        limit (int): 外部请求的目标数量。

    Returns:
        tuple[str, dict[str, float], int]: 格式化搜索词，各词汇的查询权重表，受保护的 limit。

    Raises:
        IrisToolValidationError: 如果最终词汇全为空则抛异常。
    """
    normalized_query = query.strip().lower()
    if not normalized_query:
        raise IrisToolValidationError("tool_search query 不能为空")
    query_terms = _token_weights(normalized_query)
    if not query_terms:
        raise IrisToolValidationError("tool_search query 必须包含可搜索文本")
    return normalized_query, query_terms, min(max(limit, 1), 50)


def _bm25_score(
    query_terms: dict[str, float],
    weighted_terms: dict[str, float],
    *,
    document_length: float,
    average_length: float,
    document_count: int,
    document_frequencies: dict[str, int],
) -> float:
    """计算单个文档的基础 BM25 相关性分数。

    基于经典的 BM25 公式构建基准分，并做了基础的 +0.5 平滑防止极端惩罚。

    Args:
        query_terms (dict[str, float]): 查询词及其权重频率。
        weighted_terms (dict[str, float]): 目标文档自身携带的带权频率分布。
        document_length (float): 目标文档全长积分。
        average_length (float): 集合中文档的平均长度。
        document_count (int): 过滤后文档集合总大小。
        document_frequencies (dict[str, int]): 各词条含有该词文档的数量映射。

    Returns:
        float: BM25 评分结果，越高代表相关性越强。
    """
    score = 0.0
    length_factor = 1 - _BM25_B + _BM25_B * (document_length / average_length)
    for term, query_weight in query_terms.items():
        term_frequency = weighted_terms.get(term, 0.0)
        if term_frequency <= 0:
            continue
        document_frequency = document_frequencies.get(term, 0)
        idf = math.log(
            1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
        )
        saturation = (term_frequency * (_BM25_K1 + 1)) / (
            term_frequency + _BM25_K1 * length_factor
        )
        score += idf * saturation * query_weight
    return score


def _substring_score(query_terms: dict[str, float], haystack: str) -> float:
    """为中文等未分词片段提供额外的子串加分策略。

    补偿因为简单的切词逻辑带来的召回黑洞问题。

    Args:
        query_terms (dict[str, float]): 待搜索词汇词典。
        haystack (str): 文档被索引的完整文本序列。

    Returns:
        float: 子串匹配加分累计。
    """
    return sum(
        _SUBSTRING_MATCH_WEIGHT * query_weight
        for term, query_weight in query_terms.items()
        if term in haystack
    )


def _query_coverage_score(
    query_terms: dict[str, float],
    weighted_terms: dict[str, float],
    haystack: str,
) -> float:
    """按查询词覆盖率给多条件匹配工具加分。

    核心指标：偏好多词匹配完整的文档，而非单词高频出现的文档。

    Args:
        query_terms (dict[str, float]): 待搜索的词及其权重。
        weighted_terms (dict[str, float]): 文档包含的带权词。
        haystack (str): 文档完整文本片段。

    Returns:
        float: 覆盖度综合加分。
    """
    primary_terms = [term for term, weight in query_terms.items() if weight >= 1.0]
    if not primary_terms:
        return 0.0
    matched = sum(
        1 for term in primary_terms if term in weighted_terms or term in haystack
    )
    return _QUERY_COVERAGE_WEIGHT * (matched / len(primary_terms))


def _weighted_terms(definition: ToolDefinition) -> dict[str, float]:
    """按工具元数据的重要度字段比例，展开生成带基础权值的词频表。

    Args:
        definition (ToolDefinition): 对应的工具结构体。

    Returns:
        dict[str, float]: 生成含有静态配置权重的分词分布典。
    """
    metadata = definition.metadata
    terms: dict[str, float] = {}
    _add_weighted_tokens(terms, definition.name, _FIELD_WEIGHTS["name"])
    _add_weighted_tokens(terms, definition.description, _FIELD_WEIGHTS["description"])
    _add_weighted_tokens(terms, definition.group, _FIELD_WEIGHTS["group"])
    _add_weighted_tokens(
        terms,
        " ".join(str(tag) for tag in metadata.get("tags", [])),
        _FIELD_WEIGHTS["tags"],
    )
    return terms


def _add_weighted_tokens(terms: dict[str, float], text: str, weight: float) -> None:
    """把原始文本执行分词并叠加特定字段常量比重后放入主词频表中。

    Args:
        terms (dict[str, float]): 需要被累加的主集合。
        text (str): 片段文案内容。
        weight (float): 提取字段的常识权重。
    """
    for token, token_weight in _token_weights(text).items():
        terms[token] = terms.get(token, 0.0) + weight * token_weight


def _tokenize(text: str) -> list[str]:
    """将工具搜索文本切分为小写基础 token。

    Args:
        text (str): 原始内容。

    Returns:
        list[str]: 格式化清洗后的小写字母与有效字符表。
    """
    return list(_token_weights(text))


def _token_weights(text: str) -> dict[str, float]:
    """将搜索原文通过规则解析生成附带初步分词权重的字典。

    Args:
        text (str): 候选原始搜索条件。

    Returns:
        dict[str, float]: 拆解带原始分布权重的映射。
    """
    normalized = text.replace("_", " ").replace("-", " ").lower()
    weights: dict[str, float] = {}
    for token in re.findall(r"[0-9a-z]+|[\u4e00-\u9fff]+", normalized):
        if _is_cjk_token(token):
            _add_cjk_token_weights(weights, token)
        else:
            weights[token] = weights.get(token, 0.0) + 1.0
    return weights


def _add_cjk_token_weights(weights: dict[str, float], token: str) -> None:
    """为 CJK 文本提取多维度变种（即支持整词、基于双字的 bigram 重组），兼顾不同的长短词命案。

    Args:
        weights (dict[str, float]): 输出聚合结果用的字典。
        token (str): 保证纯中文字符的片段。
    """
    if len(token) == 1:
        weights[token] = weights.get(token, 0.0) + _CJK_SINGLE_WEIGHT
        return
    if len(token) == 2:
        weights[token] = weights.get(token, 0.0) + 1.0
    else:
        weights[token] = weights.get(token, 0.0) + 1.0
        for index in range(len(token) - 1):
            bigram = token[index : index + 2]
            weights[bigram] = weights.get(bigram, 0.0) + 1.0

    # 对单独拆字作额外的弱召回容错。
    for char in token:
        weights[char] = weights.get(char, 0.0) + _CJK_SINGLE_WEIGHT


def _is_cjk_token(token: str) -> bool:
    """判断 token 是否为纯 CJK (中文字符) 文本。"""
    return all("\u4e00" <= char <= "\u9fff" for char in token)


def _search_text(definition: ToolDefinition) -> str:
    """聚合出完整代表工具本身的大文本序列，作为硬匹配靶本。

    Args:
        definition (ToolDefinition): 对应的工具定义结构对象。

    Returns:
        str: 拼接并转换为全小写的拼接长串结果。
    """
    metadata = definition.metadata
    tags = " ".join(str(tag) for tag in metadata.get("tags", []))
    return (
        f"{definition.name} {definition.description} {definition.group} {tags}".lower()
    )
