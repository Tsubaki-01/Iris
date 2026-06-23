## prompt的主要组装逻辑

```python
def _sort_sections(
    sections: Sequence[PromptSection],
    *,
    layout: PromptLayout,
) -> list[PromptSection]:
    order_map = {kind: index for index, kind in enumerate(layout.section_order)}
    cache_order = {
        PromptCacheScope.STATIC: 0,
        PromptCacheScope.SESSION: 1,
        PromptCacheScope.TURN: 2,
    }

    def sort_key(indexed_section: tuple[int, PromptSection]) -> tuple[int, int, int, int]:
        original_index, section = indexed_section
        final_offset = (
            1
            if layout.final_constraints_last
            and section.kind == PromptSectionKind.FINAL_CONSTRAINTS
            else 0
        )
        cache_index = cache_order[section.cache_scope] if layout.stable_prefix_first else 0
        layout_index = order_map.get(section.kind, len(order_map))
        return (cache_index, final_offset, layout_index, original_index)

    return [section for _, section in sorted(enumerate(sections), key=sort_key)]
```

这个函数的作用是：把一组 `PromptSection` 按稳定规则排序，保证 prompt 结构 deterministic。

核心逻辑在 `sort_key()`：

```
return (cache_index, final_offset, layout_index, original_index)
```

Python 会按 tuple 从左到右排序，所以优先级是：

1. `cache_index`
2. `final_offset`
3. `layout_index`
4. `original_index`

含义如下。

`cache_index`：按 cache 稳定性排序

```
cache_order = {
    PromptCacheScope.STATIC: 0,
    PromptCacheScope.SESSION: 1,
    PromptCacheScope.TURN: 2,
}
```

如果 `layout.stable_prefix_first=True`，排序会先放：

```
STATIC -> SESSION -> TURN
```

这样稳定内容在前，动态内容在后，有利于解释和复用 stable prefix / session prefix / turn suffix。

如果 `stable_prefix_first=False`，所有 section 的 `cache_index` 都是 `0`，就不按 cache scope 分组。

`final_offset`：让最终约束靠后

```
final_offset = (
    1
    if layout.final_constraints_last
    and section.kind == PromptSectionKind.FINAL_CONSTRAINTS
    else 0
)
```

如果开启 `final_constraints_last`，`FINAL_CONSTRAINTS` 的 offset 是 `1`，其他 section 是 `0`。

因为 `final_offset` 排在 `layout_index` 前面，所以在同一个 cache 分组里，`FINAL_CONSTRAINTS` 会被放到普通 section 后面。

注意：它不是全局最后，而是受 `cache_index` 影响。当前设计里 `FINAL_CONSTRAINTS` 本身通常是 `TURN` scope，所以实际会靠近整体末尾。

`layout_index`：按配置里的 section 顺序排序

```
order_map = {kind: index for index, kind in enumerate(layout.section_order)}
layout_index = order_map.get(section.kind, len(order_map))
```

`layout.section_order` 定义了业务顺序，比如：

```
role_and_behavior
tool_policy
session_history
current_input
final_constraints
```

如果某个 section kind 不在配置里，就给它 `len(order_map)`，也就是排在已知类型后面。

`original_index`：稳定兜底

```
original_index, section = indexed_section
```

如果前面三个排序键都一样，就保留原始输入顺序。这样排序是稳定的，不会因为相同 key 导致不可预期重排。

整体例子：

假设输入顺序是：

```
current_input       TURN
role_and_behavior   STATIC
session_history     SESSION
final_constraints   TURN
tool_policy         STATIC
```

开启 `stable_prefix_first=True` 和 `final_constraints_last=True` 后，会变成：

```
role_and_behavior   STATIC
tool_policy         STATIC
session_history     SESSION
current_input       TURN
final_constraints   TURN
```

所以这个函数本质上是在做三件事：

1. 稳定前缀优先：`STATIC` / `SESSION` / `TURN`
2. 业务布局顺序可配置：按 `layout.section_order`
3. 最终约束尽量靠后：`FINAL_CONSTRAINTS` 放在同组末尾