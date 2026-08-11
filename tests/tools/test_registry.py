from __future__ import annotations

from iris.tools import ToolRegistry


def test_function_registration_exports_schema_from_type_hints() -> None:
    def greet(name: str, excited: bool = False) -> str:
        """生成问候语。"""
        suffix = "!" if excited else "."
        return f"你好，{name}{suffix}"

    registry = ToolRegistry()
    registry.register_function(greet, description="生成问候语")

    assert registry.active_schemas() == [
        {
            "name": "greet",
            "description": "生成问候语",
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "excited": {"type": "boolean", "default": False},
                },
                "required": ["name"],
            },
        }
    ]
