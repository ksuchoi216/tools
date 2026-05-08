from typing import Any


def _get_example_items_by_depth(data: Any) -> dict[int, Any]:
    example_items: dict[int, Any] = {}

    def visit(current_node: Any, depth: int) -> None:
        if isinstance(current_node, list):
            for item in current_node:
                if depth not in example_items:
                    example_items[depth] = item
                if isinstance(item, dict):
                    visit(item, depth)
                elif isinstance(item, list):
                    visit(item, depth + 1)
            return

        if depth not in example_items:
            example_items[depth] = current_node

        if isinstance(current_node, dict):
            for value in current_node.values():
                if isinstance(value, (dict, list)):
                    visit(value, depth + 1)
                elif depth + 1 not in example_items:
                    example_items[depth + 1] = value
            return

    visit(data, 0)
    return example_items


def _print_example_items_by_depth(data: list | dict) -> None:
    print("Example items by depth:")
    for depth, example_item in sorted(_get_example_items_by_depth(data).items()):
        example_text = repr(example_item)
        if len(example_text) > 100:
            example_text = f"{example_text[:100]}..."
        print(f"depth {depth}: {example_text}")


def _get_deepest_level(data: Any) -> int:
    """
    Get the deepest level (depth) of nested structures in data.
    - Dicts and lists increase depth
    - Scalars (int, str, etc.) are at depth 0
    """
    if isinstance(data, dict):
        if not data:
            return 0
        return 1 + max(_get_deepest_level(v) for v in data.values())
    elif isinstance(data, list):
        if not data:
            return 0
        return 1 + max(_get_deepest_level(item) for item in data)
    else:
        return 0


def convert_data_to_text(
    data: list | dict,
    tab_num: int = 4,
    has_xml_tag: bool = False,
    has_numbering: bool = False,
    including_key: bool = True,
    depth_tag: dict | None = None,
    depth_numbering: dict | None = None,
    key_prefix: str = "",
    key_suffix: str = "",
    value_prefix: str = "",
    value_suffix: str = "",
    item_prefix: str = "",
    item_suffix: str = "",
    verbose: bool = False,
) -> str:
    """
    YAML에서 로드한 dict/list 객체를 LLM 입력용 구조화 텍스트로 변환한다.

    tab_num is the number of spaces used for each depth level. When has_xml_tag
    is True, dict/list data can be wrapped with XML tags using depth_tag.
    has_numbering adds an id attribute at depth 0 by default. depth_numbering
    maps depth levels to XML attribute names. including_key controls whether
    scalar values render as "key: value" or just "value". verbose prints one
    example item for each depth to help choose depth_tag and depth_numbering.
    The custom prefix/suffix options are only valid for level 1 data, such as a
    flat list of scalars or a dict with scalar values.
    """

    if not isinstance(data, (dict, list)):
        raise TypeError("convert_data_to_text only supports dict or list data")

    if not data:
        return ""

    if depth_tag and not has_xml_tag:
        raise ValueError("depth_tag requires has_xml_tag=True")

    if depth_numbering and not has_numbering:
        raise ValueError("depth_numbering requires has_numbering=True")

    if has_numbering and not has_xml_tag:
        raise ValueError("has_numbering=True requires has_xml_tag=True")

    if verbose:
        _print_example_items_by_depth(data)

    has_custom_format = any(
        [
            key_prefix,
            key_suffix,
            value_prefix,
            value_suffix,
            item_prefix,
            item_suffix,
        ]
    )
    if has_custom_format:
        if has_xml_tag or has_numbering or depth_tag or depth_numbering:
            raise ValueError(
                "custom prefix/suffix options cannot be combined with XML options"
            )

        deepest_level = _get_deepest_level(data)
        if deepest_level > 1:
            raise ValueError(
                "custom prefix/suffix options only support level 1 dict/list data"
            )

        lines: list[str] = []
        if isinstance(data, dict):
            for key, value in data.items():
                key_text = f"{key_prefix}{key}{key_suffix}"
                value_text = f"{value_prefix}{value}{value_suffix}"
                lines.append(f"{key_text}: {value_text}")
            return "\n".join(lines)

        for item in data:
            item_text = f"{item_prefix}{item}{item_suffix}"
            lines.append(item_text)
        return "\n".join(lines)

    depth_tag = dict(depth_tag or {})
    depth_numbering = dict(depth_numbering or {})
    if has_numbering and not depth_numbering:
        depth_numbering = {0: "id"}

    if has_xml_tag or has_numbering or depth_tag or depth_numbering:
        deepest_level = _get_deepest_level(data)
        max_depth = deepest_level - 1

        invalid_tag_keys = [k for k in depth_tag.keys() if k > max_depth]
        if invalid_tag_keys:
            raise ValueError(
                f"depth_tag keys {invalid_tag_keys} exceed maximum depth "
                f"{max_depth} (data deepest level: {deepest_level})"
            )

        invalid_numbering_keys = [k for k in depth_numbering.keys() if k > max_depth]
        if invalid_numbering_keys:
            raise ValueError(
                f"depth_numbering keys {invalid_numbering_keys} exceed maximum "
                f"depth {max_depth} (data deepest level: {deepest_level})"
            )

        lines: list[str] = []

        def get_numbering_attr(depth: int, item_index: int | None) -> str:
            if depth not in depth_numbering:
                return ""
            attr_name = depth_numbering[depth]
            attr_value = item_index if item_index is not None else 0
            return f" {attr_name}={attr_value}"

        def render_plain_xml_content(current_node: Any, depth: int) -> None:
            indent = " " * (depth * tab_num)

            if isinstance(current_node, dict):
                for key, value in current_node.items():
                    if isinstance(value, (dict, list)):
                        if depth_tag.get(depth + 1):
                            render_xml(value, depth + 1)
                            continue
                        if including_key:
                            lines.append(f"{indent}{key}:")
                        render_plain_xml_content(value, depth + 1)
                        continue

                    value_str = str(value).strip() if value is not None else ""
                    if value_str:
                        line = f"{key}: {value_str}" if including_key else value_str
                        lines.append(f"{indent}{line}")
                return

            if isinstance(current_node, list):
                for idx, item in enumerate(current_node):
                    before = len(lines)
                    render_xml(item, depth, idx)
                    if idx < len(current_node) - 1 and len(lines) > before:
                        lines.append("")
                return

            value_str = str(current_node).strip() if current_node is not None else ""
            if value_str:
                lines.append(f"{indent}{value_str}")

        def render_wrapped_node(
            tag_name: str,
            current_node: Any,
            depth: int,
            item_index: int | None,
        ) -> None:
            indent = " " * (depth * tab_num)
            numbering_attr = get_numbering_attr(depth, item_index)
            lines.append(f"{indent}<{tag_name}{numbering_attr}>")
            if isinstance(current_node, (dict, list)):
                render_xml(current_node, depth + 1, item_index)
            else:
                render_plain_xml_content(current_node, depth)
            lines.append(f"{indent}</{tag_name}>")

        def render_xml(
            current_node: Any,
            depth: int = 0,
            item_index: int | None = None,
        ) -> None:
            explicit_tag_name = depth_tag.get(depth)
            if explicit_tag_name:
                if isinstance(current_node, list):
                    for idx, item in enumerate(current_node):
                        render_wrapped_node(explicit_tag_name, item, depth, idx)
                    return

                render_wrapped_node(explicit_tag_name, current_node, depth, item_index)
                return

            indent = " " * (depth * tab_num)

            if isinstance(current_node, dict):
                items = list(current_node.items())
                if {"sentence", "context"}.issubset(current_node.keys()):
                    priority = {"sentence": 0, "context": 1}
                    items.sort(key=lambda item: priority.get(item[0], 2))

                for key, value in items:
                    numbering_attr = get_numbering_attr(depth, item_index)
                    lines.append(f"{indent}<{key}{numbering_attr}>")
                    if isinstance(value, (dict, list)):
                        render_plain_xml_content(value, depth + 1)
                    else:
                        value_str = str(value).strip() if value is not None else ""
                        if value_str:
                            line = f"{key}: {value_str}" if including_key else value_str
                            lines.append(f"{' ' * ((depth + 1) * tab_num)}{line}")
                    lines.append(f"{indent}</{key}>")
                return

            if isinstance(current_node, list):
                for idx, item in enumerate(current_node):
                    render_xml(item, depth, idx)
                return

            value_str = str(current_node).strip() if current_node is not None else ""
            if value_str:
                lines.append(f"{indent}{value_str}")

        render_xml(data)
        return "\n".join(lines)

    lines: list[str] = []

    def add_text(value: Any, depth: int, key: str | None = None) -> None:
        text = str(value).strip() if value is not None else ""
        if not text:
            return

        prefix = " " * (depth * tab_num)
        text_lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not text_lines:
            return

        if key and including_key:
            lines.append(f"{prefix}{key}: {text_lines[0]}")
        else:
            lines.append(f"{prefix}{text_lines[0]}")

        for line in text_lines[1:]:
            lines.append(f"{prefix}{line}")

    def render(
        current_node: Any,
        depth: int = 0,
        title: str | None = None,
    ) -> None:
        if title:
            prefix = " " * (depth * tab_num)
            lines.append(f"{prefix}{title}")
            depth += 1

        if isinstance(current_node, dict):
            for key, value in current_node.items():
                key_text = str(key).strip()
                if isinstance(value, (dict, list)):
                    render(value, depth, key_text or None)
                    continue

                add_text(value, depth, key_text or None)
            return

        if isinstance(current_node, list):
            for idx, item in enumerate(current_node):
                if isinstance(item, dict):
                    item_title = ""
                    item_title_key = ""
                    for candidate in ("name", "title", "label"):
                        candidate_value = item.get(candidate)
                        if candidate_value is None:
                            continue
                        item_title = str(candidate_value).strip()
                        if item_title:
                            item_title_key = candidate
                            break

                    if item_title_key:
                        child = {
                            key: value
                            for key, value in item.items()
                            if key != item_title_key
                        }
                        render(child, depth, item_title)
                    else:
                        before = len(lines)
                        render(item, depth)
                        if idx < len(current_node) - 1 and len(lines) > before:
                            lines.append("")
                    continue

                if isinstance(item, list):
                    before = len(lines)
                    render(item, depth)
                    if idx < len(current_node) - 1 and len(lines) > before:
                        lines.append("")
                    continue

                add_text(item, depth)
            return

        add_text(current_node, depth)

    render(data)
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)
