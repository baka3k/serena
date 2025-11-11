from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from serena.lst.adapters import LosslessSemanticTreeAdapter, LosslessSymbolContext
from solidlsp.ls_config import Language
from solidlsp.ls_utils import InvalidTextLocationError, TextUtils


@dataclass
class _VariableDeclaration:
    name: str
    type: str
    initializer: str | None = None


class CppLosslessSemanticTreeAdapter(LosslessSemanticTreeAdapter):
    """
    Adapter that emits an OpenRewrite-style LST for C++ symbols.

    NOTE: The current implementation targets function/method symbols and uses heuristics for variable declarations
    and expressions. It is intentionally structured as an adapter so that future language backends (or improved C++
    parsing via libclang) can be plugged in without changing tool semantics.
    """

    language = Language.CPP

    _MODIFIERS = {"inline", "virtual", "static", "constexpr", "explicit", "friend"}

    def build(
        self,
        context: LosslessSymbolContext,
        file_content: str,
        include_source_text: bool,
        max_depth: int,
    ) -> dict[str, Any]:
        range_info = context.range
        start_line = range_info["start"]["line"]
        start_char = range_info["start"]["character"]
        end_line = range_info["end"]["line"]
        end_char = range_info["end"]["character"]

        try:
            start_idx = TextUtils.get_index_from_line_col(file_content, start_line, start_char)
            end_idx = TextUtils.get_index_from_line_col(file_content, end_line, end_char)
        except InvalidTextLocationError as ex:
            raise ValueError(f"Invalid range for symbol {context.name_path}: {range_info}") from ex

        snippet = file_content[start_idx:end_idx]
        tokens = self._tokenize(snippet, start_line, start_char)
        signature, body = self._split_signature_and_body(snippet)
        signature_data = self._parse_signature(signature.strip())
        params = self._parse_parameters(signature_data.get("params", ""))
        method_node_id = "n_method"
        node_id_counter = Counter()
        nodes: list[dict[str, Any]] = []
        bindings: list[dict[str, str]] = []
        method_edges: dict[str, Any] = {}

        param_nodes = []
        symbol_table: dict[str, str] = {}
        for param in params:
            node_id = f"n_param_{param['name']}"
            symbol_table[param["name"]] = node_id
            param_nodes.append(
                {
                    "id": node_id,
                    "kind": "Parameter",
                    "name": param["name"],
                    "type": param["type"],
                }
            )
        if param_nodes:
            method_edges["params"] = [node["id"] for node in param_nodes]
            nodes.extend(param_nodes)

        body_nodes, body_bindings = self._build_body_nodes(
            body,
            symbol_table=symbol_table,
            node_id_counter=node_id_counter,
        )
        bindings.extend(body_bindings)
        if body_nodes:
            method_edges["body"] = [node["id"] for node in body_nodes]
            nodes.extend(body_nodes)

        method_node = {
            "id": method_node_id,
            "kind": "Method",
            "name": signature_data["name"],
            "return_type": signature_data["return_type"],
            "modifiers": signature_data["modifiers"],
            "tokens_range": [1, len(tokens)],
            "edges": method_edges,
        }
        nodes.insert(0, method_node)

        params_types = [param["type"] for param in params]
        fqn = f"{signature_data['qualified_name']}({','.join(params_types)})"

        result = {
            "version": "1.0",
            "target": {"kind": context.kind, "fqn": fqn},
            "source": {
                "file": context.relative_path,
                "span": {
                    "start": [start_line + 1, start_char + 1],
                    "end": [end_line + 1, end_char + 1],
                },
            },
            "tokens": tokens,
            "nodes": nodes,
            "bindings": bindings,
            "meta": {
                "byte_perfect": include_source_text,
                "language": context.language.value,
                "type_system": "serena-cpp-heuristic",
                "origin": context.relative_path,
            },
        }

        if include_source_text:
            result["source"]["text"] = snippet

        return result

    def _split_signature_and_body(self, snippet: str) -> tuple[str, str]:
        brace_idx = snippet.find("{")
        if brace_idx == -1:
            raise ValueError("Unable to split signature/body: missing '{'")
        signature = snippet[:brace_idx].strip()
        body_candidate = snippet[brace_idx:]

        depth = 0
        end_idx = None
        for i, ch in enumerate(body_candidate):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end_idx = i
                    break
        if end_idx is None:
            raise ValueError("Unable to split signature/body: missing closing '}'")
        body = body_candidate[1:end_idx].strip()
        return signature, body

    def _parse_signature(self, signature: str) -> dict[str, Any]:
        pattern = re.compile(
            r"^(?P<prefix>[\w\s:\*&<>~]+?)\s+(?P<name>[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)\s*\((?P<params>.*)\)$",
            re.DOTALL,
        )
        match = pattern.match(signature.replace("\n", " ").strip())
        if not match:
            raise ValueError(f"Unable to parse C++ signature: {signature}")
        prefix = match.group("prefix").strip()
        name = match.group("name").strip()
        params = match.group("params").strip()
        prefix_parts = prefix.split()
        modifiers = [p for p in prefix_parts if p in self._MODIFIERS]
        return_type_parts = [p for p in prefix_parts if p not in modifiers]
        return_type = " ".join(return_type_parts) if return_type_parts else "auto"
        qualified_name = name
        short_name = name.split("::")[-1]
        return {
            "modifiers": modifiers,
            "return_type": return_type,
            "name": short_name,
            "qualified_name": qualified_name,
            "params": params,
        }

    def _parse_parameters(self, params: str) -> list[dict[str, str]]:
        if not params or params.strip() in {"", "void"}:
            return []
        result = []
        for raw_param in [p.strip() for p in params.split(",") if p.strip()]:
            match = re.match(r"(?P<type>.+)\s+(?P<name>[A-Za-z_]\w*)$", raw_param)
            if match:
                result.append({"type": match.group("type").strip(), "name": match.group("name").strip()})
            else:
                # fallback: treat entire string as type
                result.append({"type": raw_param, "name": f"param_{len(result)+1}"})
        return result

    def _build_body_nodes(
        self,
        body: str,
        symbol_table: dict[str, str],
        node_id_counter: Counter[str],
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        nodes: list[dict[str, Any]] = []
        bindings: list[dict[str, str]] = []
        body_edges: list[str] = []
        variable_decls = self._parse_variable_declarations(body)
        for decl in variable_decls:
            decl_id = f"n_decl_{decl.name}"
            symbol_table[decl.name] = decl_id
            node = {
                "id": decl_id,
                "kind": "VariableDecl",
                "name": decl.name,
                "type": decl.type.strip(),
            }
            if decl.initializer:
                expr_id, expr_nodes, expr_bindings = self._build_expression_graph(
                    decl.initializer,
                    symbol_table,
                    node_id_counter,
                )
                node["initializer"] = expr_id
                nodes.extend(expr_nodes)
                bindings.extend(expr_bindings)
            nodes.append(node)
            body_edges.append(decl_id)

        return_expr = self._parse_return_expression(body)
        if return_expr is not None:
            expr_id, expr_nodes, expr_bindings = self._build_expression_graph(
                return_expr,
                symbol_table,
                node_id_counter,
            )
            return_node_id = "n_return"
            nodes.extend(expr_nodes)
            bindings.extend(expr_bindings)
            nodes.append({"id": return_node_id, "kind": "Return", "edges": {"expr": expr_id}})
            body_edges.append(return_node_id)

        if nodes:
            # attach body ordering via synthetic block node
            block_id = "n_body_block"
            nodes.insert(
                0,
                {
                    "id": block_id,
                    "kind": "Block",
                    "edges": {"statements": body_edges},
                },
            )

        fallback_refs = []
        for name, decl_id in symbol_table.items():
            if not decl_id.startswith("n_param_"):
                continue
            if not re.search(rf"\b{name}\b", body):
                continue
            ref_id = self._next_node_id(f"n_ref_{name}", node_id_counter)
            nodes.append({"id": ref_id, "kind": "VariableRef", "name": name, "decl": decl_id})
            bindings.append({"from": ref_id, "to": decl_id})
            fallback_refs.append(ref_id)

        if fallback_refs:
            block_node = next((n for n in nodes if n["id"] == "n_body_block"), None)
            if block_node is not None:
                block_node.setdefault("edges", {}).setdefault("ref_usage", []).extend(fallback_refs)

        return nodes, bindings

    def _parse_variable_declarations(self, body: str) -> list[_VariableDeclaration]:
        pattern = re.compile(
            r"(?P<type>[A-Za-z_][\w:\s\*&<>]*)\s+(?P<name>[A-Za-z_]\w*)\s*(?:=\s*(?P<initializer>[^;]+))?;",
            re.MULTILINE,
        )
        result: list[_VariableDeclaration] = []
        for match in pattern.finditer(body):
            initializer = match.group("initializer")
            result.append(
                _VariableDeclaration(
                    name=match.group("name"),
                    type=match.group("type"),
                    initializer=initializer.strip() if initializer else None,
                )
            )
        return result

    def _parse_return_expression(self, body: str) -> str | None:
        match = re.search(r"return\s+(?P<expr>[^;]+);", body)
        if match:
            return match.group("expr").strip()
        return None

    def _build_expression_graph(
        self,
        expr: str,
        symbol_table: dict[str, str],
        node_id_counter: Counter[str],
    ) -> tuple[str, list[dict[str, Any]], list[dict[str, str]]]:
        expr = expr.strip()
        nodes: list[dict[str, Any]] = []
        bindings: list[dict[str, str]] = []

        # simple binary expression detection
        for operator in ["+", "-", "*", "/"]:
            parts = self._smart_split(expr, operator)
            if parts and len(parts) == 2:
                lhs_id, lhs_nodes, lhs_bindings = self._build_expression_graph(parts[0], symbol_table, node_id_counter)
                rhs_id, rhs_nodes, rhs_bindings = self._build_expression_graph(parts[1], symbol_table, node_id_counter)
                nodes.extend(lhs_nodes)
                nodes.extend(rhs_nodes)
                bindings.extend(lhs_bindings)
                bindings.extend(rhs_bindings)
                node_id = self._next_node_id("n_expr", node_id_counter)
                nodes.append(
                    {
                        "id": node_id,
                        "kind": "BinaryExpr",
                        "operator": operator,
                        "edges": {"lhs": lhs_id, "rhs": rhs_id},
                    }
                )
                return node_id, nodes, bindings

        # identifier reference
        identifier = expr.strip()
        if identifier in symbol_table:
            decl_id = symbol_table[identifier]
            ref_id = self._next_node_id(f"n_ref_{identifier}", node_id_counter)
            nodes.append({"id": ref_id, "kind": "VariableRef", "name": identifier, "decl": decl_id})
            bindings.append({"from": ref_id, "to": decl_id})
            return ref_id, nodes, bindings

        # literal fallback
        literal_id = self._next_node_id("n_literal", node_id_counter)
        nodes.append({"id": literal_id, "kind": "Literal", "value": identifier})
        return literal_id, nodes, bindings

    def _tokenize(self, snippet: str, start_line: int, start_char: int) -> list[dict[str, Any]]:
        tokens: list[dict[str, Any]] = []
        line = start_line
        col = start_char
        i = 0
        token_id = 1
        snippet_len = len(snippet)
        while i < snippet_len:
            ch = snippet[i]
            if ch == "\n":
                line += 1
                col = 0
                i += 1
                continue
            if ch.isspace():
                col += 1
                i += 1
                continue
            if ch == "/" and i + 1 < snippet_len and snippet[i + 1] == "/":
                end = snippet.find("\n", i)
                if end == -1:
                    end = snippet_len
                lexeme = snippet[i:end]
                tokens.append({"id": token_id, "lexeme": lexeme.strip(), "pos": [line + 1, col + 1], "trivia": True})
                token_id += 1
                col += len(lexeme)
                i = end
                continue
            if ch.isalpha() or ch == "_" or ch.isdigit():
                start_col = col
                start_idx = i
                while i < snippet_len and (snippet[i].isalnum() or snippet[i] in {"_", ":"}):
                    i += 1
                    col += 1
                lexeme = snippet[start_idx:i]
                tokens.append({"id": token_id, "lexeme": lexeme, "pos": [line + 1, start_col + 1]})
                token_id += 1
                continue
            two_char = snippet[i : i + 2]
            if two_char in {"::", "&&", "||", "==", "!=", "<=", ">=", "->"}:
                tokens.append({"id": token_id, "lexeme": two_char, "pos": [line + 1, col + 1]})
                token_id += 1
                i += 2
                col += 2
                continue
            tokens.append({"id": token_id, "lexeme": ch, "pos": [line + 1, col + 1]})
            token_id += 1
            i += 1
            col += 1
        return tokens

    def _smart_split(self, expr: str, operator: str) -> list[str] | None:
        depth = 0
        split_index = None
        for idx, ch in enumerate(expr):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == operator and depth == 0:
                split_index = idx
                break
        if split_index is None:
            return None
        lhs = expr[:split_index].strip()
        rhs = expr[split_index + 1 :].strip()
        if lhs and rhs:
            return [lhs, rhs]
        return None

    def _next_node_id(self, prefix: str, counter: Counter[str]) -> str:
        counter[prefix] += 1
        return f"{prefix}_{counter[prefix]}"
