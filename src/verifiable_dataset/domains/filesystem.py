"""In-memory filesystem with mkdir/write/read/ls/mv/rm.

State is a nested dict tree: directories map name -> node, where a node is
either {"type": "dir", "children": {...}} or {"type": "file", "content": str}.
"""
from __future__ import annotations

import copy

from verifiable_dataset.base import BaseToolEnv, ToolCallError

TOOLS = [
    {
        "name": "mkdir",
        "description": "Create a directory at the given absolute path, creating parents as needed.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "write",
        "description": "Write (create or overwrite) a file at the given path with content.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    },
    {
        "name": "read",
        "description": "Read the content of a file at the given path.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "ls",
        "description": "List the names of entries in a directory.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "mv",
        "description": "Move/rename a file or directory from src to dst.",
        "parameters": {
            "type": "object",
            "properties": {"src": {"type": "string"}, "dst": {"type": "string"}},
            "required": ["src", "dst"],
            "additionalProperties": False,
        },
    },
    {
        "name": "rm",
        "description": "Remove a file or (empty or non-empty) directory at path.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
]


def _parts(path: str) -> list[str]:
    return [p for p in path.strip("/").split("/") if p]


class FilesystemEnv(BaseToolEnv):
    def __init__(self, initial_state: dict):
        self.root = copy.deepcopy(initial_state.get("root", {"type": "dir", "children": {}}))

    def _get_parent_and_name(self, path: str, create_parents: bool = False):
        parts = _parts(path)
        if not parts:
            raise ToolCallError("path must not be empty")
        node = self.root
        for part in parts[:-1]:
            children = node.setdefault("children", {}) if create_parents else node.get("children", {})
            if part not in children:
                if create_parents:
                    children[part] = {"type": "dir", "children": {}}
                else:
                    raise ToolCallError(f"no such directory: {part}")
            node = children[part]
            if node["type"] != "dir":
                raise ToolCallError(f"not a directory: {part}")
        return node, parts[-1]

    def mkdir(self, path: str) -> None:
        parent, name = self._get_parent_and_name(path, create_parents=True)
        children = parent.setdefault("children", {})
        if name not in children:
            children[name] = {"type": "dir", "children": {}}
        elif children[name]["type"] != "dir":
            raise ToolCallError(f"path exists and is not a directory: {path}")

    def write(self, path: str, content: str) -> None:
        parent, name = self._get_parent_and_name(path, create_parents=True)
        children = parent.setdefault("children", {})
        if name in children and children[name]["type"] == "dir":
            raise ToolCallError(f"path is a directory: {path}")
        children[name] = {"type": "file", "content": content}

    def read(self, path: str) -> str:
        parent, name = self._get_parent_and_name(path)
        node = parent.get("children", {}).get(name)
        if node is None or node["type"] != "file":
            raise ToolCallError(f"no such file: {path}")
        return node["content"]

    def ls(self, path: str) -> list[str]:
        parts = _parts(path)
        node = self.root
        for part in parts:
            node = node.get("children", {}).get(part)
            if node is None or node["type"] != "dir":
                raise ToolCallError(f"no such directory: {path}")
        return sorted(node.get("children", {}).keys())

    def mv(self, src: str, dst: str) -> None:
        src_parent, src_name = self._get_parent_and_name(src)
        if src_name not in src_parent.get("children", {}):
            raise ToolCallError(f"no such path: {src}")
        node = src_parent["children"].pop(src_name)
        dst_parent, dst_name = self._get_parent_and_name(dst, create_parents=True)
        dst_parent.setdefault("children", {})[dst_name] = node

    def rm(self, path: str) -> None:
        parent, name = self._get_parent_and_name(path)
        if name not in parent.get("children", {}):
            raise ToolCallError(f"no such path: {path}")
        del parent["children"][name]

    def state_dict(self) -> dict:
        return {"root": copy.deepcopy(self.root)}


def make_env(initial_state: dict) -> FilesystemEnv:
    return FilesystemEnv(initial_state)
