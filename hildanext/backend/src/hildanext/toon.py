from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import json
import math
import re

_NUMBERISH_RE=re.compile(r"^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$")
_BARE_KEY_RE=re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")


def _is_scalar(value:Any)->bool:
    return value is None or isinstance(value,(bool,int,float,str))


def _escape_string(text:str)->str:
    return (
        text.replace("\\","\\\\")
            .replace('"','\\"')
            .replace("\n","\\n")
            .replace("\r","\\r")
            .replace("\t","\\t")
    )


def _needs_quotes(text:str,delimiter:str)->bool:
    if text=="":
        return True
    if text!=text.strip():
        return True
    if text in {"true","false","null","-"}:
        return True
    if text.startswith("-"):
        return True
    if _NUMBERISH_RE.match(text):
        return True
    for ch in [":",'"',"\\","[","]","{","}","\n","\r","\t",delimiter]:
        if ch in text:
            return True
    return False


def _encode_scalar(value:Any,delimiter:str=",")->str:
    if value is None:
        return "null"
    if isinstance(value,bool):
        return "true" if value else "false"
    if isinstance(value,int):
        return str(value)
    if isinstance(value,float):
        if not math.isfinite(value):
            return "null"
        text=f"{value:.12f}".rstrip("0").rstrip(".")
        return text if text and text!="-0" else "0"
    text=str(value)
    if _needs_quotes(text,delimiter):
        return f"\"{_escape_string(text)}\""
    return text


def _encode_key(key:Any)->str:
    text=str(key)
    if _BARE_KEY_RE.match(text):
        return text
    return _encode_scalar(text)


def _uniform_fields(items:list[Any])->list[str]|None:
    if not items or not all(isinstance(item,dict) for item in items):
        return None
    keys=[str(k) for k in items[0].keys()]
    if not keys:
        return []
    for item in items[1:]:
        if [str(k) for k in item.keys()]!=keys:
            return None
    for item in items:
        for key in keys:
            if not _is_scalar(item.get(key)):
                return None
    return keys


def _primitive_array_inline(items:Iterable[Any],delimiter:str=",")->str:
    return delimiter.join(_encode_scalar(item,delimiter) for item in items)


def _append_value(lines:list[str],key:str|None,value:Any,indent:int,delimiter:str=",")->None:
    pad="  "*indent
    if isinstance(value,dict):
        if key is not None:
            lines.append(f"{pad}{_encode_key(key)}:")
            pad="  "*(indent+1)
            indent+=1
        for child_key,child_value in value.items():
            _append_value(lines,str(child_key),child_value,indent,delimiter)
        return

    if isinstance(value,list):
        header=f"{pad}{_encode_key(key)}" if key is not None else pad
        fields=_uniform_fields(value)
        if fields is not None:
            field_str=delimiter.join(_encode_key(field) for field in fields)
            lines.append(f"{header}[{len(value)}]{{{field_str}}}:")
            for item in value:
                row=delimiter.join(_encode_scalar(item.get(field),delimiter) for field in fields)
                lines.append(f"{pad}  {row}")
            return
        if all(_is_scalar(item) for item in value):
            inline=_primitive_array_inline(value,delimiter)
            prefix=f"{header}[{len(value)}]:"
            if inline:
                lines.append(f"{prefix} {inline}")
            else:
                lines.append(prefix)
            return
        lines.append(f"{header}[{len(value)}]:")
        for item in value:
            if _is_scalar(item):
                lines.append(f"{pad}  - {_encode_scalar(item,delimiter)}")
            elif isinstance(item,dict):
                if len(item)==1:
                    child_key,next_value=next(iter(item.items()))
                    if _is_scalar(next_value):
                        lines.append(
                            f"{pad}  - {_encode_key(child_key)}: {_encode_scalar(next_value,delimiter)}"
                        )
                        continue
                lines.append(f"{pad}  -")
                _append_value(lines,None,item,indent+2,delimiter)
            elif isinstance(item,list):
                if all(_is_scalar(x) for x in item):
                    lines.append(
                        f"{pad}  - [{len(item)}]: {_primitive_array_inline(item,delimiter)}"
                    )
                else:
                    lines.append(f"{pad}  - [{len(item)}]:")
                    for child in item:
                        if _is_scalar(child):
                            lines.append(f"{pad}      - {_encode_scalar(child,delimiter)}")
                        else:
                            _append_value(lines,None,child,indent+3,delimiter)
            else:
                lines.append(f"{pad}  - {_encode_scalar(item,delimiter)}")
        return

    if key is None:
        lines.append(f"{pad}{_encode_scalar(value,delimiter)}")
    else:
        lines.append(f"{pad}{_encode_key(key)}: {_encode_scalar(value,delimiter)}")


def dumps_toon(data:Any,root_key:str|None=None,delimiter:str=",")->str:
    lines:list[str]=[]
    _append_value(lines,root_key,data,0,delimiter)
    return "\n".join(lines)+("\n" if lines else "")


def write_toon(path:str|Path,data:Any,root_key:str|None=None,delimiter:str=",")->Path:
    target=Path(path)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(dumps_toon(data,root_key=root_key,delimiter=delimiter),encoding="utf-8")
    return target

