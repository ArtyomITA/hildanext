# Python inventory builder for ML engineer handoff.
# Entry point: python hildanext/test/build_inventory.py.
# Generates markdown with folders, files, signatures, I/O and usage logic.
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict,List,Tuple
import argparse
import ast

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"docs"/"PYTHON_INVENTORY.md"
SCAN_DIRS_DEFAULT=[ROOT/"backend"/"src"/"hildanext",ROOT/"backend"/"tests",ROOT/"test"]
SCAN_DIRS_VENDOR=[ROOT/"vendor"/"llada",ROOT/"vendor"/"dinfer"]

MODULE_LOGIC={
    "config.py":"Config-driven runtime wiring, all commands and services read here first.",
    "datasets.py":"Builds CPT/SFT datasets from local/raw sources with fallback-safe behavior.",
    "tokenization.py":"Tokenizes and packs sequences, emits doc_ids used by attention masks.",
    "masks.py":"Creates doc-boundary-safe attention masks used in train/inference.",
    "diffusion.py":"Implements WSD schedule and mixed M2T/T2T training objective.",
    "training.py":"Runs conversion and SFT loops, logs metrics and checkpoints.",
    "inference.py":"Builds dInfer/fallback engines and threshold-edit decode.",
    "api.py":"FastAPI serving layer with /health,/generate,/jobs endpoints.",
    "cli.py":"User command layer orchestrating pipelines.",
    "smoke.py":"End-to-end smoke validation across load/train/infer.",
    "formulas.py":"Paper-aligned formula helpers for M2T/WSD/Gamma-Delta checks.",
    "ar.py":"AR baseline generation for side-by-side behavior checks.",
    "benchmarks.py":"Tiny evaluation harness for pipeline sanity.",
    "utils.py":"Runtime helpers, device/dtype control, tiny fallback model/tokenizer.",
    "io.py":"Thin IO alias module.",
    "io_utils.py":"Low-level file IO utilities.",
    "test_precision.py":"Checks fp16/fp32 and numerical validity.",
    "test_vocab_mask.py":"Checks vocab length and mask token consistency.",
    "test_masks.py":"Checks document boundary masking behavior.",
    "test_ar.py":"Checks AR path produces output.",
    "test_sft_smoke.py":"Checks one-step SFT smoke path.",
    "test_formulas.py":"Checks LLaDA/LLaDA2/LLaDA2.1 formula behavior.",
    "run_tests.py":"Unified unittest runner.",
    "build_inventory.py":"Generates this inventory report."
}

def ann(node:ast.AST|None)->str:
    if node is None:
        return "Any"
    try:
        return ast.unparse(node)
    except Exception:
        return "Any"

def args_sig(n:ast.FunctionDef|ast.AsyncFunctionDef)->str:
    parts=[]
    args=n.args
    pos=args.posonlyargs+args.args
    defaults=[None]*(len(pos)-len(args.defaults))+list(args.defaults)
    for a,d in zip(pos,defaults):
        s=f"{a.arg}:{ann(a.annotation)}"
        if d is not None:
            try:
                s=f"{s}={ast.unparse(d)}"
            except Exception:
                s=f"{s}=..."
        parts.append(s)
    if args.vararg is not None:
        parts.append(f"*{args.vararg.arg}:{ann(args.vararg.annotation)}")
    for a,d in zip(args.kwonlyargs,args.kw_defaults):
        s=f"{a.arg}:{ann(a.annotation)}"
        if d is not None:
            try:
                s=f"{s}={ast.unparse(d)}"
            except Exception:
                s=f"{s}=..."
        parts.append(s)
    if args.kwarg is not None:
        parts.append(f"**{args.kwarg.arg}:{ann(args.kwarg.annotation)}")
    return ", ".join(parts)

def desc_from_name(name:str)->str:
    n=name.lower()
    if n.startswith("load"):
        return "Carica dati o stato da sorgente esterna/file."
    if n.startswith("save") or n.startswith("write"):
        return "Serializza e salva output su disco."
    if n.startswith("run"):
        return "Esegue pipeline o job completo."
    if n.startswith("build") or n.startswith("create"):
        return "Costruisce oggetto/struttura derivata dai parametri."
    if n.startswith("get"):
        return "Recupera valore/stato calcolato."
    if n.startswith("test_"):
        return "Test automatico di regressione/comportamento."
    if "mask" in n:
        return "Gestisce logica di mascheratura/filtri di posizioni."
    if "token" in n:
        return "Gestisce tokenizzazione o manipolazione token."
    if "generate" in n:
        return "Genera output testo o sequenze."
    if "train" in n:
        return "Esegue passaggi di training/update."
    return "Funzione di utilita' usata nel flusso SAFE."

def collect(scan_dirs:List[Path])->Dict[str,List[Path]]:
    out={}
    for d in scan_dirs:
        if not d.exists():
            continue
        files=sorted([p for p in d.rglob("*.py") if "__pycache__" not in str(p)])
        out[str(d.relative_to(ROOT))]=files
    return out

def parse_file(path:Path)->Tuple[List[Dict],List[Dict]]:
    tree=ast.parse(path.read_text(encoding="utf-8"))
    funcs=[]
    classes=[]
    for n in tree.body:
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):
            funcs.append({"name":n.name,"sig":args_sig(n),"ret":ann(n.returns),"desc":desc_from_name(n.name)})
        elif isinstance(n,ast.ClassDef):
            methods=[]
            for b in n.body:
                if isinstance(b,(ast.FunctionDef,ast.AsyncFunctionDef)):
                    methods.append({"name":b.name,"sig":args_sig(b),"ret":ann(b.returns),"desc":desc_from_name(b.name)})
            classes.append({"name":n.name,"methods":methods})
    return funcs,classes

def build_md(include_vendor:bool=False)->str:
    scan_dirs=list(SCAN_DIRS_DEFAULT)
    if include_vendor:
        scan_dirs.extend(SCAN_DIRS_VENDOR)
    files_by_dir=collect(scan_dirs)
    lines=[]
    lines.append("# Python Inventory")
    lines.append("Scope: `hildanext/backend/src/hildanext`, `hildanext/backend/tests`, `hildanext/test`.")
    lines.append(f"Vendor included: {'yes' if include_vendor else 'no'}.")
    lines.append("")
    lines.append("## Cartelle")
    for d,files in files_by_dir.items():
        lines.append(f"- `{d}`: {len(files)} file Python")
    lines.append("")
    lines.append("## File e Funzioni")
    for d,files in files_by_dir.items():
        lines.append(f"### `{d}`")
        for fp in files:
            rel=fp.relative_to(ROOT)
            logic=MODULE_LOGIC.get(fp.name,"Modulo di supporto nella pipeline SAFE.")
            lines.append(f"- File: `{rel}`")
            lines.append(f"  Logica d'uso: {logic}")
            funcs,classes=parse_file(fp)
            if funcs:
                lines.append("  Funzioni:")
                for f in funcs:
                    lines.append(f"  - `{f['name']}({f['sig']}) -> {f['ret']}`")
                    lines.append(f"    Descrizione: {f['desc']}")
            if classes:
                lines.append("  Classi:")
                for c in classes:
                    lines.append(f"  - `{c['name']}`")
                    for m in c["methods"]:
                        lines.append(f"    Metodo: `{m['name']}({m['sig']}) -> {m['ret']}`")
                        lines.append(f"    Descrizione: {m['desc']}")
            if not funcs and not classes:
                lines.append("  Nessuna funzione/classe top-level.")
            lines.append("")
    return "\n".join(lines)

def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--include-vendor",action="store_true")
    ap.add_argument("--output",default=str(OUT))
    args=ap.parse_args()
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(build_md(include_vendor=bool(args.include_vendor)),encoding="utf-8")
    print(str(out))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
def _legacy_main()->int:
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(build_md(include_vendor=False),encoding="utf-8")
    print(str(OUT))
    return 0
