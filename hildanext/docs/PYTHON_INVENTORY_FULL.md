# Python Inventory — FULL (con vendor)
Scope: `hildanext/backend/src/hildanext`, `hildanext/backend/tests`, `hildanext/test`, `vendor/`.
Vendor included: yes.
Ultimo aggiornamento: 2026-02-28

## Modifiche 2026-02-28 rispetto al baseline 2026-02-22

| File | Modifica |
|---|---|
| `config.py` | + `ExperimentConfig` (ablation flags); `AppConfig` + campo `experiment` |
| `tokenization.py` | + `_pack_streaming()`, `_encode_records_batch()`; `tokenize_split` riscritto (streaming/checkpoint/resume/TRUNC_PROB) |
| `inference.py` | + `_EFFORT_PARAMS`, `_resolve_effort()`; engine + `effort=`; `last_stats` + `steps_to_converge/vram_peak_bytes/json_valid_rate` |
| `api.py` | `GenerateRequest` + `effort:str="medium"` |
| `diffusion.py` | `compute_m2t_t2t_losses` ritorna `masked_token_acc` |
| `training.py` | `_run` loga `masked_token_acc`, `json_valid_rate`; print + `mta=`; eval + `avg_steps_to_converge` |
| `wsd_stage0.py` | `prepare_dolma_only` checkpoint 1/2 resume; `_build_tokenized_artifacts` progress logging |
| `scripts/make_resume_ckpt.py` | NUOVO — bootstrap resume da run interrotta |
| `runs/configs/experiment_template.yaml` | NUOVO — template ablation YAML |

## Cartelle
- `backend\src\hildanext`: 22 file Python
- `backend\tests`: 1 file Python
- `test`: 28 file Python
- `vendor\llada`: 1014 file Python
- `vendor\dinfer`: 50 file Python

## File e Funzioni
### `backend\src\hildanext`
- File: `backend\src\hildanext\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `backend\src\hildanext\api.py`
  Logica d'uso: FastAPI serving layer with /health,/generate,/jobs endpoints.
  Funzioni:
  - `create_app(cfg:AppConfig) -> FastAPI`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
  - `run_server(config_path:str, host:str='127.0.0.1', port:int=8080) -> None`
    Descrizione: Esegue pipeline o job completo.
  Classi:
  - `GenerateRequest`
  - `GenerateResponse`
  - `JobResponse`

- File: `backend\src\hildanext\ar.py`
  Logica d'uso: AR baseline generation for side-by-side behavior checks.
  Funzioni:
  - `generate_ar(cfg:AppConfig, prompt:str, max_new_tokens:int=64, seed:Optional[int]=None) -> Dict[str, Any]`
    Descrizione: Genera output testo o sequenze.

- File: `backend\src\hildanext\audit.py`
  Logica d'uso: Verifica corrispondenza formula paper↔implementazione; scrive report MD+JSON.
  Funzioni:
  - `collect_formula_impl() -> Dict` — mappa simbolo → {file, line, snippet}
  - `paper_map() -> Dict` — dizionario simboli paper (LLaDA/LLaDA2/LLaDA2.1)
  - `_safe_find_text(path, needle) -> bool`
  - `_find_test_refs(root, symbol) -> List[str]`
  - `_check_invariants(symbol) -> Tuple[bool, str]`
  - `_status(implemented, invariant_ok, keyword_found, test_found) -> str`
  - `_render_md(rows, summary) -> str`
  - `run_audit(root_dir, out_md, out_json) -> Dict`

- File: `backend\src\hildanext\benchmarks.py`
  Logica d'uso: Tiny evaluation harness for pipeline sanity.
  Funzioni:
  - `_load_tinystories_prompts(cfg:AppConfig, n:int) -> List[str]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_load_humaneval_prompts(cfg:AppConfig, n:int) -> List[Dict[str, Any]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `run_benchmarks(cfg:AppConfig, engine:Any, max_items:int=8) -> Dict[str, Any]`
    Descrizione: Esegue pipeline o job completo.

- File: `backend\src\hildanext\cli.py`
  Logica d'uso: User command layer orchestrating pipelines.
  Funzioni:
  - `_print(data:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `cmd_prepare_data(args:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `cmd_tokenize(args:Any) -> Any`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
  - `cmd_convert_wsd(args:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `cmd_sft(args:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `cmd_serve(args:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `cmd_smoke(args:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `cmd_generate(args:Any) -> Any`
    Descrizione: Genera output testo o sequenze.
  - `cmd_benchmark(args:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `cmd_merge_topk(args)`
  - `cmd_audit(args)` — lancia audit formula→impl e scrive report MD
  - `cmd_quant_bench(args)` — benchmark quantizzazione
  - `cmd_preflight(args)` — preflight WSD
  - `cmd_run_recipe(args)` — pipeline LLaDA2.1 recipe
  - `cmd_dinfer_smoke(args)` — smoke dInfer
  - `cmd_make_stage0_config(args)` — genera config Stage0
  - `cmd_dolma_manifest(args)` — scansiona raw Dolma
  - `cmd_dolma_prep(args)` — esegue data prep + tokenize
  - `cmd_dolma_verify(args)` — verifica artefatti tokenized
  - `cmd_run_wsd(args)` — lancia training WSD completo
  - `cmd_preflight_wsd(args)` — preflight dedicato WSD
  - `cmd_archive(args)` — archivia runs/
  - `build_parser() -> argparse.ArgumentParser`
  - `main()`

- File: `backend\src\hildanext\config.py`
  Logica d'uso: Config-driven runtime wiring, all commands and services read here first.
  Funzioni:
  - `_merge_dataclass(dc:Any, payload:Dict[str, Any]) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_expand(s:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `resolve_paths(cfg:AppConfig) -> AppConfig`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `to_dict(cfg:AppConfig) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `from_dict(payload:Dict[str, Any]) -> AppConfig`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `load_config(path:str | Path) -> AppConfig`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `save_config(cfg:AppConfig, path:str | Path) -> None`
    Descrizione: Serializza e salva output su disco.
  - `default_config(root:str | Path) -> AppConfig`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `clone_with_updates(cfg:AppConfig, updates:Dict[str, Any]) -> AppConfig`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `PathsConfig`
  - `DataConfig`
  - `LLaDA2Config`
  - `ModelConfig`
  - `WSDConfig`
  - `TrainConfig`
  - `Stage0Config`
  - `RecipeConfig`
  - `RemaskConfig`
  - `InferenceConfig`
  - `RuntimeConfig`
  - `ExperimentConfig` *(NUOVO 2026-02-28)*
  - `AppConfig`

- File: `backend\src\hildanext\datasets.py`
  Logica d'uso: Builds CPT/SFT datasets from local/raw sources with fallback-safe behavior.
  Funzioni:
  - `_pick_text(obj:Dict[str, Any]) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_records_from_jsonl(path:Path, source:str, max_samples:int) -> List[Dict[str, Any]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_records_from_json(path:Path, source:str, max_samples:int) -> List[Dict[str, Any]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_records_from_txt(path:Path, source:str, max_samples:int) -> List[Dict[str, Any]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_records_from_pretokenized_dir(path:Path, source:str, max_samples:int) -> List[Dict[str, Any]]`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
  - `load_local_records(path_like:str, source:str, max_samples:int) -> List[Dict[str, Any]]`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `_download_tinystories(max_samples:int) -> List[Dict[str, Any]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_download_dolma(max_samples:int) -> List[Dict[str, Any]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_synthetic_records(prefix:str, n:int) -> List[Dict[str, Any]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_to_sft_pairs(records:List[Dict[str, Any]], max_samples:int) -> List[Dict[str, Any]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_split(records:List[Dict[str, Any]], eval_ratio:float, seed:int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_build_humaneval_dummy(n:int=8) -> List[Dict[str, Any]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `prepare_data(cfg:AppConfig, download:bool=False, max_samples:int | None=None) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `backend\src\hildanext\diffusion.py`
  Logica d'uso: Implements WSD schedule and mixed M2T/T2T training objective.
  Funzioni:
  - `wsd_block(step:int, cfg:WSDConfig) -> WSDStep`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_pick_positions(base_mask:torch.Tensor, p:float) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_causal_loss(logits:torch.Tensor, labels:torch.Tensor) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_attn_for_model(mask, model) -> Tensor` — converte maschera 1D per il modello (Qwen vs LLaDA)
  - `_forward(model, input_ids, attn_1d, doc_ids, mask_mode, clean_ids, composite_block_size, trace, cfg)`
  - `_make_m2t_batch(input_ids, attn_mask, response_mask, mask_id, ratio) -> Tuple`
  - `t2t_corrupt_tokens(input_ids, attn_mask, response_mask, ratio, vocab_size) -> Tuple` — esposta per test
  - `_make_t2t_batch(input_ids, attn_mask, response_mask, ratio, vocab_size) -> Tuple`
  - `compute_m2t_t2t_losses(...) -> Dict` *(aggiornato 2026-02-28: include `masked_token_acc`)*
  - `apply_remask(tokens:torch.Tensor, confidence:torch.Tensor, mask_id:int, cfg:RemaskConfig) -> torch.Tensor`
    Descrizione: Gestisce logica di mascheratura/filtri di posizioni.
  Classi:
  - `WSDStep`

- File: `backend\src\hildanext\formulas.py`
  Logica d'uso: Paper-aligned formula helpers for M2T/WSD/Gamma-Delta checks.
  Funzioni:
  - `llada_m2t_loss(logits:torch.Tensor, target_ids:torch.Tensor, masked_pos:torch.Tensor) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `llada2_wsd_block(step:int, warmup_steps:int, stable_steps:int, decay_steps:int, start_block:int, max_block:int, end_block:int) -> Tuple[str, int]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `llada21_sets(tokens:torch.Tensor, pred_ids:torch.Tensor, confidence:torch.Tensor, mask_id:int, tau_mask:float, tau_edit:float) -> LLaDA21SetResult`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `llada21_apply(tokens:torch.Tensor, pred_ids:torch.Tensor, confidence:torch.Tensor, mask_id:int, tau_mask:float, tau_edit:float) -> Tuple[torch.Tensor, LLaDA21SetResult]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `LLaDA21SetResult`

- File: `backend\src\hildanext\inference.py`
  Logica d'uso: Builds dInfer/fallback engines and threshold-edit decode.
  Funzioni:
  Modulo-level:
  - `_EFFORT_PARAMS: Dict` *(NUOVO 2026-02-28)*
  Funzioni:
  - `_model_dir_ready(model_dir:str) -> bool`
  - `_infer_param_dtype(model) -> str`
  - `load_model_bundle(cfg, for_training=False, trace=None) -> ModelBundle`
  - `_resolve_effort(effort, cfg_steps, tau_mask, tau_edit) -> Tuple` *(NUOVO 2026-02-28)*
  - `mode_thresholds(cfg, mode, tau_mask, tau_edit) -> Tuple[float, float]`
  - `_predict_autoregressive_candidates(model, seq, prompt_len, max_new, mask_id) -> Tuple`
  - `_decode_text(tok, out_ids, mask_id, is_dummy) -> str`
  - `build_engine(cfg, trace=None) -> BaseEngine`
  Classi:
  - `ModelBundle`
  - `BaseEngine`
    Metodo: `__init__(self:Any, cfg:AppConfig) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, prompt:str, mode:str='S_MODE', tau_mask:float | None=None, tau_edit:float | None=None, max_new_tokens:int | None=None, seed:int | None=None) -> str`
    Descrizione: Genera output testo o sequenze.
    Metodo: `close(self:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `TransformersEngine`
    Metodo: `__init__(self:Any, cfg:AppConfig, fallback_reason:str='') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_decode(self:Any, prompt:str, mode:str, tau_mask:float | None, tau_edit:float | None, max_new_tokens:int | None, seed:int | None) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, prompt:str, mode:str='S_MODE', tau_mask:float | None=None, tau_edit:float | None=None, max_new_tokens:int | None=None, seed:int | None=None) -> str`
    Descrizione: Genera output testo o sequenze.
  - `DInferEngine`
    Metodo: `__init__(self:Any, cfg:AppConfig) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_init_server(self:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `close(self:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, prompt:str, mode:str='S_MODE', tau_mask:float | None=None, tau_edit:float | None=None, max_new_tokens:int | None=None, seed:int | None=None) -> str`
    Descrizione: Genera output testo o sequenze.

- File: `backend\src\hildanext\io.py`
  Logica d'uso: Thin IO alias module.
  Nessuna funzione/classe top-level.

- File: `backend\src\hildanext\io_utils.py`
  Logica d'uso: Low-level file IO utilities.
  Funzioni:
  - `ensure_dir(path:str | Path) -> Path`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `read_json(path:str | Path) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `write_json(path:str | Path, data:Any) -> None`
    Descrizione: Serializza e salva output su disco.
  - `read_jsonl(path:str | Path, max_rows:int | None=None) -> List[Dict[str, Any]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `write_jsonl(path:str | Path, rows:Iterable[Dict[str, Any]]) -> int`
    Descrizione: Serializza e salva output su disco.
  - `append_jsonl(path:str | Path, rows:Iterable[Dict[str, Any]]) -> int`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `now_iso() -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `backend\src\hildanext\masks.py`
  Logica d'uso: Creates doc-boundary-safe attention masks used in train/inference.
  Funzioni:
  - `doc_attention_mask(doc_ids:torch.Tensor, causal:bool=False) -> torch.Tensor`
    Descrizione: Gestisce logica di mascheratura/filtri di posizioni.
  - `batch_doc_attention_mask(doc_ids:torch.Tensor, causal:bool=False) -> torch.Tensor`
    Descrizione: Gestisce logica di mascheratura/filtri di posizioni.
  - `response_focus_mask(response_mask:torch.Tensor, base_mask:torch.Tensor) -> torch.Tensor`
    Descrizione: Gestisce logica di mascheratura/filtri di posizioni.

- File: `backend\src\hildanext\quant.py`
  Logica d'uso: Benchmark quantizzazione fp16/int8/nf4 su modello reale.
  Classi:
  - `QuantLoadResult`
  Funzioni:
  - `_safe_vocab(tokenizer) -> int`
  - `_can_use_bnb(device) -> Tuple[bool, str]`
  - `_fallback_tiny(tokenizer, device, reason, mode, trace=None) -> QuantLoadResult`
  - `load_quantized(model_dir, mode, device, trust_remote_code, trace, cfg) -> Tuple`
  - `_decode_tokens(tokenizer, ids, mask_id) -> str`
  - `_run_ar_once(model, tokenizer, prompt, max_new_tokens, device, seed) -> Tuple`
  - `_run_dllm_once(model, tokenizer, prompt, max_new_tokens, device, mask_id, tau_mask, tau_edit, steps, seed) -> Tuple`
  - `_run_train_probe(model, tokenizer, prompt, device, seq_len=64) -> Dict`
  - `_finite_or_none(x) -> float|None`
  - `run_quant_bench(cfg, modes, prompt, max_new_tokens, ...) -> Dict`

- File: `backend\src\hildanext\recipe.py`
  Logica d'uso: Pipeline recipe completa LLaDA2.1: preflight, warmup, stable, decay, dinfer smoke.
  Funzioni:
  - `_disk_free_gb(path) -> float`
  - `_safe_ratio(text) -> float`
  - `_ensure_sft_from_tiny(cfg, trace=None) -> Dict`
  - `_run_stage_with_watchdog(cfg, kind, steps, trace=None) -> Tuple`
  - `preflight(cfg, trace=None) -> Dict`
  - `run_recipe_llada21(cfg, trace=None) -> Dict`
  - `dinfer_smoke(cfg, trace=None) -> Dict`

- File: `backend\src\hildanext\smoke.py`
  Logica d'uso: End-to-end smoke validation across load/train/infer.
  Funzioni:
  - `run_smoke(config_path:str) -> Dict[str, Any]`
    Descrizione: Esegue pipeline o job completo.

- File: `backend\src\hildanext\tokenization.py`
  Logica d'uso: Tokenizes and packs sequences, emits doc_ids used by attention masks.
  Funzioni:
  - `load_tokenizer(model_dir:str, trust_remote_code:bool=True) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `ensure_mask_token(tokenizer:Any, mask_token:str, model:Any=None) -> int`
    Descrizione: Gestisce logica di mascheratura/filtri di posizioni.
  - `_ids_within_vocab(ids:List[int], vocab_size:int) -> List[int]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_encode_text(tokenizer:Any, text:str) -> List[int]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_encode_record(tokenizer:Any, record:Dict[str, Any], vocab_size:int) -> Tuple[List[int], List[int], str]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_pack(encoded, seq_len, pad_id, eos_id) -> List[Dict]`
  - `_pack_streaming(encoded, seq_len, pad_id, eos_id, carry_ids, carry_docs, carry_resp, carry_src, doc_offset, trunc_prob=0.0) -> Tuple` *(NUOVO 2026-02-28 — streaming carry-over + random-length truncation)*
  - `_encode_records_batch(tokenizer, records, vocab_size) -> List[Tuple]` *(NUOVO 2026-02-28 — batch Rust 5-8x speedup)*
  - `tokenize_split(cfg, input_path, output_path, max_records=None, trace=None) -> Dict` *(riscritto 2026-02-28)*
  - `tokenize_all(cfg, max_records=None, trace=None) -> Dict`

- File: `backend\src\hildanext\trace.py`
  Logica d'uso: RunTrace: registra eventi, fallback, stats per ogni run; contestual-var thread-local.
  Classi:
  - `RunTrace` — metodi: `event()`, `fallback()`, `fail()`, `stats()`, `to_dict()`, `save()`, `log_path`
  Funzioni:
  - `_now() -> str`
  - `_jsonable(x) -> Any`
  - `ensure_run_id(run_id=None) -> str`
  - `config_digest_from_cfg(cfg) -> str`
  - `set_active_trace(trace)`
  - `reset_active_trace(token)`
  - `active_trace() -> Optional[RunTrace]`
  - `use_trace(cfg=None, trace=None) -> Optional[RunTrace]`
  - `trace_from_cfg(cfg, run_id=None) -> RunTrace`
  - `exception_with_stack(err) -> str`

- File: `backend\src\hildanext\training.py`
  Logica d'uso: Runs conversion and SFT loops, logs metrics and checkpoints.
  Funzioni:
  - `_collate(batch) -> Dict[str, Tensor]`
  - `_save_checkpoint(model, tokenizer, out_dir, tag) -> str`
  - `_prune_checkpoints(ckpt_dir, keep_last) -> None`
  - `_parse_step_from_tag(name) -> int`
  - `_latest_checkpoint(ckpt_dir) -> Tuple[int, Optional[Path]]`
  - `_load_checkpoint_model(cfg, model, ckpt_path, device, trace=None)`
  - `_make_optimizer(model, cfg, device, trace=None)`
  - `_decode_safe(tokenizer, ids, mask_id, dummy=False) -> str`
  - `_periodic_eval(model, tokenizer, device, mask_id, cfg, seed) -> Dict` *(calcola avg_steps_to_converge)*
  - `_run(cfg, split_name, kind, steps, focus_response, trace, resume, ckpt_every, eval_every) -> Dict` *(aggiornato 2026-02-28: loga masked_token_acc, json_valid_rate)*
  - `run_wsd_conversion(cfg, steps=None, trace=None, resume=False, ...) -> Dict`
  - `run_sft_training(cfg, steps=None, trace=None, resume=False, ...) -> Dict`
  - `merge_topk_checkpoints(cfg, checkpoint_dirs, output_dir) -> Dict`
  Classi:
  - `TokenizedDataset`
    Metodo: `__init__(self:Any, path:str, max_rows:int | None=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__len__(self:Any) -> int`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__getitem__(self:Any, i:int) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `backend\src\hildanext\utils.py`
  Logica d'uso: Runtime helpers, device/dtype control, tiny fallback model/tokenizer.
  Funzioni:
  - `seed_everything(seed:int) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `force_math_sdpa() -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `choose_device(device_hint:str='auto') -> torch.device`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `dtype_from_name(name:str, device:torch.device) -> torch.dtype`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `mem_stats(device:torch.device) -> Dict[str, float]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `tokens_per_second(token_count:int, elapsed:float) -> float`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
  - `chunked(seq:List[int], size:int) -> Iterable[List[int]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `SimpleTokenizer`
    Metodo: `__init__(self:Any, vocab_size:int=32768) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__len__(self:Any) -> int`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_vocab(self:Any) -> Dict[str, int]`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `convert_tokens_to_ids(self:Any, token:str) -> int`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
    Metodo: `add_special_tokens(self:Any, payload:Dict[str, List[str] | str]) -> int`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
    Metodo: `encode(self:Any, text:str, add_special_tokens:bool=False) -> List[int]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__call__(self:Any, texts:List[str] | str, return_tensors:str | None=None) -> Dict[str, torch.Tensor | List[List[int]]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `decode(self:Any, ids:List[int] | torch.Tensor, skip_special_tokens:bool=True) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `TinyCausalLM`
    Metodo: `__init__(self:Any, vocab_size:int=32768, hidden_size:int=256) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, input_ids:torch.Tensor, attention_mask:torch.Tensor | None=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `backend\src\hildanext\wsd_stage0.py`
  Logica d'uso: Pipeline Stage0 Dolma: raw→processed→tokenized→artifacts + WSD training loop.
  Funzioni:
  - `_count_bytes(p) -> int`
  - `_infer_ext(name) -> str`
  - `_inspect_path(p) -> Dict`
  - `_candidate_paths(cfg) -> List[Path]`
  - `dolma_manifest(cfg, trace=None) -> Dict`
  - `_extract_text(obj) -> str`
  - `_iter_json_lines(fh, max_docs, seen) -> Iterator`
  - `stream_docs(path, max_docs=None) -> Iterator`
  - `_inspect_existing_doc_index(root) -> Dict`
  - `_resolve_external_doc_index_dir() -> Path`
  - `_iter_external_doc_rows(doc_dir, target_seq_len, max_rows) -> Iterator`
  - `_apply_external_doc_index(train_tok, eval_tok, doc_dir, seq_len) -> Dict`
  - `_segment_text(text, max_words=120) -> List[str]`
  - `_split_processed(cfg, source_path, max_docs, eval_pct, seed) -> Dict`
  - `_build_tokenized_artifacts(cfg, train_tok_path, eval_tok_path, shard_rows=1000) -> Dict` *(aggiornato 2026-02-28: progress logging)*
  - `_dolma_fingerprint(cfg, manifest) -> Dict`
  - `_check_doc_boundary_signal(tokenized_train) -> Dict`
  - `_check_no_leakage(tokenized_train) -> Dict`
  - `_verify_artifacts(cfg) -> Dict`
  - `_ensure_llada21_objective(cfg)`
  - `_select_optimizer_name() -> str`
  - `_apply_stage0_to_cfg(cfg, run_id=None) -> AppConfig`
  - `prepare_dolma_only(cfg, trace=None) -> Dict` *(aggiornato 2026-02-28: checkpoint 1/2 resume)*
  - `verify_dolma_only(cfg, trace=None) -> Dict`
  - `preflight_wsd(cfg, trace=None) -> Dict`
  - `archive_runs(cfg, trace=None) -> Dict`
  - `run_wsd(cfg, config_path, trace=None, skip_dolma_prep=False) -> Dict`
  - `create_stage0_config(cfg, path, dolma_path) -> AppConfig`

### `backend\tests`
- File: `backend\tests\test_smoke.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `test_required_commands_present() -> Any`
    Descrizione: Test automatico di regressione/comportamento.

### `test`
- File: `test\__init__.py` — package marker
- File: `test\build_backend_readiness.py` — genera report readiness build
- File: `test\build_inventory.py` — genera questo inventory (AST scan)
- File: `test\reporting.py` — helpers per report test
- File: `test\run_tests.py` — unittest runner unificato
- File: `test\test_api_generate_real_model.py` — API /generate con modello reale
- File: `test\test_ar.py` — AR path produce output
- File: `test\test_audit_report.py` — audit formula↔impl
- File: `test\test_doc_mask_no_leakage_stronger.py` — doc boundary masking no-leakage
- File: `test\test_fallback_tracing_mandatory.py` — fallback trace obbligatorio
- File: `test\test_formulas.py` — formula LLaDA/LLaDA2/LLaDA2.1
- File: `test\test_infer_parity_dinfer.py` — parità output dInfer vs Transformers
- File: `test\test_inference_threshold_decode_invariants.py` — invarianti decode loop
- File: `test\test_llada20_composite_mask_doc_gating.py` — composite mask gating doc
- File: `test\test_llada20_composite_mask_structure.py` — struttura composite mask
- File: `test\test_masks.py` — document boundary masking
- File: `test\test_model_load_ar_real.py` — caricamento modello reale AR
- File: `test\test_precision.py` — fp16/fp32 e validità numerica
- File: `test\test_pre_sft_sanity.py` — sanity pre-SFT
- File: `test\test_quant_vram_sanity.py` — VRAM sanity quantizzazione
- File: `test\test_real_dllm_decode_non_degenerate.py` — decode dLLM non-degenere su modello reale
- File: `test\test_remask_invariants.py` — invarianti remask
- File: `test\test_sft_smoke.py` — one-step SFT smoke
- File: `test\test_stage0_dolma.py` — Stage0 Dolma pipeline
- File: `test\test_t2t_corruption_and_recovery.py` — T2T corruption e recovery
- File: `test\test_tokenizer_mask_real.py` — mask token su tokenizer reale
- File: `test\test_vocab_mask.py` — vocab length e mask token consistency
- File: `test\test_wsd_ladder_and_divisibility.py` — WSD ladder e divisibility


### `vendor\llada`
1014 file Python - codice originale ML-GSAI/LLaDA, **non modificato**.
File principali: `app.py`, `chat.py`, `generate.py`, `eval_llada.py`, `eval_reverse.py`, `get_log_likelihood.py`.
Sottocartelle: `opencompass/`, `visualization/`.

### `vendor\dinfer`
~50 file Python - dInfer adapter, **non modificato**.
Richiede runtime vLLM/sglang non disponibile su Pascal sm_61; fallback automatico a `TransformersEngine`.
