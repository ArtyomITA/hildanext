# Python Inventory
Scope: `hildanext/backend/src/hildanext`, `hildanext/backend/tests`, `hildanext/test`.
Vendor included: no.

## Cartelle
- `backend\src\hildanext`: 31 file Python
- `backend\tests`: 2 file Python
- `test`: 54 file Python

## File e Funzioni
### `backend\src\hildanext`
- File: `backend\src\hildanext\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `backend\src\hildanext\api.py`
  Logica d'uso: FastAPI serving layer with /health,/generate,/jobs endpoints.
  Funzioni:
  - `_read_jsonl_tail(path:Path, n:int=2000) -> List[dict]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_map_metric_row(r:dict) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_map_fallback_log(entry:dict, idx:int) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_map_run_log_line(line:str, idx:int) -> Optional[dict]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_build_wsd_meta(cfg:Any, latest:Optional[dict], wsd_cfg_override:Any=None) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_build_wsd_insights(raw_rows:List[dict], latest:Optional[dict], run_lines:Optional[List[str]]=None) -> List[dict]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_set_active_infer_bus(bus:Optional[_InferenceEventBus]) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_get_active_infer_bus() -> Optional[_InferenceEventBus]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_infer_benchmark_from_event(event:str) -> Optional[str]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_server_log(event:str, msg:str, level:str='info', source:str='inference', lane:Optional[str]=None, scope:Optional[str]=None, benchmark:Optional[str]=None, meta:Optional[Dict[str, Any]]=None) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_resolve_scope(scope:str) -> Tuple[str, bool, bool]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_decode_params(decode_strategy:str, temperature:Optional[float]=None, top_p:Optional[float]=None) -> Tuple[Literal['greedy', 'sampling'], float, float]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_truncate_prompt_to_context(prompt:str, tokenizer:Any, context_window:Optional[int]) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_normalize_space(input_text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_preview_text(input_text:str, max_chars:int) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_split_thinking_output(text:str) -> Tuple[str, str]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_with_few_shot_chat(user_prompt:str, shots:List[Tuple[str, str]]) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_build_hellaswag_user_block(stem:str, endings:List[str]) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_build_hellaswag_prompt(stem:str, endings:List[str], few_shot_items:Optional[List[Dict[str, Any]]]=None) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_build_mmlu_user_block(question:str, options:List[str], force_cot:bool) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_build_mmlu_system_prompt(force_cot:bool) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_build_mmlu_pro_prompt(question:str, options:List[str], force_cot:bool=True, few_shot_items:Optional[List[Dict[str, Any]]]=None) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_format_gsm_answer(number_text:str, target_format:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_build_gsm_user_block(question:str, target_format:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_build_gsm8k_prompt(question:str, target_format:str='hash', few_shot_items:Optional[List[Dict[str, Any]]]=None) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_parse_choice_idx(text:str) -> Optional[int]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_parse_mmlu_answer_label(text:str) -> Optional[str]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_canonical_number(text:str) -> Optional[str]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_extract_gsm_number(text:str, target_format:str='hash') -> Optional[str]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_to_float(text:Optional[str]) -> Optional[float]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_safe_file_part(value:str, fallback:str, max_len:int=48) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_compact_stats(stats:Any) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_stable_confidence_points(raw_logs:Any) -> List[dict]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_schedule_mask_ratio(step_idx:int, total_steps:int, mask_schedule:str) -> float`
    Descrizione: Gestisce logica di mascheratura/filtri di posizioni.
  - `_resample_stability_points(points:List[dict], total_steps:int, mask_schedule:str) -> List[dict]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_normalize_messages(payload:Optional[List[Dict[str, str]]]) -> List[Dict[str, str]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_build_prompt_from_chat(prompt:str, messages:Optional[List[Dict[str, str]]], system_prompt:Optional[str], enable_thinking:Optional[bool], tokenizer:Any=None) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_find_train_python() -> str`
    Descrizione: Esegue passaggi di training/update.
  - `create_app(cfg:AppConfig, config_path:str='') -> FastAPI`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
  - `run_server(config_path:str, host:str='127.0.0.1', port:int=8080) -> None`
    Descrizione: Esegue pipeline o job completo.
  Classi:
  - `_LazyEngine`
    Metodo: `__init__(self:Any, cfg:AppConfig, trace:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_ensure(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `name(self:Any) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `bundle(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `last_stats(self:Any) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, **kwargs:Any) -> str`
    Descrizione: Genera output testo o sequenze.
    Metodo: `close(self:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `GenerateRequest`
  - `ArGenerateRequest`
  - `LoadWeightsRequest`
  - `HellaSwagItemRequest`
  - `MmluProItemRequest`
  - `Gsm8kItemRequest`
  - `Stage0StabilityRequest`
  - `Stage0DetailedLogStartRequest`
  - `Stage0DetailedLogFinishRequest`
  - `GenerateResponse`
  - `JobResponse`
  - `_InferenceEventBus`
    Metodo: `__init__(self:Any, maxlen:int=3000) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_to_int_id(self:Any, value:Any) -> int`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `emit(self:Any, payload:Dict[str, Any]) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `snapshot(self:Any, tail:int=200, after_id:Optional[str]=None) -> List[Dict[str, Any]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `subscribe(self:Any, after_id:Optional[str]=None, tail:int=0) -> _queue.Queue`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `unsubscribe(self:Any, q:_queue.Queue) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_RunManager`
    Metodo: `__init__(self:Any, api_config_path:str='') -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `start(self:Any, mode:str) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `stop(self:Any) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `snapshot(self:Any, tail:int=200) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `RunStartRequest`

- File: `backend\src\hildanext\ar.py`
  Logica d'uso: AR baseline generation for side-by-side behavior checks.
  Funzioni:
  - `_apply_penalties(logits:torch.Tensor, generated:torch.Tensor, presence_penalty:float, repetition_penalty:float) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_apply_top_k_top_p(logits:torch.Tensor, top_k:int, top_p:float) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_sample_from_logits(logits:torch.Tensor, generated:torch.Tensor, decode_mode:DecodeMode, temperature:float, top_p:float, top_k:int, presence_penalty:float, repetition_penalty:float) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_ar_decode_cached(model:Any, seq:torch.Tensor, prompt_len:int, max_new_tokens:int, decode_mode:DecodeMode, temperature:float, top_p:float, top_k:int, presence_penalty:float, repetition_penalty:float, eos_token_id:Optional[int]) -> Tuple[torch.Tensor, str]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_ar_decode_full_prefix(model:Any, seq:torch.Tensor, prompt_len:int, max_new_tokens:int, decode_mode:DecodeMode, temperature:float, top_p:float, top_k:int, presence_penalty:float, repetition_penalty:float, eos_token_id:Optional[int]) -> Tuple[torch.Tensor, str]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_run_ar_decode(bundle:ModelBundle, prompt:str, max_new_tokens:int, seed:int, decode_mode:DecodeMode='greedy', temperature:float=0.7, top_p:float=0.9, top_k:int=20, presence_penalty:float=0.0, repetition_penalty:float=1.0) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `generate_ar_from_bundle(bundle:ModelBundle, prompt:str, max_new_tokens:int=64, seed:int=42, decode_mode:DecodeMode='greedy', temperature:float=0.7, top_p:float=0.9, top_k:int=20, presence_penalty:float=0.0, repetition_penalty:float=1.0) -> Dict[str, Any]`
    Descrizione: Genera output testo o sequenze.
  - `generate_ar(cfg:AppConfig, prompt:str, max_new_tokens:int=64, seed:Optional[int]=None, trace:Any=None, decode_mode:DecodeMode='greedy', temperature:float=0.7, top_p:float=0.9, top_k:int=20, presence_penalty:float=0.0, repetition_penalty:float=1.0) -> Dict[str, Any]`
    Descrizione: Genera output testo o sequenze.

- File: `backend\src\hildanext\audit.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `collect_formula_impl() -> Dict[str, Dict[str, Any]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `paper_map() -> Dict[str, Dict[str, Any]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_safe_find_text(path:Path, needle:str) -> bool`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_find_test_refs(root:Path, symbol:str) -> List[str]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_check_invariants(symbol:str) -> Tuple[bool, str]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_status(implemented:bool, invariant_ok:bool, keyword_found:bool, test_found:bool) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_render_md(rows:List[Dict[str, Any]], summary:Dict[str, int]) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `run_audit(root_dir:str | Path, out_md:str | Path, out_json:str | Path) -> Dict[str, Any]`
    Descrizione: Esegue pipeline o job completo.

- File: `backend\src\hildanext\audit_wsd_dataset_qwen.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `analyze_schema(record:Dict[str, Any]) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `check_segmentation(text:str, s_type:str) -> List[str]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `audit_dataset() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

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
  - `_torch_runtime_info() -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_cfg_trace(config_path:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_finalize_trace(tr:Any, tk:Any) -> Any`
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
  - `cmd_merge_topk(args:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `cmd_audit(args:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `cmd_quant_bench(args:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `cmd_preflight(args:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `cmd_run_recipe(args:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `cmd_dinfer_smoke(args:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `cmd_make_stage0_config(args:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `cmd_dolma_manifest(args:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `cmd_dolma_prep(args:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `cmd_dolma_verify(args:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `cmd_preflight_wsd(args:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `cmd_run_wsd(args:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `cmd_run_stage0_inline(args:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `build_parser() -> argparse.ArgumentParser`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
  - `main() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

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
  - `ExperimentConfig`
  - `AppConfig`

- File: `backend\src\hildanext\dataset_prep_qwen.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `download_tiny_overlays(base_dir:Path) -> Tuple[Path, Path]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `stream_local_curated(dir_path:Path) -> Iterator[Dict[str, Any]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `stream_dolma_raw(raw_path:Path, max_docs:int) -> Iterator[Dict[str, Any]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `format_structured_qwen(tokenizer:Any, record:Dict[str, Any], source_type:str, strategy:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `main() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

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
  - `_records_from_packed_dir(path:Path, source:str, max_samples:int) -> List[Dict[str, Any]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
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
  - `prepare_data(cfg:AppConfig, download:bool=False, max_samples:int | None=None, trace:Any=None) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `backend\src\hildanext\diffusion.py`
  Logica d'uso: Implements WSD schedule and mixed M2T/T2T training objective.
  Funzioni:
  - `force_noncausal_attention(model:torch.nn.Module) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `wsd_block(step:int, cfg:WSDConfig, seq_len:int | None=None) -> WSDStep`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_pick_positions(base_mask:torch.Tensor, p:float) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_causal_loss(logits:torch.Tensor, labels:torch.Tensor, num_chunks:int=8) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_attn_for_model(mask:torch.Tensor, model:Any) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `reset_sdpa_probe() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_install_embed_noise_hook(model:Any, mask_id:int, noise_std:float=0.1) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_remove_embed_noise_hook() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `set_embed_noise_std(std:float) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_forward(model:Any, input_ids:torch.Tensor, attn_1d:torch.Tensor, doc_ids:torch.Tensor, mask_mode:str, clean_ids:Optional[torch.Tensor]=None, composite_block_size:int | None=None, trace:Any=None, cfg:Any=None, bidirectional:bool=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_make_m2t_batch(input_ids:torch.Tensor, attn_mask:torch.Tensor, response_mask:torch.Tensor | None, mask_id:int, ratio:float) -> Tuple[torch.Tensor, torch.Tensor]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_continuous_time_m2t_batch(input_ids:torch.Tensor, attn_mask:torch.Tensor, response_mask:torch.Tensor | None, mask_id:int, t_min:float=0.001, t_max:float=1.0) -> Tuple[torch.Tensor, torch.Tensor, float, float]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `t2t_corrupt_tokens(input_ids:torch.Tensor, attn_mask:torch.Tensor, response_mask:torch.Tensor | None, ratio:float, vocab_size:int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
  - `_make_t2t_batch(input_ids:torch.Tensor, attn_mask:torch.Tensor, response_mask:torch.Tensor | None, ratio:float, vocab_size:int) -> Tuple[torch.Tensor, torch.Tensor]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `compute_m2t_t2t_losses(model:Any, input_ids:torch.Tensor, attention_mask:torch.Tensor, doc_ids:torch.Tensor, response_mask:torch.Tensor | None, mask_id:int, vocab_size:int, cfg:TrainConfig, focus_response:bool, mask_mode:str='simple_blockdiag', composite_block_size:int | None=None, trace:Any=None, cfg_obj:Any=None, bidirectional:bool=False, time_param:str='discrete', loss_weighting:str='none', t_min:float=0.001, t_max:float=1.0, target_ids:Optional[torch.Tensor]=None) -> Dict[str, torch.Tensor]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `apply_remask(tokens:torch.Tensor, confidence:torch.Tensor, mask_id:int, cfg:RemaskConfig) -> torch.Tensor`
    Descrizione: Gestisce logica di mascheratura/filtri di posizioni.
  Classi:
  - `WSDStep`

- File: `backend\src\hildanext\formulas.py`
  Logica d'uso: Paper-aligned formula helpers for M2T/WSD/Gamma-Delta checks.
  Funzioni:
  - `_unique_sorted_pos(vals:List[int]) -> List[int]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_align_divisor(block:int, seq_len:int) -> int`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_ladder_step(step:int, warmup_steps:int, ladder:List[int]) -> int`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `llada_m2t_loss(logits:torch.Tensor, target_ids:torch.Tensor, masked_pos:torch.Tensor) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `llada2_wsd_block(step:int, warmup_steps:int, stable_steps:int, decay_steps:int, start_block:int, max_block:int, end_block:int, seq_len:int | None=None, ladder_blocks:List[int] | None=None, decay_blocks:List[int] | None=None, enforce_divisibility:bool=False) -> Tuple[str, int]`
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
  - `_model_dir_ready(model_dir:str) -> bool`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_infer_param_dtype(model:Any) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_weight_shard_count(model_dir:str) -> int`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_configure_hf_parallel_loading(model_dir:str) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_from_pretrained_best_effort(AutoModelForCausalLM:Any, model_dir:str, dtype:Any, trust_remote_code:bool) -> Any`
    Descrizione: Esegue passaggi di training/update.
  - `load_model_bundle(cfg:AppConfig, for_training:bool=False, trace:Any=None) -> ModelBundle`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `_resolve_effort(effort:str, cfg_steps:int, tau_mask:float, tau_edit:float) -> Tuple[int, float, float]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `mode_thresholds(cfg:AppConfig, mode:str, tau_mask:Optional[float], tau_edit:Optional[float]) -> Tuple[float, float]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_top1_with_confidence(logits:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_add_gumbel_noise(logits:torch.Tensor, temperature:float) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_forward_full_sequence(model:Any, seq:torch.Tensor, use_bidirectional:bool=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_predict_bidirectional(model:Any, seq:torch.Tensor, prompt_len:int, max_new:int, mask_id:int, temperature:float=0.0, use_bidirectional:bool=False) -> Tuple[torch.Tensor, torch.Tensor]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_predict_autoregressive_candidates_full(model:Any, seq:torch.Tensor, prompt_len:int, max_new:int, mask_id:int) -> Tuple[torch.Tensor, torch.Tensor]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_predict_autoregressive_candidates_cached(model:Any, seq:torch.Tensor, prompt_len:int, max_new:int, mask_id:int) -> Tuple[torch.Tensor, torch.Tensor]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_predict_autoregressive_candidates(model:Any, seq:torch.Tensor, prompt_len:int, max_new:int, mask_id:int) -> Tuple[torch.Tensor, torch.Tensor]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_token_debug_text(tok:Any, token_id:int) -> str`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
  - `_topk_token_debug(tok:Any, pred_ids:torch.Tensor, topn:int=5) -> Tuple[Dict[str, Any] | None, float, List[Dict[str, Any]]]`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
  - `_select_mask_topk(masked:torch.Tensor, confidence:torch.Tensor, k:int, exclude:torch.Tensor | None=None) -> torch.Tensor`
    Descrizione: Gestisce logica di mascheratura/filtri di posizioni.
  - `_apply_decode_policy_step(tok:Any, tokens:torch.Tensor, pred_ids:torch.Tensor, confidence:torch.Tensor, mask_id:int, tau_mask:float, tau_edit:float, budget:int, policy:str, use_bidirectional_effective:bool, remask_cfg:Any | None=None, use_remask:bool=False, is_last_step:bool=False, step_in_block:int=1, block_mask_ratio:float=1.0) -> Tuple[torch.Tensor, Dict[str, Any]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `diagnostic_dllm_decode(model:Any, tokenizer:Any, prompt:str, device:torch.device, mask_id:int, max_new_tokens:int, max_steps:int, block_size:int, tau_mask:float, tau_edit:float, temperature:float=0.0, decode_policy:str='current_base', use_bidirectional:bool=False, remask_cfg:Any | None=None, eos_guard_enabled:bool=True, plateau_patience:int=2, plateau_delta_max:int=1, cycle_guard_enabled:bool=True, allow_tau_fallback_on_degenerate:bool=False, degenerate_patience:int=2, degenerate_tau_scale:float=0.85, min_tau_mask:float=0.05, step_callback:Any | None=None, is_dummy:bool=False) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_decode_text(tok:Any, out_ids:torch.Tensor, mask_id:int, is_dummy:bool) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `build_engine(cfg:AppConfig, trace:Any=None) -> BaseEngine`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
  Classi:
  - `ModelBundle`
  - `BaseEngine`
    Metodo: `__init__(self:Any, cfg:AppConfig, trace:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, prompt:str, mode:str='S_MODE', tau_mask:float | None=None, tau_edit:float | None=None, max_new_tokens:int | None=None, seed:int | None=None, effort:str='medium', temperature:float | None=None, top_p:float | None=None, top_k:int | None=None, presence_penalty:float | None=None, repetition_penalty:float | None=None) -> str`
    Descrizione: Genera output testo o sequenze.
    Metodo: `close(self:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `TransformersEngine`
    Metodo: `__init__(self:Any, cfg:AppConfig, fallback_reason:str='', trace:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_decode(self:Any, prompt:str, mode:str, tau_mask:float | None, tau_edit:float | None, max_new_tokens:int | None, seed:int | None, effort:str='medium', temperature:float | None=None, top_p:float | None=None, top_k:int | None=None, presence_penalty:float | None=None, repetition_penalty:float | None=None) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, prompt:str, mode:str='S_MODE', tau_mask:float | None=None, tau_edit:float | None=None, max_new_tokens:int | None=None, seed:int | None=None, effort:str='medium', temperature:float | None=None, top_p:float | None=None, top_k:int | None=None, presence_penalty:float | None=None, repetition_penalty:float | None=None) -> str`
    Descrizione: Genera output testo o sequenze.
    Metodo: `close(self:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `DInferEngine`
    Metodo: `__init__(self:Any, cfg:AppConfig, trace:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_init_server(self:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `close(self:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, prompt:str, mode:str='S_MODE', tau_mask:float | None=None, tau_edit:float | None=None, max_new_tokens:int | None=None, seed:int | None=None, effort:str='medium', temperature:float | None=None, top_p:float | None=None, top_k:int | None=None, presence_penalty:float | None=None, repetition_penalty:float | None=None) -> str`
    Descrizione: Genera output testo o sequenze.

- File: `backend\src\hildanext\inference2.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `clone_hybrid_beam(state:HybridBeamState) -> HybridBeamState`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_hybrid_forward(model:Any, beam:HybridBeamState, mask_id:int, embed_layer:torch.nn.Module, use_rcd:bool=True, force_noncausal_ctx:Any=None) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `update_hybrid_rcd_state(beam:HybridBeamState, logits:torch.Tensor, embed_weight:torch.Tensor, effective_v:int, t_res:float) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `expand_hybrid_candidates(model:Any, beam:HybridBeamState, beam_size:int, mask_id:int, tokens_per_step:int, gumbel_temp:float, embed_layer:torch.nn.Module, embed_weight:torch.Tensor, effective_v:int, t_res:float, use_rcd:bool=True, force_noncausal_ctx:Any=None) -> List[Tuple[HybridBeamState, torch.Tensor, torch.Tensor]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `score_hybrid_candidate(model:Any, candidate:HybridBeamState, newly_revealed:torch.Tensor, x0_full:torch.Tensor, mask_id:int, force_noncausal_ctx:Any=None) -> float`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `score_hybrid_candidate_fallback(candidate:HybridBeamState, confidence:torch.Tensor, newly_revealed:torch.Tensor) -> float`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `prune_hybrid_beams(beams:List[HybridBeamState], beam_size:int) -> List[HybridBeamState]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `initialize_hybrid_warm_start(model:Any, init_seq:torch.Tensor, mask_id:int, embed_weight:torch.Tensor, effective_v:int, t_res:float=1.0, force_noncausal_ctx:Any=None) -> Tuple[torch.Tensor, torch.Tensor]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `inferenza2_decode(model:Any, tokenizer:Any, device:torch.device, mask_id:int, vocab_size:int, prompt:str, max_new_tokens:int=256, tau_mask:float=0.3, tau_edit:float=0.5, max_steps:int=10, seed:int=42, is_dummy:bool=False, force_noncausal_ctx:Any=None, rcd_enabled:bool=True, t_res:float=1.0, force_mask_only:bool=True, warm_start:bool=True, warm_start_model:Any=None, ots_enabled:bool=True, beam_size:int=3, gumbel_temperature:float=0.6, search_interval:int=0, pruning_mode:str='diffusion_likelihood', allow_fallback_score:bool=False, store_diagnostics:bool=True, store_trace:bool=True) -> Tuple[str, Dict[str, Any], Inference2Diagnostics]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_decode_text_hybrid(tok:Any, out_ids:torch.Tensor, mask_id:int, is_dummy:bool) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `Inference2Request`
  - `HybridCheckpointDiag`
  - `HybridStepDiag`
  - `Inference2Diagnostics`
    Metodo: `to_dict(self:Any) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `HybridBeamState`

- File: `backend\src\hildanext\inference_entrgi.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `load_reward_model(model_name:str, device:torch.device, dtype:Optional[torch.dtype]=None) -> Tuple[Any, Any]`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `_get_reward_embed_weight(reward_model:Any) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_safe_tokenizer_vocab_size(tokenizer:Any) -> int`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
  - `_tokenizer_name(tokenizer:Any) -> str`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
  - `_id_to_token(tokenizer:Any, token_id:int) -> Optional[str]`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
  - `_token_to_id(tokenizer:Any, token:str) -> Optional[int]`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
  - `build_reward_id_map(tokenizer:Any, reward_tokenizer:Any, model_vocab_size:int, reward_vocab_size:int) -> Tuple[Optional[torch.Tensor], str]`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
  - `_prepare_reward_guidance_view(tokenizer:Any, reward_tokenizer:Any, reward_embed_weight:torch.Tensor, model_vocab_size:int, mask_id:int) -> Tuple[Optional[Dict[str, Any]], str]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `compute_entropy_weights(q:torch.Tensor, vocab_size:int) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `select_lowest_entropy_mask_positions(masked_q:torch.Tensor, remaining_steps:int) -> Tuple[torch.Tensor, int, float, float]`
    Descrizione: Gestisce logica di mascheratura/filtri di posizioni.
  - `apply_entrgi_guidance(logits:torch.Tensor, seq:torch.Tensor, mask_id:int, prompt_len:int, reward_model:Any, reward_view:Dict[str, Any], guidance_scale:float, guidance_steps:int, temperature:float, device:torch.device) -> Tuple[torch.Tensor, Dict[str, float]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `entrgi_decode(model:Any, tokenizer:Any, device:torch.device, mask_id:int, vocab_size:int, prompt:str, max_new_tokens:int=256, tau_mask:float=0.3, tau_edit:float=0.5, max_steps:int=10, reward_model_name:str='Skywork/Skywork-Reward-V2-Qwen3-0.6B', guidance_scale:float=0.5, guidance_steps:int=3, entrgi_temperature:float=0.7, confidence_threshold:float=0.3, disable_guidance:bool=False, store_diagnostics:bool=True, seed:int=42, is_dummy:bool=False, force_noncausal_ctx:Any=None) -> Tuple[str, Dict[str, Any], EntRGiDiagnostics]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_decode_text(tok:Any, out_ids:torch.Tensor, mask_id:int, is_dummy:bool) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `InferenceEntRGiRequest`
  - `EntRGiStepDiag`
  - `EntRGiDiagnostics`
    Metodo: `to_dict(self:Any) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `backend\src\hildanext\inference_ots.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `clone_search_state(state:OTSBeamState) -> OTSBeamState`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_zero_mask_like(seq:torch.Tensor) -> torch.Tensor`
    Descrizione: Gestisce logica di mascheratura/filtri di posizioni.
  - `_ensure_beam_masks(state:OTSBeamState) -> None`
    Descrizione: Gestisce logica di mascheratura/filtri di posizioni.
  - `_add_gumbel_noise(logits:torch.Tensor, temperature:float) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_transfer_tokens(current:torch.Tensor, x0:torch.Tensor, confidence:torch.Tensor, mask_id:int, tokens_to_reveal:int) -> Tuple[torch.Tensor, torch.Tensor]`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
  - `_forward_model(model:Any, seq:torch.Tensor, force_noncausal_ctx:Any=None) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_denoise_active_block_step(current:torch.Tensor, x0:torch.Tensor, confidence:torch.Tensor, active_mask:torch.Tensor, mask_id:int, tau_mask:float, tau_edit:float) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_apply_block_remask(tokens:torch.Tensor, confidence:torch.Tensor, active_mask:torch.Tensor, mask_id:int, target_ratio:float, min_ratio:float) -> torch.Tensor`
    Descrizione: Gestisce logica di mascheratura/filtri di posizioni.
  - `_build_x0_full(model:Any, beam:OTSBeamState, force_noncausal_ctx:Any=None) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `score_ots_candidate(model:Any, candidate_state:OTSBeamState, newly_revealed:torch.Tensor, x0_full:torch.Tensor, mask_id:int, force_noncausal_ctx:Any=None) -> float`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `score_ots_candidate_fallback(candidate_state:OTSBeamState, confidence:torch.Tensor, newly_revealed:torch.Tensor) -> float`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `expand_ots_candidates(model:Any, beam:OTSBeamState, beam_size:int, mask_id:int, block_size:int, gumbel_temp:float, tokens_per_step:int=0, force_noncausal_ctx:Any=None) -> List[Tuple[OTSBeamState, torch.Tensor, torch.Tensor]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `prune_ots_beams(beams:List[OTSBeamState], beam_size:int) -> List[OTSBeamState]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `ots_decode(model:Any, tokenizer:Any, device:torch.device, mask_id:int, vocab_size:int, prompt:str, max_new_tokens:int=256, tau_mask:float=0.3, tau_edit:float=0.5, max_steps:int=10, beam_size:int=3, block_size:int=32, gumbel_temperature:float=0.6, search_interval:int=0, pruning_mode:str='diffusion_likelihood', allow_fallback_score:bool=False, store_trace:bool=True, seed:int=42, is_dummy:bool=False, force_noncausal_ctx:Any=None) -> Tuple[str, Dict[str, Any], OTSSearchDiagnostics]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_decode_text_ots(tok:Any, out_ids:torch.Tensor, mask_id:int, is_dummy:bool) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `InferenceOTSRequest`
  - `OTSCheckpointDiag`
  - `OTSSearchDiagnostics`
    Metodo: `to_dict(self:Any) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `OTSBeamState`

- File: `backend\src\hildanext\inference_rcd.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `compute_normalized_entropy(probs:torch.Tensor, vocab_size:int) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_compute_alpha(probs:torch.Tensor, alpha_mode:str, vocab_size:int) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `compute_rcd_residuals_from_probs(probs:torch.Tensor, embedding_weight:torch.Tensor) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `build_rcd_inputs_embeds(input_ids:torch.Tensor, mask_id:int, embedding_layer:torch.nn.Module, alpha_prev:torch.Tensor, delta_prev:torch.Tensor, force_mask_only:bool=True) -> torch.Tensor`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
  - `initialize_rcd_warm_start(model:Any, input_ids:torch.Tensor, mask_id:int, embedding_weight:torch.Tensor, vocab_size:int, t_res:float=1.0, force_noncausal_ctx:Any=None) -> Tuple[torch.Tensor, torch.Tensor]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_select_masked_positions(current:torch.Tensor, confidence:torch.Tensor, mask_id:int, tokens_per_step:int, tau_mask:float) -> torch.Tensor`
    Descrizione: Gestisce logica di mascheratura/filtri di posizioni.
  - `_commit_selected_tokens(current:torch.Tensor, pred_ids:torch.Tensor, selected:torch.Tensor) -> torch.Tensor`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
  - `_looks_like_llada_model(model:Any, tokenizer:Any=None) -> bool`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `rcd_decode(model:Any, tokenizer:Any, device:torch.device, mask_id:int, vocab_size:int, prompt:str, max_new_tokens:int=256, tau_mask:float=0.3, tau_edit:float=0.5, max_steps:int=10, t_res:float=1.0, latent_logits_temperature:Optional[float]=None, alpha_mode:str='normalized_entropy', force_mask_only:bool=True, warm_start:bool=True, warm_start_model:Any=None, single_token_mode:Optional[bool]=None, store_diagnostics:bool=True, seed:int=42, is_dummy:bool=False, force_noncausal_ctx:Any=None) -> Tuple[str, Dict[str, Any], RCDInferenceDiagnostics]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_get_embedding_layer(model:Any) -> torch.nn.Module`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_decode_text_rcd(tok:Any, out_ids:torch.Tensor, mask_id:int, is_dummy:bool) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `InferenceRCDMRequest`
  - `RCDStepDiagnostics`
  - `RCDInferenceDiagnostics`
    Metodo: `to_dict(self:Any) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `backend\src\hildanext\inference_s2d2.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `find_first_contiguous_mask_span(seq:torch.Tensor, mask_id:int, prompt_len:int, region_end:Optional[int]=None) -> Tuple[int, int]`
    Descrizione: Gestisce logica di mascheratura/filtri di posizioni.
  - `estimate_expected_accept_prefix(draft_logits:torch.Tensor, span_start:int, span_end:int, estimator:str='entropy', entropy_beta:float=1.0, margin_threshold:float=0.1, vocab_size:int=1) -> float`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `compute_verification_score(k_hat:float, cost:float, n_hi:int=0, mode:str='static') -> float`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `route_s2d2_verification(span_len:int, draft_logits:torch.Tensor, span_start:int, span_end:int, confidence:torch.Tensor, mask_positions:torch.Tensor, mask_id:int, tau:float, policy:str, min_verify_span:int, score_threshold:float, score_cost:float, score_mode:str, hysteresis_state:bool, hysteresis_on:float, hysteresis_off:float, estimator:str, entropy_beta:float, margin_threshold:float, vocab_size:int) -> Tuple[bool, bool, float]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_attn_for_model(mask:torch.Tensor, model:Any) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_model_name_candidates(model:Any, tokenizer:Any=None) -> List[str]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `detect_s2d2_verifier_mode(model:Any, tokenizer:Any=None) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `build_block_draft_attention_mask(seq_len:int, block_start:int, block_end:int, device:torch.device) -> torch.Tensor`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
  - `build_verifier_inputs_position_aligned(drafted_tokens:torch.Tensor, mask_id:int, span_start:int, span_end:int, full_seq:torch.Tensor) -> S2D2VerifierInputs`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
  - `_prefill_causal_cache(model:Any, prefix_ids:torch.Tensor) -> Tuple[Any, Optional[torch.Tensor]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_extend_causal_cache(model:Any, past_kv:Any, next_logits:Optional[torch.Tensor], tokens:torch.Tensor) -> Tuple[Any, Optional[torch.Tensor]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_clone_past_key_values(past_kv:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_past_seq_len(past_kv:Any) -> int`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_draft_forward_block(model:Any, seq:torch.Tensor, block_start:int, block_end:int, force_noncausal_ctx:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `run_s2d2_verification(model:Any, verifier_inputs:S2D2VerifierInputs, draft_probs:torch.Tensor, drafted_ids:torch.Tensor, span_start:int, span_end:int, force_noncausal_ctx:Any=None, block_prefix_cache:Any=None, block_prefix_next_logits:Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor, int, int, bool]`
    Descrizione: Esegue pipeline o job completo.
  - `s2d2_decode(model:Any, tokenizer:Any, device:torch.device, mask_id:int, vocab_size:int, prompt:str, max_new_tokens:int=256, tau_mask:float=0.3, tau_edit:float=0.5, max_steps:int=10, block_size:int=32, denoising_steps_per_block:int=0, confidence_threshold:float=0.3, routing_policy:str='min_span', min_verify_span:int=2, score_threshold:float=0.0, score_cost:float=1.0, score_mode:str='static', hysteresis_on:float=1.0, hysteresis_off:float=-5.0, acceptance_estimator:str='entropy', entropy_beta:float=1.0, margin_threshold:float=0.1, store_diagnostics:bool=True, seed:int=42, is_dummy:bool=False, force_noncausal_ctx:Any=None) -> Tuple[str, Dict[str, Any], S2D2Diagnostics]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_decode_text(tok:Any, out_ids:torch.Tensor, mask_id:int, is_dummy:bool) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `InferenceS2D2Request`
  - `S2D2StepDiag`
  - `S2D2VerifierInputs`
  - `S2D2Diagnostics`
    Metodo: `to_dict(self:Any) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

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
  - `_composite_llada20_mask(doc_ids:torch.Tensor, base_len:int, block_size:int) -> torch.Tensor`
    Descrizione: Gestisce logica di mascheratura/filtri di posizioni.
  - `batch_doc_attention_mask(doc_ids:torch.Tensor, causal:bool=False, mask_mode:str='simple_blockdiag', block_size:int | None=None, base_len:int | None=None) -> torch.Tensor`
    Descrizione: Gestisce logica di mascheratura/filtri di posizioni.
  - `response_focus_mask(response_mask:torch.Tensor, base_mask:torch.Tensor) -> torch.Tensor`
    Descrizione: Gestisce logica di mascheratura/filtri di posizioni.

- File: `backend\src\hildanext\quant.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_safe_vocab(tokenizer:Any) -> int`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_can_use_bnb(device:torch.device) -> Tuple[bool, str]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_fallback_tiny(tokenizer:Any, device:torch.device, reason:str, mode:str, trace:Any=None) -> QuantLoadResult`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `load_quantized(model_dir:str, mode:str, device:str='auto', trust_remote_code:bool=True, trace:Any=None, cfg:Any=None) -> Tuple[Any | None, Any, Dict[str, Any]]`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `_decode_tokens(tokenizer:Any, ids:torch.Tensor, mask_id:int) -> str`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
  - `_run_ar_once(model:Any, tokenizer:Any, prompt:str, max_new_tokens:int, device:torch.device, seed:int) -> Tuple[str, float]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_run_dllm_once(model:Any, tokenizer:Any, prompt:str, max_new_tokens:int, device:torch.device, mask_id:int, tau_mask:float, tau_edit:float, steps:int, seed:int) -> Tuple[str, float]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_run_train_probe(model:Any, tokenizer:Any, prompt:str, device:torch.device, seq_len:int=64) -> Dict[str, Any]`
    Descrizione: Esegue passaggi di training/update.
  - `_finite_or_none(x:Any) -> float | None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `run_quant_bench(cfg:AppConfig, modes:List[str], prompt:str, max_new_tokens:int, engine_name:str='transformers', seed:int | None=None, out_json:str | Path | None=None, train_probe:bool=False, trace:Any=None) -> Dict[str, Any]`
    Descrizione: Esegue pipeline o job completo.
  Classi:
  - `QuantLoadResult`

- File: `backend\src\hildanext\recipe.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_disk_free_gb(path:str) -> float`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_safe_ratio(text:str) -> float`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_ensure_sft_from_tiny(cfg:AppConfig, trace:Any=None) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_run_stage_with_watchdog(cfg:AppConfig, kind:str, steps:int, trace:Any=None) -> Tuple[Dict[str, Any], AppConfig]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `preflight(cfg:AppConfig, trace:Any=None) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `run_recipe_llada21(cfg:AppConfig, trace:Any=None) -> Dict[str, Any]`
    Descrizione: Esegue pipeline o job completo.
  - `dinfer_smoke(cfg:AppConfig, trace:Any=None) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `backend\src\hildanext\smoke.py`
  Logica d'uso: End-to-end smoke validation across load/train/infer.
  Funzioni:
  - `run_smoke(config_path:str) -> Dict[str, Any]`
    Descrizione: Esegue pipeline o job completo.

- File: `backend\src\hildanext\stage0_benchmarks.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_bench_root(cfg:AppConfig) -> Path`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_hellaswag_path(cfg:AppConfig, split:str) -> Path`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_download_to(url:str, dst:Path) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `ensure_hellaswag_split(cfg:AppConfig, split:str='val', force_download:bool=False) -> Tuple[Path, str]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `ensure_hellaswag_validation(cfg:AppConfig, force_download:bool=False) -> Tuple[Path, str]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_row_to_item(row:Dict[str, Any]) -> Dict[str, Any] | None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `load_hellaswag_items(cfg:AppConfig, limit:int=64, seed:int=42, force_download:bool=False, split:str='val') -> Dict[str, Any]`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `_sample_indices(total:int, limit:int, seed:int) -> List[int]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_load_hf_dataset(name:str, split:str, config_name:str | None=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `load_mmlu_pro_items(cfg:AppConfig, limit:int=150, seed:int=42, split:str='test') -> Dict[str, Any]`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `load_gsm8k_items(cfg:AppConfig, limit:int=150, seed:int=42, split:str='test', config_name:str='main') -> Dict[str, Any]`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `backend\src\hildanext\tokenization.py`
  Logica d'uso: Tokenizes and packs sequences, emits doc_ids used by attention masks.
  Funzioni:
  - `load_tokenizer(model_dir:str, trust_remote_code:bool=True, trace:Any=None, cfg:Any=None) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `ensure_mask_token(tokenizer:Any, mask_token:str, model:Any=None) -> int`
    Descrizione: Gestisce logica di mascheratura/filtri di posizioni.
  - `_ids_within_vocab(ids:List[int], vocab_size:int) -> List[int]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_encode_text(tokenizer:Any, text:str) -> List[int]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_encode_record(tokenizer:Any, record:Dict[str, Any], vocab_size:int) -> Tuple[List[int], List[int], str]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_pack(encoded:List[Tuple[List[int], List[int], str]], seq_len:int, pad_id:int, eos_id:int) -> List[Dict[str, Any]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_pack_streaming(encoded:List[Tuple[List[int], List[int], str]], seq_len:int, pad_id:int, eos_id:int, carry_ids:list, carry_docs:list, carry_resp:list, carry_src:str, doc_offset:int, trunc_prob:float=0.0) -> Tuple[list, list, list, list, str, int]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_encode_records_batch(tokenizer:Any, records:List[Dict[str, Any]], vocab_size:int) -> List[Tuple[List[int], List[int], str]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `tokenize_split(cfg:AppConfig, input_path:str, output_path:str, max_records:int | None=None, max_output_seqs:int | None=None, trace:Any=None) -> Dict[str, Any]`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
  - `tokenize_all(cfg:AppConfig, max_records:int | None=None, trace:Any=None) -> Dict[str, Any]`
    Descrizione: Gestisce tokenizzazione o manipolazione token.

- File: `backend\src\hildanext\toon.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_is_scalar(value:Any) -> bool`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_escape_string(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_needs_quotes(text:str, delimiter:str) -> bool`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_encode_scalar(value:Any, delimiter:str=',') -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_encode_key(key:Any) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_uniform_fields(items:list[Any]) -> list[str] | None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_primitive_array_inline(items:Iterable[Any], delimiter:str=',') -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_append_value(lines:list[str], key:str | None, value:Any, indent:int, delimiter:str=',') -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `dumps_toon(data:Any, root_key:str | None=None, delimiter:str=',') -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `write_toon(path:str | Path, data:Any, root_key:str | None=None, delimiter:str=',') -> Path`
    Descrizione: Serializza e salva output su disco.

- File: `backend\src\hildanext\trace.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_now() -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_jsonable(x:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `ensure_run_id(run_id:Optional[str]=None) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `config_digest_from_cfg(cfg:Any) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `set_active_trace(trace:Optional[RunTrace]) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `reset_active_trace(token:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `active_trace() -> Optional[RunTrace]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `use_trace(cfg:Any=None, trace:Optional[RunTrace]=None) -> Optional[RunTrace]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `trace_from_cfg(cfg:Any, run_id:Optional[str]=None) -> RunTrace`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `exception_with_stack(err:Exception) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `RunTrace`
    Metodo: `__init__(self:Any, run_id:str, root_log_dir:str | Path, strict_fallbacks:bool=False, fallback_whitelist:Optional[List[str]]=None, blocking_actions:Optional[List[str]]=None, blocking_reasons:Optional[List[str]]=None, config_digest:str='') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_base(self:Any, module:str, func:str, event_type:str, action:str, reason:str, timestamp_utc:Optional[str], exception_str:Optional[str], extra_dict:Optional[Dict[str, Any]]) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_append_jsonl(self:Any, path:Path, row:Dict[str, Any]) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_emit_console(self:Any, row:Dict[str, Any]) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_is_whitelisted(self:Any, row:Dict[str, Any]) -> bool`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `is_blocking(self:Any, row:Dict[str, Any], numpy_ok:bool | None=None) -> bool`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `record_fallback(self:Any, event:str, module:str, func:str, action:str, reason:str, exception_str:Optional[str]=None, extra_dict:Optional[Dict[str, Any]]=None, timestamp_utc:Optional[str]=None) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `record_env_issue(self:Any, name:str, detail:str, module:str='env', func:str='record_env_issue', extra_dict:Optional[Dict[str, Any]]=None, timestamp_utc:Optional[str]=None) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `record_notice(self:Any, module:str, func:str, action:str, reason:str, extra_dict:Optional[Dict[str, Any]]=None, timestamp_utc:Optional[str]=None) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `record_metric(self:Any, name:str, value:Any, step:Optional[int]=None, module:str='metrics', func:str='record_metric', extra_dict:Optional[Dict[str, Any]]=None, timestamp_utc:Optional[str]=None) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `snapshot_fallbacks(self:Any, limit:int=64) -> List[Dict[str, Any]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `all_events(self:Any) -> List[Dict[str, Any]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `count_fallbacks(self:Any) -> int`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `count_blocking_fallbacks(self:Any) -> int`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `flush(self:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `backend\src\hildanext\training.py`
  Logica d'uso: Runs conversion and SFT loops, logs metrics and checkpoints.
  Funzioni:
  - `_signal_handler(signum:Any, frame:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_install_signal_handlers() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_gpu_temp_celsius() -> Optional[int]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_compute_lr(step:int, warmup_steps:int, total_steps:int, base_lr:float, min_ratio:float=0.1, stable_end_step:int=0) -> float`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_t_bucket_key(t:float) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_collate(batch:List[Dict[str, Any]]) -> Dict[str, torch.Tensor]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_save_checkpoint(model:Any, tokenizer:Any, out_dir:Path, tag:str, optimizer:Any=None, watchdog:Any=None) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_prune_checkpoints(ckpt_dir:Path, keep_last:int) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_parse_step_from_tag(name:str) -> int`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_latest_checkpoint(ckpt_dir:Path) -> Tuple[int, Optional[Path]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_load_checkpoint_model(cfg:AppConfig, model:Any, ckpt_path:Path, device:torch.device, trace:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_load_optimizer_state(optimizer:Any, ckpt_path:Path, device:torch.device) -> bool`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_make_optimizer(model:Any, cfg:AppConfig, device:torch.device, trace:Any=None, kind:str='cpt') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_decode_safe(tokenizer:Any, ids:torch.Tensor, mask_id:int, dummy:bool=False) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_periodic_eval(model:Any, tokenizer:Any, device:torch.device, mask_id:int, cfg:AppConfig, seed:int) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_run(cfg:AppConfig, split_name:str, kind:str, steps:int, focus_response:bool, trace:Any=None, resume:bool=False, ckpt_every:int | None=None, eval_every:int | None=None) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `run_wsd_conversion(cfg:AppConfig, steps:int | None=None, trace:Any=None, resume:bool=False, ckpt_every:int | None=None, eval_every:int | None=None) -> Dict[str, Any]`
    Descrizione: Esegue pipeline o job completo.
  - `run_sft_training(cfg:AppConfig, steps:int | None=None, trace:Any=None, resume:bool=False, ckpt_every:int | None=None, eval_every:int | None=None) -> Dict[str, Any]`
    Descrizione: Esegue pipeline o job completo.
  - `merge_topk_checkpoints(cfg:AppConfig, checkpoint_dirs:List[str], output_dir:str) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `_TrainingWatchdog`
    Metodo: `__init__(self:Any, timeout_sec:float=600, callback:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `heartbeat(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `stop(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_run(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `TokenizedDataset`
    Metodo: `__init__(self:Any, path:str, max_rows:int | None=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__len__(self:Any) -> int`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__getitem__(self:Any, i:int) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `MmapShardedDataset`
    Metodo: `__init__(self:Any, shard_root:str, max_rows:int | None=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__len__(self:Any) -> int`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_shard_for_row(self:Any, idx:int) -> Tuple[int, int]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_get_mmap(self:Any, shard_idx:int) -> Tuple[np.ndarray, np.ndarray]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__getitem__(self:Any, idx:int) -> Dict[str, torch.Tensor]`
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
  - `env_issues() -> Dict[str, str]`
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
    Metodo: `forward(self:Any, input_ids:torch.Tensor | None=None, attention_mask:torch.Tensor | None=None, inputs_embeds:torch.Tensor | None=None, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `backend\src\hildanext\wsd_stage0.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_count_bytes(p:Path) -> int`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_infer_ext(name:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_inspect_path(p:Path) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_candidate_paths(cfg:AppConfig) -> List[Path]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `dolma_manifest(cfg:AppConfig, trace:Any=None) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_extract_text(obj:Dict[str, Any]) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_iter_json_lines(fh:Any, max_docs:int | None, seen:int) -> Iterator[Tuple[str, str, int]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `stream_docs(path:Path, max_docs:int | None=None) -> Iterator[Tuple[str, str]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_inspect_existing_doc_index(root:Path) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_resolve_external_doc_index_dir() -> Path`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_iter_external_doc_rows(doc_dir:Path, target_seq_len:int, max_rows:int) -> Iterator[List[int]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_apply_external_doc_index(train_tok:Path, eval_tok:Path, doc_dir:Path, seq_len:int) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_segment_text(text:str, max_words:int=120) -> List[str]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_split_processed(cfg:AppConfig, source_path:Path, max_docs:int, eval_pct:float, seed:int, max_segments:int | None=None) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_build_tokenized_artifacts(cfg:AppConfig, train_tok_path:Path, eval_tok_path:Path, shard_rows:int=1000) -> Dict[str, Any]`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
  - `_dolma_fingerprint(cfg:AppConfig, manifest:Dict[str, Any]) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_check_doc_boundary_signal(tokenized_train:Path) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_check_no_leakage(tokenized_train:Path) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_verify_artifacts(cfg:AppConfig) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_ensure_llada21_objective(cfg:AppConfig) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_select_optimizer_name() -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_apply_stage0_to_cfg(cfg:AppConfig, run_id:str | None=None) -> AppConfig`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `prepare_dolma_only(cfg:AppConfig, trace:Any=None) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `verify_dolma_only(cfg:AppConfig, trace:Any=None) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `preflight_wsd(cfg:AppConfig, trace:Any=None) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `archive_runs(cfg:AppConfig, trace:Any=None) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `run_wsd(cfg:AppConfig, config_path:str, trace:Any=None, skip_dolma_prep:bool=False) -> Dict[str, Any]`
    Descrizione: Esegue pipeline o job completo.
  - `create_stage0_config(cfg:AppConfig, path:Path, dolma_path:str) -> AppConfig`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.

### `backend\tests`
- File: `backend\tests\test_api_generate_smoke.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_has_model(model_dir:str) -> bool`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `test_api_generate_smoke() -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `_cfg_for_inference_log_tests() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `test_inference_logs_tail_after_and_benchmark_tags(monkeypatch:pytest.MonkeyPatch) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `test_inference_logs_stream_route_is_registered(monkeypatch:pytest.MonkeyPatch) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `_make_engine_cfg(**runtime_updates:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_fake_bundle() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `test_stop_guard_eos(monkeypatch:pytest.MonkeyPatch) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `test_stop_guard_plateau(monkeypatch:pytest.MonkeyPatch) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `test_stop_guard_cycle(monkeypatch:pytest.MonkeyPatch) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `test_policy_step_current_base_reports_tau_inactive() -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `test_policy_step_shadow_tau_mask_controls_reveal() -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `test_policy_step_shadow_tau_edit_and_remask() -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `test_policy_step_threshold_cap_uses_threshold_then_fallback() -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `test_policy_step_shadow_cap_topup_reduces_threshold_stall() -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `test_policy_step_delayed_edit_gates_delta_on_first_step() -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  Classi:
  - `_FakeTokenizer`
    Metodo: `__len__(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `encode(self:Any, text:Any, add_special_tokens:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `decode(self:Any, ids:Any, skip_special_tokens:Any=True) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__call__(self:Any, texts:Any, return_tensors:Any='pt') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_FakeModel`
    Metodo: `eval(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `train(self:Any, mode:Any=False) -> Any`
    Descrizione: Esegue passaggi di training/update.
    Metodo: `__call__(self:Any, input_ids:Any, use_cache:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_FakeEngine`
    Metodo: `__init__(self:Any, cfg:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, **kwargs:Any) -> Any`
    Descrizione: Genera output testo o sequenze.
    Metodo: `close(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `backend\tests\test_smoke.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `test_required_commands_present() -> Any`
    Descrizione: Test automatico di regressione/comportamento.

### `test`
- File: `test\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `test\build_backend_readiness.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_read_text(p:Path) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_read_json(p:Path) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_read_jsonl(p:Path) -> List[Dict[str, Any]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_code_block(txt:str, lang:str='text') -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_strip_leading_json_block(txt:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_toon_scalar(v:Any) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_uniform_obj_array(arr:List[Any]) -> List[str]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_to_toon(v:Any, indent:int=0, key:str | None=None) -> List[str]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_as_toon(v:Any) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_formula_map() -> List[Dict[str, str]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_table(rows:List[List[str]]) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_status_of(unit:Dict[str, Any], needle:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `build() -> str`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
  - `main() -> int`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `test\build_inventory.py`
  Logica d'uso: Generates this inventory report.
  Funzioni:
  - `ann(node:ast.AST | None) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `args_sig(n:ast.FunctionDef | ast.AsyncFunctionDef) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `desc_from_name(name:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `collect(scan_dirs:List[Path]) -> Dict[str, List[Path]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `parse_file(path:Path) -> Tuple[List[Dict], List[Dict]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `build_md(include_vendor:bool=False) -> str`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
  - `main() -> int`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_legacy_main() -> int`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `test\inspect_truncation.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `test\overhead_bench\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `test\overhead_bench\_common.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `fix_stdout_encoding() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_gpu_temp() -> int`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_gpu_temp_pynvml() -> int`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `gpu_info() -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `load_cfg(cfg_path:Optional[str]=None, overrides:Optional[Dict[str, Any]]=None) -> AppConfig`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `load_bundle(cfg:AppConfig, for_training:bool=True) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `make_loader(cfg:AppConfig, shuffle:bool=True) -> DataLoader`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `make_optimizer(name:str, model:Any, lr:float, wd:float, betas:Tuple[float, float]=(0.9, 0.95)) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `reset_vram() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `vram_stats() -> Dict[str, float]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `append_jsonl(path:Path, rows:list) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `write_json(path:Path, data:Any) -> Any`
    Descrizione: Serializza e salva output su disco.
  - `forward_backward(model:Any, batch:Any, bundle:Any, cfg:Any, use_amp:bool, ct_t_min:float, ct_t_max:float, mask_mode:str='simple_blockdiag', bidirectional:bool=False, grad_acc:int=1) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `test\overhead_bench\_validate_config2.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `test\overhead_bench\bench_optimizer_quality.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_run_quality_test(label:str, opt_name:str, embed_noise:bool, use_grad_scaler:bool, model:Any, bundle:Any, cfg:Any, loader_iter:Any, n_opt_steps:int, grad_acc:int) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `main() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `test\overhead_bench\bench_overhead_items.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_bench_isfinite_item(loss_tensor:torch.Tensor, n:int) -> list`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_bench_item_calls(tensors:list, n:int) -> list`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_bench_deferred_item(tensors:list, n:int) -> list`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_bench_nvidia_smi(n:int) -> list`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_bench_pynvml(n:int) -> list`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_bench_empty_cache(n:int) -> list`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_bench_wsd_block(n:int, cfg:Any) -> list`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_bench_remainder(n:int, seq_len:int=1024) -> list`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_stats(times:list) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `main() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `test\overhead_bench\bench_seq_len.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_retokenize_dolma(cfg:Any, target_seq_len:int, out_dir:Path) -> dict`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
  - `_run_seq_test(seq_len:int, model:Any, bundle:Any, cfg:Any, loader_iter:Any, n_opt_steps:int, grad_acc:int, opt_name:str='AdamW8bit') -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `main() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `test\overhead_bench\bench_vram_matrix.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_toggle_grad_ckpt(model:Any, enable:bool) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_run_single_config(label:str, opt_name:str, grad_ckpt:bool, seq_len:int, model:Any, bundle:Any, cfg:Any, loader_iter:Any, n_opt_steps:int, grad_acc:int) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_est_4000_steps(result:dict) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `main() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `test\reporting.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_now() -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_jsonable(x:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `reset_payload_log() -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `emit_payload(test_id:str, description:str, payload:Dict[str, Any]) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `load_payloads() -> List[Dict[str, Any]]`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `test\run_tests.py`
  Logica d'uso: Unified unittest runner.
  Funzioni:
  - `_is_mdm() -> bool`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `main() -> int`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `DetailedTextResult`
    Metodo: `__init__(self:Any, *args:Any, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `startTest(self:Any, test:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_append(self:Any, test:Any, status:Any, err:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `addSuccess(self:Any, test:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `addSkip(self:Any, test:Any, reason:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `addFailure(self:Any, test:Any, err:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `addError(self:Any, test:Any, err:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `DetailedRunner`

- File: `test\safe_composite_diag.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_vram() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_vram_reserved() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_sync() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_check(label:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_cleanup() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `timed(label:Any, fn:Any, abort_on_slow:Any=True) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `main() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_print_summary(dt_load:Any, dt_mask:Any, dt_attn:Any, dt_fwd_ng:Any, dt_bwd:Any, dt_mtf:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `test\test_api_generate_real_model.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_has_model(model_dir:str) -> bool`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `APIRealModelTests`
    Metodo: `test_api_generate_real_model(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `test\test_ar.py`
  Logica d'uso: Checks AR path produces output.
  Classi:
  - `ARTests`
    Metodo: `test_ar_generation_dummy(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `test\test_audit_report.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `AuditReportTests`
    Metodo: `test_audit_outputs_exist_and_have_expected_keys(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `test\test_backward_safe.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_cleanup() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_vram_mb() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `run_all() -> Any`
    Descrizione: Esegue pipeline o job completo.

- File: `test\test_bidirectional_attention.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_determine_bidirectional(phase:str, attn_mode:str) -> bool`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `BidirectionalAttentionTests`
    Metodo: `test_causal_mask_is_lower_triangular(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_bidirectional_mask_is_symmetric(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_bidirectional_still_blocks_cross_doc(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_warmup_is_causal_in_bidir_only_stable(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_stable_is_bidirectional_in_bidir_only_stable(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_decay_is_causal_in_bidir_only_stable(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_bidirectional_always_all_phases(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_causal_always_all_phases(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_wsd_phase_at_8k_boundaries(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_experiment_config_defaults(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `test\test_composite_s512.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `test\test_continuous_time_elbo.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `ContinuousTimeElboTests`
    Metodo: `_make_batch(self:Any, seq_len:int=32, t_min:float=0.001, t_max:float=1.0) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `test_t_within_bounds(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_t_distribution_covers_range(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_mask_ratio_correlates_with_t(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_mask_is_iid_per_token(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_at_least_one_token_masked(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_labels_only_at_masked_positions(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_inv_t_scaling(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_inv_t_clamped_at_tmin(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_inv_t_scaling_at_t1(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_inv_t_increases_loss_for_small_t(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `test\test_doc_mask_no_leakage_stronger.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `DocMaskNoLeakageStrongerTests`
    Metodo: `test_strict_no_cross_doc_leakage(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_causal_inside_same_doc(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `test\test_fallback_tracing_mandatory.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_read_jsonl(p:Path) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `FallbackTracingMandatoryTests`
    Metodo: `test_dinfer_missing_is_traced_in_logs_and_stats(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `test\test_formulas.py`
  Logica d'uso: Checks LLaDA/LLaDA2/LLaDA2.1 formula behavior.
  Classi:
  - `FormulaTests`
    Metodo: `test_llada_m2t_loss_matches_masked_ce(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_llada2_wsd_phase_boundaries(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_llada21_gamma_delta_sets(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `test\test_infer_parity_dinfer.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `DInferParityTests`
    Metodo: `test_one_step_threshold_parity(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `test\test_inference2.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_make_dummy_model(vocab:int=64, hidden:int=16) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_make_dummy_tokenizer(vocab:int=64) -> Any`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
  Classi:
  - `TestNonRegression`
    Metodo: `test_rcd_still_works(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_ots_still_works(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `TestInferenza2Routing`
    Metodo: `test_inferenza2_endpoint_exists(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_old_endpoints_still_exist(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `TestHybridBeamState`
    Metodo: `test_clone_has_independent_rcd_state(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_expansion_produces_children_with_rcd(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `TestRCDMathReuse`
    Metodo: `test_residual_from_probs_times_codebook(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_normalized_entropy_bounds(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `TestOTSSearchInHybrid`
    Metodo: `test_hybrid_full_decode_has_checkpoints(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `TestHybridForwardPath`
    Metodo: `test_hybrid_forward_with_rcd(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_hybrid_forward_without_rcd(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `TestHybridScoring`
    Metodo: `test_diffusion_native_score_returns_float(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_fallback_score_when_enabled(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `TestGracefulFallback`
    Metodo: `test_invalid_pruning_mode_raises(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_invalid_pruning_mode_falls_back(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `TestDeterminism`
    Metodo: `test_same_seed_same_output(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `TestAblationModes`
    Metodo: `test_rcd_only_mode(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_ots_only_mode(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_full_hybrid_mode(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `test\test_inference_entrgi.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_make_dummy_model(vocab:int=VOCAB, hidden:int=16) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_make_dummy_tokenizer(vocab:int=VOCAB) -> Any`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
  Classi:
  - `_AlignedTokenizer`
    Metodo: `__init__(self:Any, vocab_size:int) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__len__(self:Any) -> int`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `convert_ids_to_tokens(self:Any, idx:int) -> str`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
    Metodo: `convert_tokens_to_ids(self:Any, token:str) -> int`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
    Metodo: `get_vocab(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `_ShiftedTokenizer`
    Metodo: `get_vocab(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `convert_tokens_to_ids(self:Any, token:str) -> int`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
  - `TestEntRGiRequestSchema`
    Metodo: `test_default_values(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `TestEntropyWeights`
    Metodo: `test_uniform_distribution_has_weight_one(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_delta_distribution_has_weight_zero(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `TestLowestEntropySelection`
    Metodo: `test_budget_uses_remaining_steps(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_prefers_low_entropy_positions(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `TestRewardAlignment`
    Metodo: `test_identity_alignment(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_misaligned_tokenizer_fails(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `TestRewardLoading`
    Metodo: `test_invalid_model_returns_none(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_extract_embed_weight(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `TestEntRGiEndToEnd`
    Metodo: `test_dummy_decode_runs_without_reward_model(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_diagnostics_payload_updated(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_disable_guidance_keeps_entropy_ranked_decode(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_models_stay_frozen(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `TestEntRGiRouting`
    Metodo: `test_inferenceentrgi_endpoint_exists(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_original_generate_still_exists(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_inferences2d2_still_exists(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_inferenceots_still_exists(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `TestStopGradientLogic`
    Metodo: `test_gradient_flows_through_soft_path(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `test\test_inference_ots.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_make_dummy_model(vocab:int=64, hidden:int=16) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_make_dummy_tokenizer(vocab:int=64) -> Any`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
  Classi:
  - `TestOTSRouting`
    Metodo: `test_original_generate_endpoint_exists(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_inferenceots_endpoint_exists(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `TestOTSNonRegression`
    Metodo: `test_tinycausal_basic_forward(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `TestOTSBeamLifecycle`
    Metodo: `test_multi_beam_init(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `TestOTSExpansion`
    Metodo: `test_expansion_produces_distinct_children(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `TestOTSPruning`
    Metodo: `test_prune_to_beam_size(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `TestOTSScorePath`
    Metodo: `test_diffusion_likelihood_score(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_fallback_score_is_different_path(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `TestOTSDeterminism`
    Metodo: `test_deterministic_output(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `TestOTSGracefulFallback`
    Metodo: `test_unknown_mode_error(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_fallback_allowed_mode(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `TestOTSGumbelNoise`
    Metodo: `test_gumbel_varies_argmax(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_zero_temp_no_noise(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `TestOTSEndToEnd`
    Metodo: `test_e2e_basic(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_beam1_degenerates_to_standard(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `TestOTSCloneState`
    Metodo: `test_clone_independence(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `TestOTSSchema`
    Metodo: `test_default_values(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `test\test_inference_rcd.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_make_dummy_model(vocab:int=64, hidden:int=16) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_make_dummy_tokenizer(vocab:int=64) -> Any`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
  Classi:
  - `TestRCDRouting`
    Metodo: `test_original_generate_endpoint_exists(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_inferencercdm_endpoint_exists(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `TestRCDNonRegression`
    Metodo: `test_tinycausal_input_ids_still_works(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_tinycausal_inputs_embeds_works(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `TestRCDMath`
    Metodo: `test_residual_vector_eq1(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `TestMaskOnlyInjection`
    Metodo: `test_committed_tokens_untouched(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_mask_positions_interpolated(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `TestEntropyAlpha`
    Metodo: `test_entropy_bounds(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_temperature_scaled_alpha(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `TestWarmStart`
    Metodo: `test_warm_start_produces_residuals(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `TestNoHiddenStateShortcut`
    Metodo: `test_residual_source_is_probs_not_hidden(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_get_embedding_layer_returns_embedding(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `TestGracefulFallback`
    Metodo: `test_request_defaults(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `TestRCDDecodeEndToEnd`
    Metodo: `test_rcd_decode_runs(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_rcd_decode_without_warm_start(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `test\test_inference_s2d2.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_make_dummy_model(vocab:int=64, hidden:int=16) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_make_dummy_tokenizer(vocab:int=64) -> Any`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
  Classi:
  - `TestS2D2Routing`
    Metodo: `test_inferences2d2_endpoint_exists(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_original_generate_still_exists(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_inferenceots_still_exists(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_inferencercdm_still_exists(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `TestS2D2RequestSchema`
    Metodo: `test_default_values(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_custom_values(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `TestContiguousSpan`
    Metodo: `test_all_masked(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_partial_masked(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_no_masked(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_single_mask(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_gap_then_mask(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `TestAcceptPrefixEstimation`
    Metodo: `test_entropy_estimator(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_margin_estimator(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_empty_span(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `TestVerificationScore`
    Metodo: `test_static_score(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_dynamic_score(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `TestRoutingPolicies`
    Metodo: `_make_dummy_args(self:Any, span_len:Any=4, policy:Any='min_span') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `test_min_span_policy_above(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_min_span_policy_below(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_always_policy(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_never_policy(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_score_threshold_policy(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_hysteresis_transitions(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `TestVerifierReusesSameModel`
    Metodo: `test_same_model_used(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_verification_produces_results(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `TestAcceptanceLeftToRight`
    Metodo: `test_acceptance_stops_at_rejection(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `TestFallbackBehavior`
    Metodo: `test_never_policy_uses_fallback(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `TestS2D2EndToEnd`
    Metodo: `test_e2e_basic(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_e2e_min_span(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_diagnostics_structure(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_detect_verifier_mode_defaults_to_position_aligned(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_not_equivalent_to_global_ar(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `TestBlockDraftAttentionMask`
    Metodo: `test_shape(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_prefix_causal(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_block_sees_prefix(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_block_fully_visible(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_no_prefix(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `TestVerifierMaskPathNoTypo`
    Metodo: `test_diagnostics_default(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_e2e_diagnostics(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `TestRightShiftedARPath`
    Metodo: `test_right_shifted_detection_for_dream(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_right_shifted_detection_for_fast_dllm(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_right_shifted_verifier_inputs_built(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `test\test_inference_threshold_decode_invariants.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `ThresholdDecodeInvariantTests`
    Metodo: `test_gamma_delta_membership_and_disjointness(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_apply_semantics(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_delta_monotonicity_vs_tau_edit(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `test\test_left_shift_labels.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LeftShiftLabelTests`
    Metodo: `test_causal_loss_uses_left_shift(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_position_0_never_predicted(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_position_1_is_first_target(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_short_sequence_returns_zero(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_accuracy_shift_alignment(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_pred_positions_count_matches_labels(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_no_labels_means_zero_pred_positions(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `test\test_llada20_composite_mask_doc_gating.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LLaDA20CompositeMaskDocGatingTests`
    Metodo: `test_doc_gating_with_padding(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `test\test_llada20_composite_mask_structure.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LLaDA20CompositeMaskStructureTests`
    Metodo: `test_composite_mask_regions(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `test\test_lr_schedule.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LRScheduleTests`
    Metodo: `test_lr_at_step_0_is_zero(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_lr_at_warmup_end_equals_base(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_lr_at_total_steps_equals_min(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_warmup_is_linear(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_warmup_midpoint(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_cosine_decay_monotonically_decreasing(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_cosine_midpoint(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_min_ratio_0_reaches_zero(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_min_ratio_1_keeps_constant(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_warmup_0_starts_at_base(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_lr_never_negative(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `test\test_masks.py`
  Logica d'uso: Checks document boundary masking behavior.
  Classi:
  - `MaskTests`
    Metodo: `test_doc_attention_mask_blocks_cross_doc(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_batch_doc_attention_mask_shape(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `test\test_model_load_ar_real.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_has_model_dir(p:str) -> bool`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `RealModelARTests`
    Metodo: `test_model_load_and_ar_greedy_determinism(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `test\test_mtf_t2t.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_make_batch(B:Any=1, S:Any=64, V:Any=128, mask_id:Any=126) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_default_cfg(**overrides:Any) -> TrainConfig`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `test_return_fields() -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `test_target_ids() -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `test_mtf_loop_gradients() -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `test_mtf_turns_differ() -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `test_mtf_1_backward_compat() -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `test_config_mtf_value() -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `test_wsd_stage0_mtf() -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `test_vram_stability() -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  Classi:
  - `_DummyLMHead`
    Metodo: `__init__(self:Any, vocab_size:int=128, hidden:int=32, seq_len:int=64) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, input_ids:Any, attention_mask:Any=None, **kw:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `test\test_option3_plus_2.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `test\test_option3_slim.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `test\test_pre_sft_sanity.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_has_real_model(cfg:Any) -> bool`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `PreSFTSanityTests`
    Metodo: `test_prepare_tokenize_and_one_step_training(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `test\test_precision.py`
  Logica d'uso: Checks fp16/fp32 and numerical validity.
  Classi:
  - `PrecisionTests`
    Metodo: `test_dtype_mapping_cpu(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_tiny_forward_fp32(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_tiny_forward_fp16_if_cuda(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `test\test_quant_vram_sanity.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_has_model_dir(p:str) -> bool`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_finite_or_none(x:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `QuantVRAMSanityTests`
    Metodo: `test_quant_bench_report_schema(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `test\test_real_dllm_decode_non_degenerate.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_has_model(model_dir:str) -> bool`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `RealDLLMDecodeTests`
    Metodo: `test_real_decode_non_degenerate(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `test\test_remask_invariants.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `RemaskInvariantTests`
    Metodo: `test_output_domain_and_shape(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_low_confidence_prefers_remask(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_remask_count_bound(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `test\test_sft_smoke.py`
  Logica d'uso: Checks one-step SFT smoke path.
  Funzioni:
  - `_model_exists(model_dir:str) -> bool`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `SFTSmokeTests`
    Metodo: `test_sft_one_step(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `test\test_stage0_dolma.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_find_real_path(cfg:Any) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `Stage0DolmaTests`
    Metodo: `setUpClass(cls:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `tearDownClass(cls:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `test_dolma_manifest_real_ok(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_no_synthetic_dolma_allowed(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_tokenized_artifacts_exist(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_doc_boundary_signal_real(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_stream_read_first_k_docs(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `test\test_t2t_corruption_and_recovery.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `T2TCorruptionTests`
    Metodo: `test_t2t_corruption_and_recovery(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `test\test_tokenizer_mask_real.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_has_model_dir(p:str) -> bool`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `TokenizerMaskRealTests`
    Metodo: `test_mask_token_id_and_embedding_resize(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `test\test_vocab_mask.py`
  Logica d'uso: Checks vocab length and mask token consistency.
  Classi:
  - `VocabMaskTests`
    Metodo: `test_simple_tokenizer_mask(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_local_tokenizer_mask(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `test\test_wsd_benchmark_suite.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_vram_mb() -> float`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_vram_peak_mb() -> float`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_ram_mb() -> float`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_full_cleanup() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_set_seed(seed:int=SEED) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_load_model(grad_ckpt:bool=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_load_batch(n_rows:int=1) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_get_mask_id(tok:Any) -> int`
    Descrizione: Gestisce logica di mascheratura/filtri di posizioni.
  - `_get_train_cfg(mtf:int=1) -> Any`
    Descrizione: Esegue passaggi di training/update.
  - `_make_optimizer(model:Any, name:str, lr:float=5e-05, wd:float=0.1, betas:Tuple[float, float]=(0.9, 0.95)) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_run_n_steps(model:Any, opt:Any, mask_id:int, vocab_size:int, n_steps:int, mask_mode:str, block_size:int, bidirectional:bool, mtf:int=1, seq_len:int=SEQ_LEN, use_amp:bool=True) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `test_1_optimizer_comparison() -> Dict[str, Any]`
    Descrizione: Test automatico di regressione/comportamento.
  - `test_2_gradient_checkpointing(best_optimizer:str='paged_adamw8bit') -> Dict[str, Any]`
    Descrizione: Test automatico di regressione/comportamento.
  - `test_3_mask_mode_backward_time() -> Dict[str, Any]`
    Descrizione: Test automatico di regressione/comportamento.
  - `test_4_reduced_seqlen() -> Dict[str, Any]`
    Descrizione: Test automatico di regressione/comportamento.
  - `test_5_mtf_impact() -> Dict[str, Any]`
    Descrizione: Test automatico di regressione/comportamento.
  - `test_6_eta_per_phase(test3_results:Optional[Dict]=None) -> Dict[str, Any]`
    Descrizione: Test automatico di regressione/comportamento.
  - `test_7_fp16_vs_fp32() -> Dict[str, Any]`
    Descrizione: Test automatico di regressione/comportamento.
  - `main() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `test\test_wsd_config_8k.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `WSDConfig8kTests`
    Metodo: `_make_base_cfg(self:Any) -> AppConfig`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `test_create_stage0_config_steps(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_create_stage0_config_seq_len(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_create_stage0_config_experiment_flags(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_create_stage0_config_wsd_fractions(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_create_stage0_config_grad_accum(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_create_stage0_config_saves_loadable_json(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_apply_stage0_wsd_steps(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_apply_stage0_grad_clip(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_apply_stage0_lr_min_ratio(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_apply_stage0_seq_len_propagated(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_apply_stage0_ladder_blocks(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_train_config_has_grad_clip(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_train_config_has_lr_min_ratio(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_t_bucket_key_ranges(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_t_bucket_names_consistent(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_cooldown_settings(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_eval_and_save_every(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_config_roundtrip_preserves_experiment(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `test\test_wsd_ladder_and_divisibility.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `WSDLadderTests`
    Metodo: `test_wsd_ladder_and_divisibility(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `test\test_wsd_perf_diag.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_vram_mb() -> float`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_vram_peak_mb() -> float`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_ram_mb() -> float`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_full_cleanup() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_load_model() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_create_optimizer(model:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_load_batch(n_rows:int=1) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_fmt(mb:float) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `test_1_baseline_load() -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `test_2_simple_fwd_bwd() -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `test_3_composite_fwd_bwd() -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `test_4_composite_gradckpt_mtf2() -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `test_5_full_optim_step() -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `test_6_dataset_load() -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `test_7_bidir_conformity() -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `main() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
