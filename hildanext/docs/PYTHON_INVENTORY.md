# Python Inventory
Scope: `hildanext/backend/src/hildanext`, `hildanext/backend/tests`, `hildanext/test`.
Vendor included: no.

## Cartelle
- `backend\src\hildanext`: 17 file Python
- `backend\tests`: 1 file Python
- `test`: 9 file Python

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
  - `cmd_merge_topk(args:Any) -> Any`
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
  - `ModelConfig`
  - `WSDConfig`
  - `TrainConfig`
  - `RemaskConfig`
  - `InferenceConfig`
  - `RuntimeConfig`
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
  - `_forward(model:Any, input_ids:torch.Tensor, attn_1d:torch.Tensor, doc_ids:torch.Tensor) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_make_m2t_batch(input_ids:torch.Tensor, attn_mask:torch.Tensor, response_mask:torch.Tensor | None, mask_id:int, ratio:float) -> Tuple[torch.Tensor, torch.Tensor]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_make_t2t_batch(input_ids:torch.Tensor, attn_mask:torch.Tensor, response_mask:torch.Tensor | None, ratio:float, vocab_size:int) -> Tuple[torch.Tensor, torch.Tensor]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `compute_m2t_t2t_losses(model:Any, input_ids:torch.Tensor, attention_mask:torch.Tensor, doc_ids:torch.Tensor, response_mask:torch.Tensor | None, mask_id:int, vocab_size:int, cfg:TrainConfig, focus_response:bool) -> Dict[str, torch.Tensor]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
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
  - `load_model_bundle(cfg:AppConfig, for_training:bool=False) -> ModelBundle`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `mode_thresholds(cfg:AppConfig, mode:str, tau_mask:Optional[float], tau_edit:Optional[float]) -> Tuple[float, float]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `build_engine(cfg:AppConfig) -> BaseEngine`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
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
  - `_pack(encoded:List[Tuple[List[int], List[int], str]], seq_len:int, pad_id:int, eos_id:int) -> List[Dict[str, Any]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `tokenize_split(cfg:AppConfig, input_path:str, output_path:str, max_records:int | None=None) -> Dict[str, Any]`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
  - `tokenize_all(cfg:AppConfig, max_records:int | None=None) -> Dict[str, Any]`
    Descrizione: Gestisce tokenizzazione o manipolazione token.

- File: `backend\src\hildanext\training.py`
  Logica d'uso: Runs conversion and SFT loops, logs metrics and checkpoints.
  Funzioni:
  - `_collate(batch:List[Dict[str, Any]]) -> Dict[str, torch.Tensor]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_save_checkpoint(model:Any, tokenizer:Any, out_dir:Path, tag:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_run(cfg:AppConfig, split_name:str, kind:str, steps:int, focus_response:bool) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `run_wsd_conversion(cfg:AppConfig, steps:int | None=None) -> Dict[str, Any]`
    Descrizione: Esegue pipeline o job completo.
  - `run_sft_training(cfg:AppConfig, steps:int | None=None) -> Dict[str, Any]`
    Descrizione: Esegue pipeline o job completo.
  - `merge_topk_checkpoints(cfg:AppConfig, checkpoint_dirs:List[str], output_dir:str) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
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

### `backend\tests`
- File: `backend\tests\test_smoke.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `test_required_commands_present() -> Any`
    Descrizione: Test automatico di regressione/comportamento.

### `test`
- File: `test\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

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

- File: `test\run_tests.py`
  Logica d'uso: Unified unittest runner.
  Funzioni:
  - `main() -> int`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `test\test_ar.py`
  Logica d'uso: Checks AR path produces output.
  Classi:
  - `ARTests`
    Metodo: `test_ar_generation_dummy(self:Any) -> Any`
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

- File: `test\test_masks.py`
  Logica d'uso: Checks document boundary masking behavior.
  Classi:
  - `MaskTests`
    Metodo: `test_doc_attention_mask_blocks_cross_doc(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_batch_doc_attention_mask_shape(self:Any) -> Any`
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

- File: `test\test_sft_smoke.py`
  Logica d'uso: Checks one-step SFT smoke path.
  Classi:
  - `SFTSmokeTests`
    Metodo: `test_sft_one_step(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `test\test_vocab_mask.py`
  Logica d'uso: Checks vocab length and mask token consistency.
  Classi:
  - `VocabMaskTests`
    Metodo: `test_simple_tokenizer_mask(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_local_tokenizer_mask(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
