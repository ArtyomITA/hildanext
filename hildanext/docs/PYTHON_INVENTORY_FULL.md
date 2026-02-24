# Python Inventory
Scope: `hildanext/backend/src/hildanext`, `hildanext/backend/tests`, `hildanext/test`.
Vendor included: yes.

## Cartelle
- `backend\src\hildanext`: 17 file Python
- `backend\tests`: 1 file Python
- `test`: 9 file Python
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

### `vendor\llada`
- File: `vendor\llada\app.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `parse_constraints(constraints_text:Any) -> Any`
    Descrizione: Esegue passaggi di training/update.
  - `format_chat_history(history:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `add_gumbel_noise(logits:Any, temperature:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_num_transfer_tokens(mask_index:Any, steps:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `generate_response_with_visualization(model:Any, tokenizer:Any, device:Any, messages:Any, gen_length:Any=64, steps:Any=32, constraints:Any=None, temperature:Any=0.0, cfg_scale:Any=0.0, block_length:Any=32, remasking:Any='low_confidence') -> Any`
    Descrizione: Genera output testo o sequenze.
  - `create_chatbot_demo() -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.

- File: `vendor\llada\chat.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `chat() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\eval_llada.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `set_seed(seed:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `LLaDAEvalHarness`
    Metodo: `__init__(self:Any, model_path:Any='', mask_id:Any=126336, max_length:Any=4096, batch_size:Any=32, mc_num:Any=128, is_check_greedy:Any=True, cfg:Any=0.0, steps:Any=1024, gen_length:Any=1024, block_length:Any=1024, remasking:Any='low_confidence', device:Any='cuda', **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `rank(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `world_size(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_forward_process(self:Any, batch:Any, prompt_index:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_logits(self:Any, batch:Any, prompt_index:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_loglikelihood(self:Any, prefix:Any, target:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `suffix_greedy_prediction(self:Any, prefix:Any, target:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_encode_pair(self:Any, context:Any, continuation:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `loglikelihood(self:Any, requests:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `loglikelihood_rolling(self:Any, requests:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate_until(self:Any, requests:list[Instance]) -> Any`
    Descrizione: Genera output testo o sequenze.

- File: `vendor\llada\eval_reverse.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `parse_args() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `next_predition_pairs(poems:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `prev_predition_pairs(poems:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\generate.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `add_gumbel_noise(logits:Any, temperature:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_num_transfer_tokens(mask_index:Any, steps:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `generate(model:Any, prompt:Any, attention_mask:Any=None, steps:Any=128, gen_length:Any=128, block_length:Any=128, temperature:Any=0.0, cfg_scale:Any=0.0, remasking:Any='low_confidence', mask_id:Any=126336, logits_eos_inf:Any=False, confidence_eos_eot_inf:Any=False) -> Any`
    Descrizione: Genera output testo o sequenze.
  - `main() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\get_log_likelihood.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `forward_process(batch:Any, prompt_index:Any, mask_id:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_logits(model:Any, batch:Any, prompt_index:Any, cfg_scale:Any, mask_id:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `get_log_likelihood(model:Any, prompt:Any, answer:Any, mc_num:Any=128, batch_size:Any=16, cfg_scale:Any=0.0, mask_id:Any=126336) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `main() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\docs\en\conf.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_version() -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `builder_inited_handler(app:Any) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
  - `setup(app:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\docs\en\statis.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `table_format(data_list:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `generate_table(data_list:Any, title:Any=None) -> Any`
    Descrizione: Genera output testo o sequenze.

- File: `vendor\llada\opencompass\docs\zh_cn\conf.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_version() -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `builder_inited_handler(app:Any) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
  - `setup(app:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\docs\zh_cn\statis.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `table_format(data_list:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `generate_table(data_list:Any, title:Any=None) -> Any`
    Descrizione: Genera output testo o sequenze.

- File: `vendor\llada\opencompass\examples\llada_1p5_gen_gpqa_length256_block16.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\examples\llada_1p5_gen_gsm8k_length256_block16_confidence.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\examples\llada_1p5_gen_humaneval_length512_block32_confidence.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\examples\llada_1p5_gen_ifeval_length256_block16_confidence.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\examples\llada_1p5_gen_math_length1024_block128_confidence.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\examples\llada_1p5_gen_mbpp_length512_block32_confidence.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\examples\llada_base_gen_bbh_length256_block256.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\examples\llada_base_gen_gsm8k_length256_block256.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\examples\llada_base_gen_humaneval_length256_block256.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\examples\llada_base_gen_math_length256_block256.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\examples\llada_base_gen_mbpp_length256_block256.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\examples\llada_instruct_gen_arcc_length512_block512.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\examples\llada_instruct_gen_gpqa_length128_block64.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\examples\llada_instruct_gen_gpqa_length64_block64_confidence.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\examples\llada_instruct_gen_gsm8k_length256_block8.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\examples\llada_instruct_gen_gsm8k_length512_block512_confidence.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\examples\llada_instruct_gen_hellaswag_length3_block3.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\examples\llada_instruct_gen_humaneval_length512_block32.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\examples\llada_instruct_gen_humaneval_length512_block512_logits.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\examples\llada_instruct_gen_ifeval_length512_block512_confidence.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\examples\llada_instruct_gen_math_length256_block256.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\examples\llada_instruct_gen_math_length512_block512_confidence.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\examples\llada_instruct_gen_math_length512_block64.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\examples\llada_instruct_gen_mbpp_length256_block256_confidence.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\examples\llada_instruct_gen_mbpp_length512_block32.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\examples\llada_instruct_gen_mmlu_length3_block3.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\examples\llada_instruct_gen_mmlupro_length256_block256.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\cli\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\cli\main.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `parse_args() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `parse_slurm_args(slurm_parser:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `parse_dlc_args(dlc_parser:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `parse_hf_args(hf_parser:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `parse_custom_dataset_args(custom_dataset_parser:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `main() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\configs\dataset_collections\chat_OC15.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\ARC_c\ARC_c_clean_ppl.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\ARC_c\ARC_c_cot_gen_926652.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\ARC_c\ARC_c_few_shot_gen_e9b043.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\ARC_c\ARC_c_few_shot_ppl.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\ARC_c\ARC_c_gen.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\ARC_c\ARC_c_gen_1e0de5.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\ARC_c\ARC_c_ppl.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\ARC_c\ARC_c_ppl_2ef631.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\ARC_c\ARC_c_ppl_a450bd.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\ARC_c\ARC_c_ppl_d52a21.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\bbh\bbh_0shot_nocot_academic_gen.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\bbh\bbh_0shot_nocot_gen_925fc4.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\bbh\bbh_0shot_nocot_gen_9c32f6.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\bbh\bbh_0shot_nocot_gen_ea7952.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\bbh\bbh_3shot_gen.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\bbh\bbh_gen.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\bbh\bbh_gen_2879b0.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\bbh\bbh_gen_4a31fa.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\bbh\bbh_gen_5b92b0.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\bbh\bbh_gen_5bf00b.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\bbh\bbh_gen_98fba6.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\bbh\bbh_gen_ee62e9.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\bbh\bbh_llm_judge_gen.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\bbh\bbh_llmjudge_gen_b5bdf1.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\bbh\bbh_subset_settings.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\gpqa\gpqa_0shot_nocot_gen_772ea0.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\gpqa\gpqa_0shot_nocot_genericllmeval_gen_772ea0.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\gpqa\gpqa_0shot_nocot_genericllmeval_xml_gen_772ea0.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\gpqa\gpqa_0shot_nocot_llmjudge_gen_772ea0.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\gpqa\gpqa_cascade_eval_academic.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\gpqa\gpqa_cascade_eval_gen_772ea0.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\gpqa\gpqa_few_shot_ppl_4b5a83.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\gpqa\gpqa_fewshot_gen.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `GPQASimpleEval_postprocess_(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\gpqa\gpqa_gen.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\gpqa\gpqa_gen_015262.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\gpqa\gpqa_gen_4baadb.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\gpqa\gpqa_llm_judge_gen.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\gpqa\gpqa_openai_simple_evals_gen_5aeece.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\gpqa\gpqa_ppl_6bf57a.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\gsm8k\deprecated_gsm8k_agent_gen_be1606.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\gsm8k\gsm8k_0shot_gen_a58960.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\gsm8k\gsm8k_0shot_nocot_gen_6cbf22.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\gsm8k\gsm8k_0shot_v2_gen_17d799.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\gsm8k\gsm8k_0shot_v2_gen_6e39a4.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\gsm8k\gsm8k_0shot_v2_gen_a58960.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\gsm8k\gsm8k_agent_gen_c3dff3.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\gsm8k\gsm8k_gen.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\gsm8k\gsm8k_gen_17d0dc.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\gsm8k\gsm8k_gen_1d7fe4.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\gsm8k\gsm8k_gen_1dce88.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\gsm8k\gsm8k_gen_3309bd.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\gsm8k\gsm8k_gen_57b0b1.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\gsm8k\gsm8k_gen_701491.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\gsm8k\gsm8k_gen_a3e34a.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\gsm8k\gsm8k_gen_base.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\gsm8k\gsm8k_gen_d6de81.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\gsm8k\gsm8k_gen_e9e91e.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\gsm8k\gsm8k_gen_ee684f.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\gsm8k\gsm8k_model_postprocess_gen_a58960.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\gsm8k\gsm8k_xfinder_gen_a58960.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\hellaswag\hellaswag_10shot_gen_e42710.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\hellaswag\hellaswag_10shot_ppl_59c85e.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\hellaswag\hellaswag_clean_ppl.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\hellaswag\hellaswag_gen.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\hellaswag\hellaswag_gen_6faab5.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\hellaswag\hellaswag_llm_judge_gen.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\hellaswag\hellaswag_llmjudge_gen_809ef1.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\hellaswag\hellaswag_ppl.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\hellaswag\hellaswag_ppl_47bff9.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\hellaswag\hellaswag_ppl_7d7f2d.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\hellaswag\hellaswag_ppl_9dbb12.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\hellaswag\hellaswag_ppl_a6e128.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\humaneval\deprecated_humaneval_gen_4a6eef.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\humaneval\deprecated_humaneval_gen_6d1cc2.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\humaneval\deprecated_humaneval_gen_a82cae.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\humaneval\deprecated_humaneval_gen_d2537e.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\humaneval\deprecated_humaneval_gen_fd5822.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\humaneval\deprecated_humaneval_gen_ff7054.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\humaneval\humaneval_gen.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\humaneval\humaneval_gen_66a7f4.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\humaneval\humaneval_gen_8e312c.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\humaneval\humaneval_gen_base.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\humaneval\humaneval_openai_sample_evals_gen_159614.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\humaneval\humaneval_openai_sample_evals_gen_dcae0e.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\humaneval\humaneval_openai_sample_evals_o1_gen_5e7b00.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\humaneval\humaneval_openai_sample_evals_repeat_gen_dcae0e.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\humaneval\humaneval_passk_gen_8e312c.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\humaneval\humaneval_repeat10_gen_8e312c.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\humaneval\internal_humaneval_gen_ce6b06.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\humaneval\internal_humaneval_gen_d2537e.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\IFEval\IFEval_gen.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\IFEval\IFEval_gen_3321a3.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\IFEval\IFEval_gen_353ae7.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\math\deprecated_math_agent_evaluatorv2_gen_861b4f.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\math\deprecated_math_evaluatorv2_gen_265cce.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\math\math_0shot_gen_11c4b5.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\math\math_0shot_gen_393424.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\math\math_0shot_llm_judge_gen_393424.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\math\math_0shot_llm_judge_v2_gen_31d777.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\math\math_4shot_base_gen_43d5b6.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\math\math_4shot_base_gen_db136b.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\math\math_4shot_example_from_google_research.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\math\math_500_cascade_eval_gen_6ff468.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\math\math_500_gen.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\math\math_500_llmjudge_gen_6ff468.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\math\math_agent_evaluatorv2_gen_0c1b4e.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\math\math_agent_gen_0c1b4e.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\math\math_agent_gen_861b4f.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\math\math_agent_gen_af2293.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\math\math_evaluatorv2_gen_2f4a71.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\math\math_evaluatorv2_gen_cecb31.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\math\math_gen.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\math\math_gen_0957ff.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\math\math_gen_1ed9c2.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\math\math_gen_265cce.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\math\math_gen_559593.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\math\math_gen_5e8458.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\math\math_gen_736506.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\math\math_gen_78ced2.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\math\math_gen_943d32.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\math\math_gen_a58d9d.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\math\math_intern_evaluator_gen_265cce.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\math\math_llm_judge_gen.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\math\math_llm_judge_gen_56606f.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\math\math_prm800k_500_0shot_cot_academic_gen.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\math\math_prm800k_500_0shot_cot_gen.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\math\math_prm800k_500_0shot_cot_gen_11c4b5.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\math\math_prm800k_500_0shot_nocot_gen_b27274.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\math\math_prm800k_500_0shot_nocot_genericllmeval_gen_63a000.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\math\math_prm800k_500_0shot_nocot_genericllmeval_gen_6ff468.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\math\math_prm800k_500_0shot_nocot_genericllmeval_xml_gen_63a000.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\math\math_prm800k_500_0shot_nocot_llmjudge_gen_63a000.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\math\math_prm800k_500_gen.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\math\math_prm800k_500_gen_393424.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\math\math_prm800k_500_llm_judge_gen.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\math\math_prm800k_500_llmverify_gen_6ff468.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\math\math_prm800k_500_llmverify_repeat4_gen_97b203.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mbpp\deprecated_mbpp_gen_1e1056.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mbpp\deprecated_mbpp_gen_6590b0.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mbpp\deprecated_mbpp_gen_caa7ab.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mbpp\deprecated_mbpp_passk_gen_1e1056.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mbpp\deprecated_mbpp_repeat10_gen_1e1056.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mbpp\deprecated_sanitized_mbpp_gen_1e1056.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mbpp\deprecated_sanitized_mbpp_gen_cb43ef.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mbpp\deprecated_sanitized_mbpp_passk_gen_1e1056.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mbpp\deprecated_sanitized_mbpp_repeat10_gen_1e1056.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mbpp\mbpp_gen.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mbpp\mbpp_gen_4shot.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mbpp\mbpp_gen_830460.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mbpp\mbpp_gen_base.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mbpp\mbpp_passk_gen_830460.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mbpp\mbpp_repeat10_gen_830460.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mbpp\mbpp_repeat_gen_18dd1b.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mbpp\sanitized_mbpp_gen_742f0c.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mbpp\sanitized_mbpp_gen_830460.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mbpp\sanitized_mbpp_gen_a0fc46.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mbpp\sanitized_mbpp_mdblock_0shot_nocot_gen_a2e416.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mbpp\sanitized_mbpp_mdblock_gen_a447ff.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mbpp\sanitized_mbpp_passk_gen_830460.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mbpp\sanitized_mbpp_repeat10_gen_830460.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mmlu\mmlu_all_sets.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mmlu\mmlu_clean_ppl.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mmlu\mmlu_gen.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mmlu\mmlu_gen_23a9a9.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mmlu\mmlu_gen_4d595a.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mmlu\mmlu_gen_5d1409.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mmlu\mmlu_gen_79e572.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mmlu\mmlu_gen_a484b3.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mmlu\mmlu_llm_judge_gen.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mmlu\mmlu_llmjudge_gen_f4336b.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mmlu\mmlu_model_postprocess_gen_4d595a.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mmlu\mmlu_openai_0shot_nocot_llmjudge_gen_216503.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mmlu\mmlu_openai_simple_evals_gen_b618ea.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mmlu\mmlu_ppl.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mmlu\mmlu_ppl_ac766d.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mmlu\mmlu_stem_0shot_cascade_eval_gen_216503.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mmlu\mmlu_stem_0shot_gen_216503.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mmlu\mmlu_stem_0shot_xml_gen_216503.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mmlu\mmlu_stem_sets.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mmlu\mmlu_xfinder_gen_4d595a.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mmlu\mmlu_zero_shot_gen_47e2c0.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mmlu_pro\mmlu_pro_0shot_cot_gen_08c1de.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mmlu_pro\mmlu_pro_0shot_nocot_genericllmeval_gen_08c1de.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mmlu_pro\mmlu_pro_biomed_0shot_cot_gen_057927.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mmlu_pro\mmlu_pro_biomed_0shot_nocot_genericllmeval_gen_057927.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mmlu_pro\mmlu_pro_categories.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mmlu_pro\mmlu_pro_few_shot_gen_bfaf90.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mmlu_pro\mmlu_pro_gen.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mmlu_pro\mmlu_pro_gen_cdbebf.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\datasets\mmlu_pro\mmlu_pro_llm_judge_gen.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\models\dllm\llada_15_instruct_8b.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\models\dllm\llada_base_8b.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\models\dllm\llada_instruct_8b.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\models\hf_llama\hf_llama2_13b.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\models\hf_llama\hf_llama2_13b_chat.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\models\hf_llama\hf_llama2_70b.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\models\hf_llama\hf_llama2_70b_chat.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\models\hf_llama\hf_llama2_7b.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\models\hf_llama\hf_llama2_7b_chat.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\models\hf_llama\hf_llama3_1_70b_instruct.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\models\hf_llama\hf_llama3_1_8b.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\models\hf_llama\hf_llama3_1_8b_instruct.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\models\hf_llama\hf_llama3_2_3b_instruct.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\models\hf_llama\hf_llama3_70b.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\models\hf_llama\hf_llama3_70b_instruct.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\models\hf_llama\hf_llama3_8b.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\models\hf_llama\hf_llama3_8b_instruct.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\models\hf_llama\hf_llama_13b.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\models\hf_llama\hf_llama_30b.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\models\hf_llama\hf_llama_65b.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\models\hf_llama\hf_llama_7b.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\models\hf_llama\lmdeploy_llama2_13b.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\models\hf_llama\lmdeploy_llama2_13b_chat.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\models\hf_llama\lmdeploy_llama2_70b.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\models\hf_llama\lmdeploy_llama2_70b_chat.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\models\hf_llama\lmdeploy_llama2_7b.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\models\hf_llama\lmdeploy_llama2_7b_chat.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\models\hf_llama\lmdeploy_llama3_1_70b_instruct.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\models\hf_llama\lmdeploy_llama3_1_8b.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\models\hf_llama\lmdeploy_llama3_1_8b_instruct.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\models\hf_llama\lmdeploy_llama3_2_3b_instruct.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\models\hf_llama\lmdeploy_llama3_3_70b_instruct.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\models\hf_llama\lmdeploy_llama3_70b.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\models\hf_llama\lmdeploy_llama3_70b_instruct.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\models\hf_llama\lmdeploy_llama3_8b.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\models\hf_llama\lmdeploy_llama3_8b_instruct.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\models\hf_llama\lmdeploy_llama_13b.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\models\hf_llama\lmdeploy_llama_30b.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\models\hf_llama\lmdeploy_llama_65b.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\models\hf_llama\lmdeploy_llama_7b.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\models\hf_llama\vllm_llama_series.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\summarizers\groups\bbh.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\summarizers\groups\mathbench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\summarizers\groups\mmlu.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\configs\summarizers\groups\mmlu_pro.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\advglue.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `AdvDataset`
    Metodo: `__init__(self:Any, subset:str, filter_keys:Union[str, List[str]], **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `aug_with_original_data(self:Any, dataset:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `load(self:Any, path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `AdvSst2Dataset`
    Metodo: `__init__(self:Any, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `AdvQqpDataset`
    Metodo: `__init__(self:Any, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `AdvMnliDataset`
    Metodo: `__init__(self:Any, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `AdvMnliMMDataset`
    Metodo: `__init__(self:Any, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `AdvQnliDataset`
    Metodo: `__init__(self:Any, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `AdvRteDataset`
    Metodo: `__init__(self:Any, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `AccDropEvaluator`
    Metodo: `__init__(self:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:List, references:List) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\afqmcd.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `AFQMCDatasetV2`
    Metodo: `load(path:Any, local_mode:Any=False) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\agieval\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\agieval\agieval.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `AGIEvalDataset`
    Metodo: `load(path:str, name:str, setting_name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `AGIEvalDataset_v2`
    Metodo: `load(path:str, name:str, setting_name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `AGIEvalEvaluator`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `AGIEvalEvaluator_mcq`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\agieval\constructions.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `TaskSchema`
    Metodo: `__init__(self:Any, passage:Any=None, question:Any=None, options:Any=None, label:Any=None, answer:Any=None, other:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `to_dict(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `AgiInstance`
    Metodo: `__init__(self:Any, task_description:Any, data_source:Any, task_schema:Any, output:Any, evaluation_metric:Any, task_example:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `to_dict(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `ChatGPTSchema`
    Metodo: `__init__(self:Any, context:Any=None, metadata:Any='') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `to_dict(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `ResultsForHumanSchema`
    Metodo: `__init__(self:Any, index:Any, problem_input:Any, label:Any, model_input:Any='', model_output:Any='', parse_result:Any='', first_stage_output:Any='', second_stage_input:Any='', is_correct:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `to_dict(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `to_tsv(result_list:Any, path:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\agieval\dataset_loader.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `convert_zero_shot(line:Any, dataset_name:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `convert_zero_shot_CoT_stage1(line:Any, dataset_name:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `combine_prompt(prompt_path:Any, dataset_name:Any, load_explanation:Any=True, chat_mode:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_lazy_load_enc() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `concat_prompt(demos:Any, dataset_name:Any, max_tokens:Any, end_of_example:Any='\n', verbose:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `concat_prompt_chat_mode(demos:Any, dataset_name:Any, max_tokens:Any, end_of_example:Any='\n', verbose:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `convert_few_shot(line:Any, dataset_name:Any, demo:Any, n_shot:Any, chat_mode:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `load_dataset(dataset_name:Any, setting_name:Any, parent_path:Any, prompt_path:Any=None, max_tokens:Any=None, end_of_example:Any='\n', chat_mode:Any=False, verbose:Any=False) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `generate_second_stage_input(dataset_name:Any, input_list:Any, output_list:Any, with_format_prompt:Any=False) -> Any`
    Descrizione: Genera output testo o sequenze.
  - `load_dataset_as_result_schema(dataset_name:Any, parent_path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\agieval\evaluation.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `convert_to_set(item:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `evaluate_single_sample(dataset_name:Any, prediction:Any, label:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\agieval\math_equivalence.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_fix_fracs(string:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_fix_a_slash_b(string:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_remove_right_units(string:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_fix_sqrt(string:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_strip_string(string:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `is_equiv(str1:Any, str2:Any, verbose:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\agieval\post_process.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `extract_last_line(string:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `remove_few_shot_prefix(string:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `try_parse_few_shot_qa_single_answer(string:Any, setting_name:Any, language:Any='en') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `try_parse_few_shot_pattern(string:str, dataset_name:Any, setting_name:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `parse_few_shot_qa_single_answer(string:Any, setting_name:Any, language:Any='en') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `find_first_capital_letter(answer:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `extract_answer_in_bracket(answer:Any, prefix:Any='【', suffix:Any='】') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `parse_math_answer(setting_name:Any, raw_string:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `parse_qa_multiple_answer(string:Any, setting_name:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `post_process(dataset_name:Any, setting_name:Any, prediction:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\agieval\utils.py`
  Logica d'uso: Runtime helpers, device/dtype control, tiny fallback model/tokenizer.
  Funzioni:
  - `read_jsonl(path:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `save_jsonl(lines:Any, directory:Any) -> Any`
    Descrizione: Serializza e salva output su disco.
  - `extract_answer(js:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\aime2024.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `Aime2024Dataset`
    Metodo: `load(path:Any, **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\anli.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `AnliDataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\anthropics_evals.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `AiRiskDataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `PersonaDataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `SycophancyDataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\apps.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `timeout_handler(signum:Any, frame:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `run_test(sample:Any, test:Any=None, debug:Any=False) -> Any`
    Descrizione: Esegue pipeline o job completo.
  - `custom_compare_(output:Any, ground_truth:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `stripped_string_compare(s1:Any, s2:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `call_method(method:Any, inputs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `reliability_guard(maximum_memory_bytes:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `APPSDataset`
    Metodo: `load(path:str, num_repeats:int=1) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `APPS_miniDataset`
    Metodo: `load(path:str, num_repeats:int=1) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `APPSEvaluator`
    Metodo: `post_process(self:Any, text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `check_correctness(self:Any, sample:Any, generation:Any, timeout:Any, debug:Any=True) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `evaluate_generations(self:Any, generations:Any, samples:Any, idx:Any=None, debug:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `estimate_pass_at_k(self:Any, num_samples:Any, num_correct:Any, k:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `compute_metrics(self:Any, results:Any, k_list:Any=[1, 10, 100]) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any, test_set:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `CODE_TYPE`
  - `TimeoutException`
  - `Capturing`
    Metodo: `__enter__(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__exit__(self:Any, *args:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\arc.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `ARCDataset`
    Metodo: `load(path:str, name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `ARCDatasetClean`
    Metodo: `load_contamination_annotations(path:Any, split:Any='val') -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
    Metodo: `load(path:str, name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\arc_prize_public_evaluation.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `extract_solution(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `pad_array_with_value(array:Any, target_shape:Any, pad_value:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `compare_solutions_with_padding(generated_output:List[int], correct_output:List[int], pad_value:Any=-1) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `ARCPrizeDataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `ARCPrizeEvaluator`
    Metodo: `score(self:Any, predictions:List[str], references:List[List[int]]) -> Dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\ax.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `AXDatasetV2`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\babilong\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\babilong\babilong.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `BabiLongDataset`
    Metodo: `load(path:Any, task:Any, split_name:Any, use_instruction:Any=True, use_examples:Any=True, use_post_prompt:Any=True) -> Dataset`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `BabiLongEvaluator`
    Metodo: `score(self:Any, predictions:Any, gold:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\babilong\babilong_utils.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `compare_answers(target:Any, output:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_dataset_df(dataset_path:Any, max_n_facts:Any=None) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `sum_lengths(sentences:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `TaskDataset`
    Metodo: `__init__(self:Any, dataset_path:Any, max_n_facts:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__getitem__(self:Any, ind:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__len__(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `SentenceSampler`
    Metodo: `__init__(self:Any, dataset:Any, tokenizer:Any, min_sentence_len:Any=10, max_sentence_len:Any=None, shuffle:Any=False, random_seed:Any=42) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_sample(self:Any, sample_size:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `sample_sentences_(self:Any, sample_size:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `next_sample_(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `length_is_ok(self:Any, tokenized:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `NoiseInjectionDataset`
    Metodo: `__init__(self:Any, task_dataset:Any, noise_sampler:Any, tokenizer:Any, task_start_pct:Any=None, task_end_pct:Any=None, sample_size:Any=1024, mixed_length_ratio:Any=0.0, random_seed:Any=42) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__getitem__(self:Any, ind:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__len__(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_sample_size(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\babilong\prompts.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_formatted_input(context:Any, question:Any, examples:Any, instruction:Any, post_prompt:Any, template:Any=DEFAULT_TEMPLATE) -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\base.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `BaseDataset`
    Metodo: `__init__(self:Any, reader_cfg:Optional[Dict]={}, k:Union[int, List[int]]=1, n:int=1, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_init_reader(self:Any, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `train(self:Any) -> Any`
    Descrizione: Esegue passaggi di training/update.
    Metodo: `test(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `load(**kwargs:Any) -> Union[Dataset, DatasetDict]`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\bbeh.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `bbeh_freeform_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `bbeh_mcq_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `BBEHDataset`
    Metodo: `load(path:str, name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `BBEHEvaluator`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `BBEHEvaluator_mcq`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\bbh.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `bbh_mcq_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `bbh_freeform_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `BBHDataset`
    Metodo: `load(path:str, name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `BBHEvaluator`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `BBHEvaluator_mcq`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\benbench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `exact_match_score(predicted_text:Any, original_text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `edit_similarity_score(predicted_text:Any, original_text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `rouge_l_score(predicted_text:Any, original_text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `BenBenchDataset`
    Metodo: `load(path:str, tokenizer_path:str, tokenizer_kwargs:Optional[Dict]=dict(), num_gram:int=5, num_replica:int=5) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `BenbenEvaluator`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\bigcodebench\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\bigcodebench\bigcodebench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `BigCodeBenchDataset`
    Metodo: `load(path:str='opencompass/bigcodebench', local_mode:bool=False, release_version:str='v0.1.2', dataset_version:str='full') -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `BigCodeBenchEvaluator`
    Metodo: `__init__(self:Any, release_version:Any='v0.1.2', eval_type:Any='instruct', remote_execute_api:Any='https://bigcode-bigcodebench-evaluator.hf.space/', dataset_version:str='full', local_mode:bool=False, path:str='opencompass/bigcodebench', pass_k:str='1,5,10', parallel:int=-1, min_time_limit:float=1, max_as_limit:int=30 * 1024, max_data_limit:int=30 * 1024, max_stack_limit:int=10, check_gt_only:bool=False, no_gt:bool=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_results_processor(self:Any, results:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\bigcodebench\extractor.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `syntax_check(code:Any, verbose:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `code_extract(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_deps(nodes:List[Tuple[str, Node]]) -> Dict[str, Set[str]]`
    Descrizione: Recupera valore/stato calcolato.
  - `get_function_dependency(entrypoint:str, call_graph:Dict[str, str]) -> Set[str]`
    Descrizione: Recupera valore/stato calcolato.
  - `get_definition_name(node:Node) -> str`
    Descrizione: Recupera valore/stato calcolato.
  - `traverse_tree(node:Node) -> Generator[Node, None, None]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `has_return_statement(node:Node) -> bool`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `extract_target_code_or_empty(code:str, entrypoint:Optional[str]=None) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `extract_code_generation(model_output:str, entrypoint:Optional[str]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\boolq.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `BoolQDataset`
    Metodo: `load(**kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `BoolQDatasetV2`
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `BoolQDatasetV3`
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\bustum.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `bustumDataset_V2`
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\c3.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `C3Dataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `C3Dataset_V2`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\calm.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `CaLMDataset`
    Metodo: `load(path:str, prompt_style:str) -> datasets.Dataset`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `CaLMEvaluator`
    Metodo: `__init__(self:Any, core_metrics:Any, error_analysis:Any, prompt_style:Any, task:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:List, references:List) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\data_processing\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\data_processing\generate_questions.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_get_prompt_func(task:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `generate_question_list(dataset_path:Any, prompt_style:Any) -> Any`
    Descrizione: Genera output testo o sequenze.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\data_processing\prompt\AC-B_causal_judgement.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_prompt(task_name:Any, prompt_style:Any, item:Any, prompt_style_str:Any='') -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\data_processing\prompt\AR-B_CaLM-AR.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_prompt(task_name:Any, prompt_style:Any, item:Any, prompt_style_str:Any='') -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\data_processing\prompt\ATE.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_prompt(task_name:Any, prompt_style:Any, item:Any, prompt_style_str:Any='') -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\data_processing\prompt\BAS-B_backadj.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_prompt(task_name:Any, prompt_style:Any, item:Any, prompt_style_str:Any='') -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\data_processing\prompt\BAS-C_max-BAS.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_prompt(task_name:Any, prompt_style:Any, item:Any, prompt_style_str:Any='') -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\data_processing\prompt\BAS-C_min-BAS.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_prompt(task_name:Any, prompt_style:Any, item:Any, prompt_style_str:Any='') -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\data_processing\prompt\BAS-C_mix-BAS.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_prompt(task_name:Any, prompt_style:Any, item:Any, prompt_style_str:Any='') -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\data_processing\prompt\CA-B_FA.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_prompt(task_name:Any, prompt_style:Any, item:Any, prompt_style_str:Any='') -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\data_processing\prompt\CA-B_FP.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_prompt(task_name:Any, prompt_style:Any, item:Any, prompt_style_str:Any='') -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\data_processing\prompt\CB-B_collider-bias.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_prompt(task_name:Any, prompt_style:Any, item:Any, prompt_style_str:Any='') -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\data_processing\prompt\CDE.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_prompt(task_name:Any, prompt_style:Any, item:Any, prompt_style_str:Any='') -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\data_processing\prompt\CEG-O_E-CARE.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_prompt(task_name:Any, prompt_style:Any, item:Any, prompt_style_str:Any='') -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\data_processing\prompt\CEI-B.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_prompt(task_name:Any, prompt_style:Any, item:Any, prompt_style_str:Any='') -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\data_processing\prompt\CORR-B_correlation.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_prompt(task_name:Any, prompt_style:Any, item:Any, prompt_style_str:Any='') -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\data_processing\prompt\CR-B_det-counterfactual.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_prompt(task_name:Any, prompt_style:Any, item:Any, prompt_style_str:Any='') -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\data_processing\prompt\CR-C_CRASS.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_prompt(task_name:Any, prompt_style:Any, item:Any, prompt_style_str:Any='') -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\data_processing\prompt\EAE-B_exp-away.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_prompt(task_name:Any, prompt_style:Any, item:Any, prompt_style_str:Any='') -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\data_processing\prompt\ECI-B_CTB.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_prompt(task_name:Any, prompt_style:Any, item:Any, prompt_style_str:Any='') -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\data_processing\prompt\ECI-B_ESC.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_prompt(task_name:Any, prompt_style:Any, item:Any, prompt_style_str:Any='') -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\data_processing\prompt\ECI-B_MAVEN-ERE.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_prompt(task_name:Any, prompt_style:Any, item:Any, prompt_style_str:Any='') -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\data_processing\prompt\ETT.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_prompt(task_name:Any, prompt_style:Any, item:Any, prompt_style_str:Any='') -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\data_processing\prompt\FAS-C_FAS.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_prompt(task_name:Any, prompt_style:Any, item:Any, prompt_style_str:Any='') -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\data_processing\prompt\IV-C_CaLM-IV.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_prompt(task_name:Any, prompt_style:Any, item:Any, prompt_style_str:Any='') -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\data_processing\prompt\NDE.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_prompt(task_name:Any, prompt_style:Any, item:Any, prompt_style_str:Any='') -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\data_processing\prompt\NIE.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_prompt(task_name:Any, prompt_style:Any, item:Any, prompt_style_str:Any='') -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\data_processing\prompt\PCD-B_COPA.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_prompt(task_name:Any, prompt_style:Any, item:Any, prompt_style_str:Any='') -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\data_processing\prompt\PCD-B_E-CARE.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_prompt(task_name:Any, prompt_style:Any, item:Any, prompt_style_str:Any='') -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\data_processing\prompt\PCD-C_COPA.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_prompt(task_name:Any, prompt_style:Any, item:Any, prompt_style_str:Any='') -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\data_processing\prompt\PCD-C_E-CARE.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_prompt(task_name:Any, prompt_style:Any, item:Any, prompt_style_str:Any='') -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\data_processing\prompt\PN.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_prompt(task_name:Any, prompt_style:Any, item:Any, prompt_style_str:Any='') -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\data_processing\prompt\PS.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_prompt(task_name:Any, prompt_style:Any, item:Any, prompt_style_str:Any='') -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\data_processing\task_hiearchy.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\evaluation\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\evaluation\accuracy\choice.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `compute_acc(gt_list:Any, pred_list:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\evaluation\accuracy\open-ended.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `is_chinese(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `compute_acc(gt_list:Any, pred_list:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\evaluation\accuracy\prob.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `compute_acc(gt_list:Any, pred_list:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\evaluation\core_metrics.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `initialize_core_metric_evaluation_components(task:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `compute_core_metrics(items:Any, task:Any, prompt_style:Any, gt_items:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\evaluation\error\basic_adversarial\AC-B_causal_judgement.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `check_standalization(model_response:Any, prompt_style:Any, type:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `check_empty(model_response:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `check_repetition(model_response:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `contains_chinese(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `contains_english(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `check_abnormality(preds:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\evaluation\error\basic_adversarial\AR-B_CaLM-AR.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `check_standalization(model_response:Any, prompt_style:Any, type:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `check_empty(model_response:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `check_repetition(model_response:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `contains_chinese(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `contains_english(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `check_abnormality(preds:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\evaluation\error\basic_adversarial\AS.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `check_standalization(model_response:Any, prompt_style:Any, type:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `check_empty(model_response:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `check_repetition(model_response:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `contains_chinese(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `contains_english(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `check_abnormality(preds:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\evaluation\error\basic_adversarial\CA-B.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `check_standalization(model_response:Any, prompt_style:Any, type:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `check_empty(model_response:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `check_repetition(model_response:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `contains_chinese(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `contains_english(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `check_abnormality(preds:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\evaluation\error\basic_adversarial\CEI-B.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `check_standalization(model_response:Any, prompt_style:Any, type:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `check_empty(model_response:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `check_repetition(model_response:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `contains_chinese(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `contains_english(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `check_abnormality(preds:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\evaluation\error\basic_adversarial\CLADDER.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `check_standalization(model_response:Any, prompt_style:Any, type:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `check_empty(model_response:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `check_repetition(model_response:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `contains_chinese(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `contains_english(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `check_abnormality(preds:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\evaluation\error\basic_adversarial\CR-C_CRASS.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `check_standalization(model_response:Any, prompt_style:Any, type:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `check_empty(model_response:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `check_repetition(model_response:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `contains_chinese(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `contains_english(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `check_abnormality(preds:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\evaluation\error\basic_adversarial\ECI.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `check_standalization(model_response:Any, prompt_style:Any, type:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `check_empty(model_response:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `check_repetition(model_response:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `contains_chinese(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `contains_english(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `check_abnormality(preds:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\evaluation\error\basic_adversarial\Natural.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `check_standalization(model_response:Any, prompt_style:Any, type:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `check_empty(model_response:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `check_repetition(model_response:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `contains_chinese(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `contains_english(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `check_abnormality(preds:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\evaluation\error\basic_adversarial\PCD-B.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `check_standalization(model_response:Any, prompt_style:Any, type:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `check_empty(model_response:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `check_repetition(model_response:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `contains_chinese(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `contains_english(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `check_abnormality(preds:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\evaluation\error\basic_adversarial\PCD-C.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `check_standalization(model_response:Any, prompt_style:Any, type:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `check_empty(model_response:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `check_repetition(model_response:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `contains_chinese(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `contains_english(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `check_abnormality(preds:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\evaluation\error\basic_adversarial\Probability.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `check_standalization(model_response:Any, prompt_style:Any, type:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `check_empty(model_response:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `check_repetition(model_response:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `contains_chinese(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `contains_english(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `check_abnormality(preds:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\evaluation\errors.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `initialize_error_identification_components(task:Any, prompt_style:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `identify_model_errors(items:Any, task:Any, prompt_style:Any, gt_items:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_item_error(model_response:Any, task:Any, error_module:Any, prompt_style:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\evaluation\labeling\AC-B_causal_judgement.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_gt_label(item:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `get_pred_label(model_response:Any, item:Any, prompt_style:Any, type:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\evaluation\labeling\AR-B_CaLM-AR.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_gt_label(item:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `get_pred_label(model_response:Any, item:Any, prompt_style:Any, type:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\evaluation\labeling\AS.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_gt_label(item:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `get_pred_label(model_response:Any, item:Any, prompt_style:Any, type:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\evaluation\labeling\CA-B_FA.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_gt_label(item:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `get_pred_label(model_response:Any, item:Any, prompt_style:Any, type:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\evaluation\labeling\CA-B_FP.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_gt_label(item:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `get_pred_label(model_response:Any, item:Any, prompt_style:Any, type:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\evaluation\labeling\CEG-O_E-CARE.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_gt_label(item:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `get_pred_label(model_response:Any, item:Any, prompt_style:Any, type:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\evaluation\labeling\CEI-B.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_gt_label(item:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `get_pred_label(model_response:Any, item:Any, prompt_style:Any, type:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\evaluation\labeling\CLADDER.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_gt_label(item:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `get_pred_label(model_response:Any, item:Any, prompt_style:Any, type:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\evaluation\labeling\common_answers.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `is_numeric(value:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `add_quotes_to_unquoted(json_str:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `change_quotation(json_str:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\evaluation\labeling\CR-C_CRASS.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_gt_label(item:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `get_pred_label(model_response:Any, item:Any, prompt_style:Any, type:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\evaluation\labeling\ECI.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_gt_label(item:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `get_pred_label(model_response:Any, item:Any, prompt_style:Any, type:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\evaluation\labeling\Natural.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_gt_label(item:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `extract_answer(model_response:Any, item:Any, prompt_style:Any, type:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_pred_label(model_response:Any, item:Any, prompt_style:Any, type:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\evaluation\labeling\PCD-B.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_gt_label(item:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `get_pred_label(model_response:Any, item:Any, prompt_style:Any, type:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\evaluation\labeling\PCD-C.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_gt_label(item:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `get_pred_label(model_response:Any, item:Any, prompt_style:Any, type:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\evaluation\labeling\Probability.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_gt_label(item:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `extract_prob(model_response:Any, prompt_style:Any, type:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_pred_label(model_response:Any, item:Any, prompt_style:Any, type:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\utils\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\calm\utils\load_items.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `load_query_instances(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\CARDBiomedBench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_parse(item:Any, prompt_mode:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `CARDBiomedBenchDataset`
    Metodo: `load(path:str, prompt_mode:str, **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\cb.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `CBDatasetV2`
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\ceval.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `CEvalDataset`
    Metodo: `load(path:str, name:str, local_mode:bool=False) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `CEvalDatasetClean`
    Metodo: `load_contamination_annotations(path:Any, split:Any='val') -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
    Metodo: `load(path:str, name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\charm.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `charm_reason_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `charm_memory_eval(pred:str, ref:Union[str, List[str]]) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `CharmReasonEvaluator`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `CharmMemoryEvaluator`
    Metodo: `__init__(self:Any, prompt_template:Any=None, *nargs:Any, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `CharmDataset`
    Metodo: `load(path:str, name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\chem_exam.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `chem_exam_score_llmjudge_postprocess(output:Any, output_path:Any, dataset:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `ChemExamDataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\chembench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `ChemBenchDataset`
    Metodo: `load(path:str, name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\chid.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `CHIDDataset`
    Metodo: `load(**kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `CHIDDatasetV2`
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\chinese_simpleqa.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `chinese_simpleqa_preprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `post_process_csimpleqa(completion:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_judgeanswer_and_reference(result:Any, filename:Any, post_process:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `calculate_metrics(judged_answers:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_results(judged_answers:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `csimpleqa_postprocess(output:dict, output_path:str) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `CsimpleqaDataset`
    Metodo: `load(self:Any, path:str, name:str, *args:Any, **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\cibench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `load_experiment(file:str) -> dict`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `check_internet() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `sklearn_ssim(pred_img:Any, target_img:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `vl_model_score(model:Any, pred_img:Any, ori_prompt:Any, judge_prompt:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `CIBenchDataset`
    Metodo: `load(path:str, internet_check:bool=False) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `CIBenchEvaluator`
    Metodo: `__init__(self:Any, text_evaluator:Optional[dict]=None, vis_evaluator:Optional[dict]=None, output_dir:Optional[str]=None, with_ipynb:bool=False, lang:str='en', user_data_dir:str='ENV') -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `check_user_data_dir(self:Any, user_data_dir:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `valid_step(step:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `correct_step(step:Any, target:Any) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `text_step(self:Any, step:Any, target:Any) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `vis_similarity_step(self:Any, step:Any, target:Any, ori_prompt:Any) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `save_results(self:Any, origin_prompt:Any, steps:Any, references:Any) -> Any`
    Descrizione: Serializza e salva output su disco.
    Metodo: `set_data_dir(self:Any, work_dir:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `unset_data_dir(self:Any, work_dir:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `single_exp(self:Any, gold:Any, steps:Any, single_ori_prompt:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_output_dir(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `score(self:Any, predictions:List, references:List, steps:List, origin_prompt:List) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\circular.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_origin_patterns(option_keys:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `get_circular_patterns(option_keys:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `get_all_possible_patterns(option_keys:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  Classi:
  - `CircularDatasetMeta`
    Metodo: `make_circular_items(origin_item:Any, circular_patterns:Any, option_keys:Any, answer_key:Any, answer_key_switch_method:Any, qid:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `make_circular_dataset(dataset:Any, circular_patterns:Any, option_keys:Any, answer_key:Any, answer_key_switch_method:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `make_circular(dataset:Union[Dataset, DatasetDict], circular_splits:Optional[List[str]]=['test'], circular_patterns:str='circular', option_keys:List[str]=['A', 'B', 'C', 'D'], answer_key:Optional[str]='answer', answer_key_switch_method:Optional[Callable]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__new__(cls:Any, name:Any, bases:Any, dct:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `CircularCEvalDataset`
  - `CircularMMLUDataset`
  - `CircularCMMLUDataset`
  - `CircularCSQADataset`
  - `CircularARCDataset`
    Metodo: `default_answer_key_switch_method(item:Any, circular_pattern:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `CircularHSWAGDataset`
  - `CircularOBQADataset`
  - `CircularRaceDataset`
  - `CircularXiezhiDataset`
  - `CircularsiqaDataset`
  - `CircularPIQADataset`
    Metodo: `default_answer_key_switch_method(item:Any, circular_pattern:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `CircularEvaluator`
    Metodo: `__init__(self:Any, circular_pattern:Any='circular') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any, test_set:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\civilcomments.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `CivilCommentsDataset`
    Metodo: `load(**kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\climaqa.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `ClimaQADataset`
    Metodo: `load(path:str, task:str, **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\ClinicBench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `ClinicBenchDataset`
    Metodo: `load_single(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\clozeTest_maxmin.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `MaxminDataset`
    Metodo: `load(test_path:Any, answer_path:Any=None) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\cluewsc.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `CluewscDataset`
    Metodo: `load(**kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `CluewscDatasetV2`
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\cmb.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `CMBDataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\cmmlu.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `CMMLUDataset`
    Metodo: `load(path:str, name:str, **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\cmnli.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `CMNLIDataset`
    Metodo: `load(path:Any, local_mode:bool=False) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `CMNLIDatasetV2`
    Metodo: `load(path:Any, local_mode:bool=False) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\cmo_fib.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `CMOFibDataset`
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\cmrc.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `cmrc_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `CMRCDataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\codecompass\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\codecompass\CodeCompass.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `CodeCompassCodeGenerationDataset`
    Metodo: `load(path:str='opencompass/CodeCompass', difficulty:Optional[str]=None, source:Optional[str]=None, system_prompt:Optional[str]=None, problem_template:Optional[str]=None) -> DatasetDict`
    Descrizione: Carica dati o stato da sorgente esterna/file.
    Metodo: `_extract_limits(problem_text:str) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_process_item(item:Dict[str, Any], system_prompt:str, problem_template:str) -> Optional[Dict[str, Any]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_create_evaluation_sample(item:Dict[str, Any]) -> Optional[Dict[str, Any]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `validate_dataset(dataset:DatasetDict) -> bool`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\codecompass\codecompass_runner.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `run_test_for_cpp_problem(sample:dict, generations:list, timeout:int, memory_limit_mb:int, temp_base_dir:Any='tmp') -> list`
    Descrizione: Esegue pipeline o job completo.

- File: `vendor\llada\opencompass\opencompass\datasets\codecompass\evaluator.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `CodeCompassEvaluator`
    Metodo: `__init__(self:Any, num_process_evaluate:int=16, timeout:int=15, k_list:List[int]=None, temp_base_dir:str=None, dataset_path:str=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_build_results(self:Any, extracted_predictions:Dict[int, List[str]], metrics:Dict[str, float], eval_results:Dict[int, List[List[int]]], final_metadata:List[Dict]) -> Dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:List[Any], references:List[Any]) -> Dict[str, float]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_prepare_sample(self:Any, reference:Any, idx:int=-1) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_run_parallel_evaluation(self:Any, tasks:List[tuple]) -> List[List[List[int]]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\codecompass\executor.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LocalExecutor`
    Metodo: `__init__(self:Any, timeout:int=10, memory_limit_mb:int=512, temp_base_dir:Optional[str]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_set_resource_limits(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_compile_cpp(self:Any, source_file:Path, temp_dir:Path) -> tuple`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_run_executable(self:Any, exec_file:Path, stdin_data:str) -> Dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `execute_code(self:Any, source_code:str, stdin:str, language:str, temp_dir:Path) -> Dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `verify_output(self:Any, result:Dict, expected_output:str) -> Dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `submit_code(self:Any, source_code:str, stdin:str, expected_output:str, language:str='C++') -> Dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\codecompass\metrics.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `estimate_pass_at_k(num_samples:Any, num_correct:Any, k:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `compute_metrics_from_results(results:dict, k_list:Any=[1]) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\codecompass\utils.py`
  Logica d'uso: Runtime helpers, device/dtype control, tiny fallback model/tokenizer.
  Funzioni:
  - `extract_cpp_code(model_output:str, model_type:str='chat') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `extract_cpp_code_with_debug(model_output:str, model_type:str='chat') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\commonsenseqa.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `commonsenseqaDataset`
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\commonsenseqa_cn.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `CommonsenseQADataset_CN`
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\compassbench_obj.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_number(options:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `compassbench_objective_v1_3_postprocess(text:str, name:Any) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `CompassBenchObjectiveV1_3`
    Metodo: `load(path:str, name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `CompassBenchObjectiveMath`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\copa.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `COPADatasetV2`
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\crowspairs.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `crowspairs_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `CrowspairsDataset`
    Metodo: `load(**kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `CrowspairsDatasetV2`
    Metodo: `load(**kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `CrowspairsEvaluator`
    Metodo: `__init__(self:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:List, references:List) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\crowspairs_cn.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `CrowspairsDatasetCN`
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\csl.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `CslDataset`
    Metodo: `load(**kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `CslDatasetV2`
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\custom.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `stringfy_types(obj:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `make_mcq_gen_config(meta:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `make_circular_mcq_gen_config(meta:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `make_qa_gen_config(meta:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `make_mcq_ppl_config(meta:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `make_circular_mcq_ppl_config(meta:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `parse_example_dataset(config:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `make_custom_dataset_config(config:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `OptionSimAccEvaluator`
    Metodo: `__init__(self:Any, options:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `match_any_label(self:Any, pred:Any, test_item:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:List, references:List, test_set:Any) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `CircularOptionSimAccEvaluator`
    Metodo: `__init__(self:Any, options:Any, circular_pattern:Any='circular') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any, test_set:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `CustomDataset`
    Metodo: `load(path:Any, file_name:Any=None, local_mode:Any=False) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `CodeCustomDataset`
    Metodo: `load(path:Any, file_name:Any=None, local_mode:Any=False, num_repeats:Any=1, **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `CircularCustomDataset`

- File: `vendor\llada\opencompass\opencompass\datasets\cvalues.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `CValuesDataset`
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\dingo.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `DingoDataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `DingoLongDataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `DingoEvaluator`
    Metodo: `score(self:Any, origin_prompt:List, predictions:List) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\drcd.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `drcd_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `DRCDDataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\drop.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `dropDataset`
    Metodo: `get_answers(validated_answers:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `load(path:Any, only_number:Any=True) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\drop_simple_eval.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `normalize(s:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `fuzzy_match(s1:str, s2:str) -> bool`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `DropOpenAIDataset`
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `DropOpenAIEvaluator`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\ds1000.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `ds1000_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `ds1000_completion_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `ds1000_matplotlib_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `import_source_file(fname:Any, modname:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `DS1000Dataset`
    Metodo: `get_data(self:Any, problem_path:str) -> dict`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `load(self:Any, path:str, libs:Optional[Union[str, list]]=None, mode:str='Insertion') -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `DS1000Evaluator`
    Metodo: `__init__(self:Any, num_workers:Any=16) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score_single(self:Any, pred:Any, refer:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `Command`
    Metodo: `__init__(self:Any, cmd:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `run(self:Any, timeout:Any) -> Any`
    Descrizione: Esegue pipeline o job completo.
  - `DS1000ServiceEvaluator`
    Metodo: `__init__(self:Any, lib:str, ip_address:Any='localhost', port:Any='', timeout:Any=600) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_code_eval_service(self:Any, file_path:str) -> tuple`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\ds1000_interpreter.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `DS1000Dataset_Interperter`
    Metodo: `load(self:Any, path:str, libs:Optional[Union[str, list]]=None, mode:str='Insertion') -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `DS1000InterpreterEvaluator`
    Metodo: `__init__(self:Any, action:str='PythonInterpreter') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_action(self:Any, step:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `score(self:Any, predictions:List, references:List, steps:List) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\Earth_Silver.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `Earth_Silver_MCQDataset`
    Metodo: `load(path:str, prompt_mode:str='zero-shot', **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\eese\eese.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `EESEDataset`
    Metodo: `load(path:str, file_name:str='EESE.jsonl', **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\eese\eese_postprocessors.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `eese_score_postprocess_dict(output:dict, output_path:str) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\eese\utils.py`
  Logica d'uso: Runtime helpers, device/dtype control, tiny fallback model/tokenizer.
  Funzioni:
  - `extract_first_numeric_score(score_text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `process_results(results:Any, overall_avg:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\eprstmt.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `EprstmtDatasetV2`
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\FinanceIQ.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `FinanceIQDataset`
    Metodo: `load(path:str, name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\flores.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `flores_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `flores_postprocess_chinese(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `FloresFirst100Dataset`
    Metodo: `load_single(src_path:Any, tgt_path:Any, src_lang:Any, tgt_lang:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
    Metodo: `load(path:Any, name:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\game24.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_current_numbers(y:str) -> str`
    Descrizione: Recupera valore/stato calcolato.
  - `game24_postprocess(output:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `Game24Dataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `Game24PromptWrapper`
    Metodo: `__init__(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `standard_prompt_wrap(x:str, y:str='') -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `cot_prompt_wrap(x:str, y:str='') -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `propose_prompt_wrap(x:str, y:str='') -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `value_prompt_wrap(x:str, y:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `value_outputs_unwrap(x:str, y:str, value_outputs:list) -> float`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `Game24Evaluator`
    Metodo: `__init__(self:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `check_nums(self:Any, prediction:Any, reference:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:List, references:List) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\gaokao_math.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `extract_boxed_answer(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `GaoKaoMATHDataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `GaoKaoMATHEvaluator`
    Metodo: `__init__(self:Any, model_name:Any, url:Any, question_type:Any=None, language:Any='en', with_postprocess:Any=False, post_url:Any=[], post_model_name:Any='', **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `batch_response(self:Any, models:Any, inputs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `postprocess(self:Any, questions:Any, predictions:Any, question_type:Any='None') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any, origin_prompt:Any, test_set:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\GaokaoBench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `GaokaoBenchDataset`
    Metodo: `load(path:str, filename:str, name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `GaokaoBenchEvaluator`
    Metodo: `__init__(self:Any, question_type:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `do_predictions_postprocess(self:Any, model_output:Any, answer_lenth:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `ensure_same_length(self:Any, pred:Any, refr:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\generic.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_final_results(judged_answers:Any, references:Any, origial_responses:Any, metric_name:Any='accuracy') -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `_generic_llmjudge_postprocess(judgement:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `generic_llmjudge_postprocess(output:dict, output_path:str) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `generic_llmjudge_academic_postprocess(output:dict, output_path:str, metric_name:str='accuracy') -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\govrepcrs.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `GovRepcrsDataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\gpqa.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `GPQA_Simple_Eval_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `GPQA_Simple_Eval_postprocess_(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `GPQADataset`
    Metodo: `load(path:str, name:str, **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `GPQADataset_`
    Metodo: `load(path:str, name:str, seed:int=0, **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `GPQAEvaluator`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `GPQASimpleEvalDataset`
    Metodo: `load(path:str, name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\gsm8k.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `gsm8k_dataset_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `gsm8k_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `gsm8k_postprocess_(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `GSM8KDataset`
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `Gsm8kEvaluator`
    Metodo: `is_equal(self:Any, pred:Any, refer:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `Gsm8kAgentEvaluator`
    Metodo: `__init__(self:Any, action:str='PythonInterpreter') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `is_equal(self:Any, pred:Any, refer:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `soft_equal(self:Any, pred:Any, refer:Any, step:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_action(self:Any, step:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `score(self:Any, predictions:Any, references:Any, steps:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\gsm_hard.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `GSMHardDataset`
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\healthbench\healthbench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `map_with_progress(f:Callable, xs:list[Any], num_threads:int=os.cpu_count() or 10, pbar:bool=True) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_parse(item:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `parse_json_to_dict(json_string:str) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `calculate_score(rubric_items:list[RubricItem], grading_response_list:list[dict]) -> float | None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_usage_dict(response_usage:Any) -> dict[str, int | None]`
    Descrizione: Recupera valore/stato calcolato.
  - `_compute_clipped_stats(values:list, stat:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_aggregate_get_clipped_mean(single_eval_results:list[SingleEvalResult]) -> EvalResult`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `RubricItem`
    Metodo: `__init__(self:Any, criterion:str, points:float, tags:list[str]) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__str__(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `to_dict(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `from_dict(cls:Any, d:dict) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `HealthBenchDataset`
    Metodo: `load(path:str, **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `HealthBenchEvaluator`
    Metodo: `__init__(self:Any, subset_name:Any=Literal['hard', 'consensus'] | None, n_repeats:Any=1, n_threads:Any=1) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `grade_sample(self:Any, prompt:list[dict[str, str]], response_text:str, example_tags:list[str], rubric_items:list[RubricItem]) -> tuple[dict, str, list[dict]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any, test_set:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\healthbench\sampler\chat_completion_sampler.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `ChatCompletionSampler`
    Metodo: `__init__(self:Any, model:str='gpt-3.5-turbo', system_message:str | None=None, temperature:float=0.5, max_tokens:int=1024) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_handle_image(self:Any, image:str, encoding:str='base64', format:str='png', fovea:int=768) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_handle_text(self:Any, text:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_pack_message(self:Any, role:str, content:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__call__(self:Any, message_list:MessageList) -> SamplerResponse`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\healthbench\types.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `SamplerResponse`
  - `SamplerBase`
    Metodo: `__call__(self:Any, message_list:MessageList) -> SamplerResponse`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `EvalResult`
  - `SingleEvalResult`
  - `Eval`
    Metodo: `__call__(self:Any, sampler:SamplerBase) -> EvalResult`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\hellaswag.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `HellaswagDataset`
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `HellaswagDataset_V2`
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `HellaswagDataset_V3`
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `HellaswagDatasetwithICE`
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `HellaswagDatasetClean`
    Metodo: `load_contamination_annotations(path:Any, split:Any='val') -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\hle.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `HLEDataset`
    Metodo: `load(path:str, category:str | None=None) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\huggingface.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `HFDataset`
    Metodo: `load(**kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\humaneval.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `humaneval_postprocess_v2(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `humaneval_postprocess_v2_(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `humaneval_postprocess_v3(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `humaneval_internal_v2_postprocess(text:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `humaneval_internal_v1_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `HumanevalDataset`
    Metodo: `load(path:str, num_repeats:int=1, local_mode:bool=False) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `HumanEvalEvaluator`
    Metodo: `__init__(self:Any, k:List[int]=[1, 10, 100]) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any, test_set:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `HumanEvalPlusEvaluator`
    Metodo: `__init__(self:Any, k:List[int]=[1, 10, 100]) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any, test_set:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\humaneval_multi.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `HumanevalMultiDataset`
    Metodo: `load(path:Any, language:Any, version:Any, num_repeats:int=1, **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `HumanevalMultiEvaluator`
    Metodo: `__init__(self:Any, language:Any, ip_address:Any='localhost', port:Any=5000, retry:Any=2, timeout:Any=600) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `stop_at_stop_token(self:Any, decoded_string:Any, stop_tokens:Any) -> Any`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
    Metodo: `_code_eval_service(self:Any, file_path:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `estimator(self:Any, n:int, c:int, k:int) -> float`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `for_file(self:Any, path:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any, test_set:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\humaneval_pro.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `HumanevalevalProDataset`
    Metodo: `load(path:Any, local_mode:Any=False) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `HumanevalProEvaluator`
    Metodo: `score(self:Any, predictions:List, references:List, test_set:Dataset) -> Dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\humanevalx.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_clean_up_code(text:str, language_type:str, reference:Any) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `HumanevalXDataset`
    Metodo: `load(path:Any, language:Any, **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
    Metodo: `_stream_jsonl_all(filename:str) -> Iterable[Dict]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `HumanevalXEvaluator`
    Metodo: `__init__(self:Any, language:Any, ip_address:Any='localhost', port:Any='', retry:Any=2, timeout:Any=600) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_code_eval_service(self:Any, file_path:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\hungarian_math.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `HungarianExamMathDataset`
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\IFEval\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\IFEval\evaluation_main.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `test_instruction_following_strict(inp:Any, response:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `test_instruction_following_loose(inp:Any, response:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  Classi:
  - `InputExample`
  - `OutputExample`

- File: `vendor\llada\opencompass\opencompass\datasets\IFEval\ifeval.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `IFEvalDataset`
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `IFEvaluator`
    Metodo: `score(self:Any, predictions:Any, references:Any, origin_prompt:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\IFEval\instructions.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `Instruction`
    Metodo: `__init__(self:Any, instruction_id:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `build_description(self:Any, **kwargs:Any) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
    Metodo: `get_instruction_args(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_instruction_args_keys(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `check_following(self:Any, value:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `ResponseLanguageChecker`
    Metodo: `build_description(self:Any, language:Any=None) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
    Metodo: `get_instruction_args(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_instruction_args_keys(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `check_following(self:Any, value:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `NumberOfSentences`
    Metodo: `build_description(self:Any, num_sentences:Any=None, relation:Any=None) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
    Metodo: `get_instruction_args(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_instruction_args_keys(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `check_following(self:Any, value:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `PlaceholderChecker`
    Metodo: `build_description(self:Any, num_placeholders:Any=None) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
    Metodo: `get_instruction_args(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_instruction_args_keys(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `check_following(self:Any, value:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `BulletListChecker`
    Metodo: `build_description(self:Any, num_bullets:Any=None) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
    Metodo: `get_instruction_args(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_instruction_args_keys(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `check_following(self:Any, value:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `ConstrainedResponseChecker`
    Metodo: `build_description(self:Any) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
    Metodo: `get_instruction_args(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_instruction_args_keys(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `check_following(self:Any, value:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `ConstrainedStartChecker`
    Metodo: `build_description(self:Any, starter:Any=None) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
    Metodo: `get_instruction_args(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_instruction_args_keys(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `check_following(self:Any, value:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `HighlightSectionChecker`
    Metodo: `build_description(self:Any, num_highlights:Any=None) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
    Metodo: `get_instruction_args(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_instruction_args_keys(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `check_following(self:Any, value:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `SectionChecker`
    Metodo: `build_description(self:Any, section_spliter:Any=None, num_sections:Any=None) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
    Metodo: `get_instruction_args(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_instruction_args_keys(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `check_following(self:Any, value:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `ParagraphChecker`
    Metodo: `build_description(self:Any, num_paragraphs:Any=None) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
    Metodo: `get_instruction_args(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_instruction_args_keys(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `check_following(self:Any, value:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `PostscriptChecker`
    Metodo: `build_description(self:Any, postscript_marker:Any=None) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
    Metodo: `get_instruction_args(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_instruction_args_keys(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `check_following(self:Any, value:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `RephraseChecker`
    Metodo: `build_description(self:Any, original_message:Any) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
    Metodo: `get_instruction_args(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_instruction_args_keys(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `check_following(self:Any, value:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `is_change(self:Any, response:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `strip_changes(self:Any, response:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `KeywordChecker`
    Metodo: `build_description(self:Any, keywords:Any=None) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
    Metodo: `get_instruction_args(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_instruction_args_keys(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `check_following(self:Any, value:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `KeywordFrequencyChecker`
    Metodo: `build_description(self:Any, keyword:Any=None, frequency:Any=None, relation:Any=None) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
    Metodo: `get_instruction_args(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_instruction_args_keys(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `check_following(self:Any, value:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `NumberOfWords`
    Metodo: `build_description(self:Any, num_words:Any=None, relation:Any=None) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
    Metodo: `get_instruction_args(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_instruction_args_keys(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `check_following(self:Any, value:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `JsonFormat`
    Metodo: `build_description(self:Any) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
    Metodo: `get_instruction_args(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_instruction_args_keys(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `check_following(self:Any, value:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `ParagraphFirstWordCheck`
    Metodo: `build_description(self:Any, num_paragraphs:Any=None, nth_paragraph:Any=None, first_word:Any=None) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
    Metodo: `get_instruction_args(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_instruction_args_keys(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `check_following(self:Any, value:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `KeySentenceChecker`
    Metodo: `build_description(self:Any, key_sentences:Any=None, num_sentences:Any=None) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
    Metodo: `get_instruction_args(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_instruction_args_keys(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `check_following(self:Any, value:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `ForbiddenWords`
    Metodo: `build_description(self:Any, forbidden_words:Any=None) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
    Metodo: `get_instruction_args(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_instruction_args_keys(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `check_following(self:Any, value:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `RephraseParagraph`
    Metodo: `build_description(self:Any, original_paragraph:Any, low:Any, high:Any) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
    Metodo: `get_instruction_args(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_instruction_args_keys(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `check_following(self:Any, value:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `TwoResponsesChecker`
    Metodo: `build_description(self:Any) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
    Metodo: `get_instruction_args(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_instruction_args_keys(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `check_following(self:Any, value:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `RepeatPromptThenAnswer`
    Metodo: `build_description(self:Any, prompt_to_repeat:Any=None) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
    Metodo: `get_instruction_args(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_instruction_args_keys(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `check_following(self:Any, value:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `EndChecker`
    Metodo: `build_description(self:Any, end_phrase:Any=None) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
    Metodo: `get_instruction_args(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_instruction_args_keys(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `check_following(self:Any, value:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `TitleChecker`
    Metodo: `build_description(self:Any) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
    Metodo: `get_instruction_args(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_instruction_args_keys(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `check_following(self:Any, value:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LetterFrequencyChecker`
    Metodo: `build_description(self:Any, letter:Any=None, let_frequency:Any=None, let_relation:Any=None) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
    Metodo: `get_instruction_args(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_instruction_args_keys(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `check_following(self:Any, value:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `CapitalLettersEnglishChecker`
    Metodo: `build_description(self:Any) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
    Metodo: `get_instruction_args(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_instruction_args_keys(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `check_following(self:Any, value:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LowercaseLettersEnglishChecker`
    Metodo: `build_description(self:Any) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
    Metodo: `get_instruction_args(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_instruction_args_keys(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `check_following(self:Any, value:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `CommaChecker`
    Metodo: `build_description(self:Any) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
    Metodo: `get_instruction_args(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_instruction_args_keys(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `check_following(self:Any, value:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `CapitalWordFrequencyChecker`
    Metodo: `build_description(self:Any, capital_frequency:Any=None, capital_relation:Any=None) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
    Metodo: `get_instruction_args(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_instruction_args_keys(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `check_following(self:Any, value:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `QuotationChecker`
    Metodo: `build_description(self:Any) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
    Metodo: `get_instruction_args(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_instruction_args_keys(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `check_following(self:Any, value:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\IFEval\instructions_registry.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `conflict_make(conflicts:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\IFEval\instructions_util.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `split_into_sentences(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `count_words(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_get_sentence_tokenizer() -> Any`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
  - `count_sentences(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `generate_keywords(num_keywords:Any) -> Any`
    Descrizione: Genera output testo o sequenze.

- File: `vendor\llada\opencompass\opencompass\datasets\inference_ppl.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `InferencePPLDataset`
    Metodo: `load(path:str, name:List[str]=None, samples:int=None) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\infinitebench\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\infinitebench\infinitebench_codedebug.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `InfiniteBenchcodedebugDataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\infinitebench\infinitebench_coderun.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `InfiniteBenchcoderunDataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\infinitebench\infinitebench_endia.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `InfiniteBenchendiaDataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `InfiniteBenchendiaEvaluator`
    Metodo: `score(self:Any, predictions:List, references:List) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\infinitebench\infinitebench_enmc.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `InfiniteBenchenmcDataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\infinitebench\infinitebench_enqa.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `InfiniteBenchenqaDataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\infinitebench\infinitebench_ensum.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `InfiniteBenchensumDataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\infinitebench\infinitebench_mathcalc.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `InfiniteBenchmathcalcDataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `InfiniteBenchmathcalcEvaluator`
    Metodo: `score(self:Any, predictions:List, references:List) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\infinitebench\infinitebench_mathfind.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `InfiniteBenchmathfindDataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\infinitebench\infinitebench_retrievekv.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `InfiniteBenchretrievekvDataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `InfiniteBenchretrievekvEvaluator`
    Metodo: `score(self:Any, predictions:List, references:List) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\infinitebench\infinitebench_retrievenumber.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `InfiniteBenchretrievenumberDataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\infinitebench\infinitebench_retrievepasskey.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `InfiniteBenchretrievepasskeyDataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\infinitebench\infinitebench_zhqa.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `InfiniteBenchzhqaDataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\infinitebench\utils.py`
  Logica d'uso: Runtime helpers, device/dtype control, tiny fallback model/tokenizer.
  Funzioni:
  - `iter_jsonl(path:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `InfiniteBench_first_number_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\internsandbox.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `InternSandboxDataset`
    Metodo: `load(path:str, sandbox:str, local_mode:bool=False) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `InternSandboxEvaluator`
    Metodo: `__init__(self:Any, short_penalty:bool=False, format_penalty:bool=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any, test_set:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\iwslt2017.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `IWSLT2017Dataset`
    Metodo: `load(**kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\jigsawmultilingual.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `JigsawMultilingualDataset`
    Metodo: `load(path:Any, label:Any, lang:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\jsonl.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `JsonlDataset`
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\judge\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\judge\judgebench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `JudgeBenchDataset`
    Metodo: `load(self:Any, path:str, name:str, *args:Any, **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\judge\judgerbenchv2.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `generate_balanced_list(length:Any) -> Any`
    Descrizione: Genera output testo o sequenze.
  Classi:
  - `Judgerbenchv2Dataset`
    Metodo: `load(self:Any, path:str, name:str, *args:Any, **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\judge\rewardbench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `RewardBenchDataset`
    Metodo: `load(self:Any, path:str, name:str, *args:Any, **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\judge\rmb.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `RMBDataset`
    Metodo: `load(self:Any, path:str, name:str, *args:Any, **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
    Metodo: `load_pair(self:Any, item:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
    Metodo: `loadbon(self:Any, item:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\kaoshi.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_number(options:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  Classi:
  - `KaoshiDataset`
    Metodo: `load(path:str, name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `KaoshiEvaluator`
    Metodo: `__init__(self:Any, question_type:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `do_predictions_postprocess(self:Any, model_output:Any, answer_lenth:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `ensure_same_length(self:Any, pred:Any, refr:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\kcle.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `KCLEDataset`
    Metodo: `load(path:Any, **kwargs:Any) -> datasets.Dataset`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\korbench\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\korbench\korbench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `korbenchDataset`
    Metodo: `load(path:Any, prompt_mode:Any, category:Any, **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `korbenchEvaluator`
    Metodo: `__init__(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `sample_score(self:Any, prediction:Any, reference:Any, test_item:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any, test_set:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\korbench\korbench_dataset_config\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\korbench\korbench_dataset_config\config_wrapper.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `initialize_config(config_path:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_config_wrapper() -> Any`
    Descrizione: Recupera valore/stato calcolato.
  Classi:
  - `ConfigWrapper`
    Metodo: `__init__(self:Any, config_path:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__setattr__(self:Any, key:Any, value:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__getattr__(self:Any, key:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_id(self:Any, data:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `print_all_keys(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\korbench\korbench_dataset_config\prompt\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\korbench\korbench_utils.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `load_yaml(yaml_path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `load_json_or_jsonl(file_path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `find_file(base_path:Any, sub_path:Any, extensions:Any=('json', 'jsonl')) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `load_json_or_jsonl_with_idx(data_path:Any, split:Any='', idx:Any=None) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `load_split_data(base_path:Any, split_name:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `process_mixed_data(base_path:Any, mode:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `initialize_config(config_path:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_config_wrapper() -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `read_yaml(config:Any='default') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `write_jsonl_lines(file:Any, data:Any) -> Any`
    Descrizione: Serializza e salva output su disco.
  - `print_info(info:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `read_json_or_jsonl(data_path:Any, split:Any='', mapping_key:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `read_json_or_jsonl_with_idx(data_path:Any, split:Any='', idx:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `clean_json_string(json_str:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `is_in_idx_ranges(idx:Any, idx_ranges:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `extract_json(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `extract_all_responses_from_json(response_json:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `clean_latex(latex_expr:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `extract_text_from_brackets(text:Any, clean_level:Any='basic') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `extract_inner_text_from_brackets(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `extract_numbers(str:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `extract_and_sort_inequalities(latex_expr:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `rule5_normalize_content(content:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `normalize_string(s:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `remove_commas_and_spaces(s:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `remove_non_alphanumeric(s:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `contains_or(answer:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `compare_multi_results(response:Any, answer:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `split_or_expression(expression:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `compare_math_expressions(response:Any, answer:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `method_equal(response_text:Any, answer:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `method_1(response_text:Any, answer:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `method_2(response_text:Any, answer:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `method_3(response_text:Any, answer:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `method_4(response_text:Any, answer:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `method_5(response_text:Any, answer:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `method_9(response_text:Any, answer:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `method_10(response_text:Any, answer:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `method_18(response_text:Any, answer:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `method_general(response_text:Any, answer:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `evaluate_response_vs_answer(response:Any, answer:Any, question_type:Any, rule_id:Any, idx:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `compute_one_mixed_question_pass_rate(idx:Any, question_list:Any, response_json:Any, base_path:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `evaluate_responses(data:Any, mode:Any, base_path:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `ConfigWrapper`
    Metodo: `__init__(self:Any, config_path:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__setattr__(self:Any, key:Any, value:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__getattr__(self:Any, key:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_id(self:Any, data:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `print_all_keys(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\lambada.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `lambadaDataset`
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `LambadaEvaluator`
    Metodo: `__init__(self:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\lawbench\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\lawbench\evaluation_functions\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\lawbench\evaluation_functions\cjft.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `compute_cjft(data_dict:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\lawbench\evaluation_functions\flzx.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `compute_flzx(data_dict:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\lawbench\evaluation_functions\ftcs.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `compute_ftcs(data_dict:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\lawbench\evaluation_functions\jdzy.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `compute_jdzy(data_dict:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\lawbench\evaluation_functions\jec_ac.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `compute_jec_ac(data_dict:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\lawbench\evaluation_functions\jec_kd.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `compute_jec_kd(data_dict:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\lawbench\evaluation_functions\jetq.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `compute_jetq(data_dict:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\lawbench\evaluation_functions\lblj.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `compute_lblj(data_dict:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\lawbench\evaluation_functions\ljp_accusation.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `compute_ljp_accusation(data_dict:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\lawbench\evaluation_functions\ljp_article.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `replace_match(match:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `compute_ljp_article(data_dict:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\lawbench\evaluation_functions\ljp_imprison.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `compute_ljp_imprison(data_dict:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\lawbench\evaluation_functions\sjjc.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `compute_sjjc(data_dict:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `compute_cfcy(data_dict:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\lawbench\evaluation_functions\wbfl.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `compute_wbfl(data_dict:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\lawbench\evaluation_functions\wsjd.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `compute_wsjd(data_dict:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\lawbench\evaluation_functions\xxcq.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `compute_xxcq(data_dict:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\lawbench\evaluation_functions\ydlj.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `compute_ydlj(data_dict:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\lawbench\evaluation_functions\yqzy.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `compute_yqzy(data_dict:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\lawbench\evaluation_functions\zxfl.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `compute_zxfl(data_dict:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\lawbench\lawbench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LawBenchDataset`
    Metodo: `load(path:str, index:str) -> Dataset`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `LawBenchEvaluator`
    Metodo: `__init__(self:Any, index:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any, origin_prompt:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\lawbench\utils\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\lawbench\utils\char_smi.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `tree_edit_distance(tree_a:Any, tree_b:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `edit_distance(string_a:Any, string_b:Any, name:Any='Levenshtein') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `string_to_tree(string:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `pinyin_map(standard_pinyin:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `parse_args() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `CharFuncs`
    Metodo: `__init__(self:Any, char_meta_fname:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `load_char_meta(fname:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
    Metodo: `shape_distance(self:Any, char1:Any, char2:Any, safe:Any=True, as_tree:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `pronunciation_distance(self:Any, char1:Any, char2:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `load_dict(fname:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
    Metodo: `similarity(self:Any, char1:Any, char2:Any, weights:Any=(0.8, 0.2, 0.0), as_tree:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `shape_similarity(self:Any, char1:Any, char2:Any, safe:Any=True, as_tree:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `pronunciation_similarity(self:Any, char1:Any, char2:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\lawbench\utils\compare_m2_for_evaluation.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `main() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `parse_args() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `simplify_edits(sent:Any, max_answer_num:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `process_edits(edits:Any, args:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `evaluate_edits(src:Any, hyp_dict:Any, ref_dict:Any, best:Any, sent_id:Any, args:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `compareEdits(hyp_edits:Any, ref_edits:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `computeFScore(tp:Any, fp:Any, fn:Any, beta:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `merge_dict(dict1:Any, dict2:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `processCategories(cat_dict:Any, setting:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `print_results(best:Any, best_cats:Any, args:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\lawbench\utils\comprehension_scores.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `__find_substring_starts(s:Any, target:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `compute_rc_f1(hyps:Any, refs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `compute_ie_f1(hyps:Any, refs:Any, entity_types:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `__extract_entities_ref(ref:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `__extract_entities_pred(pred:Any, entity_types:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\lawbench\utils\function_utils.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `compute_f1_two_sets(pred_set:Any, gt_set:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `multi_choice_judge(prediction:Any, option_list:Any, answer_token:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `compute_rouge(hyps:Any, refs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `compute_gleu(hyps:Any, refs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\lawbench\utils\modules\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\lawbench\utils\modules\alignment.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `check_all_chinese(word:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `read_cilin() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `read_confusion() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `Alignment`
    Metodo: `__init__(self:Any, semantic_dict:Dict, confusion_dict:Dict, granularity:str='word') -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__call__(self:Any, src:List[Tuple], tgt:List[Tuple], verbose:bool=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_get_semantic_class(self:Any, word:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_get_class_diff(a_class:Any, b_class:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_get_semantic_cost(self:Any, a:Any, b:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_get_pos_cost(self:Any, a_pos:Any, b_pos:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_get_char_cost(self:Any, a:Any, b:Any, pinyin_a:Any, pinyin_b:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_get_spell_cost(self:Any, a:Any, b:Any, pinyin_a:Any, pinyin_b:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_sub_cost(self:Any, a_seg:Any, b_seg:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `align(self:Any, src:List[Tuple], tgt:List[Tuple]) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_dfs(self:Any, i:Any, j:Any, align_seq_now:Any, oper_matrix:Any, strategy:Any='all') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_cheapest_align_seq(self:Any, oper_matrix:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\lawbench\utils\modules\annotator.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `Annotator`
    Metodo: `__init__(self:Any, align:Alignment, merger:Merger, classifier:Classifier, granularity:str='word', strategy:str='first') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `create_default(cls:Any, granularity:str='word', strategy:str='first') -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
    Metodo: `__call__(self:Any, src:List[Tuple], tgt:List[Tuple], annotator_id:int=0, verbose:bool=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\lawbench\utils\modules\classifier.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `check_spell_error(src_span:str, tgt_span:str, threshold:float=0.8) -> bool`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `Classifier`
    Metodo: `__init__(self:Any, granularity:str='word') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_pos_type(pos:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `__call__(self:Any, src:Any, tgt:Any, edits:Any, verbose:bool=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\lawbench\utils\modules\merger.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `Merger`
    Metodo: `__init__(self:Any, granularity:str='word', merge:bool=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_merge_edits(seq:Any, tag:Any='X') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_check_revolve(span_a:Any, span_b:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_process_seq(self:Any, seq:Any, src_tokens:Any, tgt_tokens:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__call__(self:Any, align_obj:Any, src:List, tgt:List, verbose:bool=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\lawbench\utils\modules\tokenization.py`
  Logica d'uso: Tokenizes and packs sequences, emits doc_ids used by attention masks.
  Funzioni:
  - `convert_to_unicode(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `printable_text(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `load_vocab(vocab_file:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `convert_by_vocab(vocab:Any, items:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `convert_tokens_to_ids(vocab:Any, tokens:Any) -> Any`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
  - `convert_ids_to_tokens(inv_vocab:Any, ids:Any) -> Any`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
  - `whitespace_tokenize(text:Any) -> Any`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
  - `_is_whitespace(char:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_is_control(char:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_is_punctuation(char:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `FullTokenizer`
    Metodo: `__init__(self:Any, vocab_file:Any, do_lower_case:Any=True) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `tokenize(self:Any, text:Any) -> Any`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
    Metodo: `convert_tokens_to_ids(self:Any, tokens:Any) -> Any`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
    Metodo: `convert_ids_to_tokens(self:Any, ids:Any) -> Any`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
  - `BasicTokenizer`
    Metodo: `__init__(self:Any, do_lower_case:Any=True) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `tokenize(self:Any, text:Any) -> Any`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
    Metodo: `_run_strip_accents(self:Any, text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_run_split_on_punc(self:Any, text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_tokenize_chinese_chars(self:Any, text:Any) -> Any`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
    Metodo: `_is_chinese_char(self:Any, cp:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_clean_text(self:Any, text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `WordpieceTokenizer`
    Metodo: `__init__(self:Any, vocab:Any, unk_token:Any='[UNK]', max_input_chars_per_word:Any=100) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `tokenize(self:Any, text:Any) -> Any`
    Descrizione: Gestisce tokenizzazione o manipolazione token.

- File: `vendor\llada\opencompass\opencompass\datasets\lawbench\utils\modules\tokenizer.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `Tokenizer`
    Metodo: `__init__(self:Any, granularity:str='word', device:str='cpu', segmented:bool=False, bpe:bool=False) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__repr__(self:Any) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__call__(self:Any, input_strings:List[str]) -> List`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `split_char(self:Any, input_strings:List[str], bpe:Any=False) -> List`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `split_word(self:Any, input_strings:List[str]) -> List`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\lawbench\utils\parallel_to_m2.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `annotate_with_time_out(line:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `annotate(line:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `firsttime_process(args:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `main(args:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\lawbench\utils\rc_f1.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `CJRCEvaluator`
    Metodo: `__init__(self:Any, gold_file:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `gold_answers_to_dict(gold_file:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `preds_to_dict(pred_file:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `normalize_answer(s:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_tokens(s:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `compute_exact(a_gold:Any, a_pred:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `compute_f1(a_gold:Any, a_pred:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_compute_turn_score(a_gold_list:Any, a_pred:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `compute_turn_score(self:Any, qid:Any, a_pred:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_raw_scores(self:Any, pred_data:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_raw_scores_human(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `human_performance(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `model_performance(self:Any, pred_data:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_total_scores(self:Any, exact_scores:Any, f1_scores:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\LCBench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `swallow_io() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `time_limit(seconds:float) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `execution(programs:Any, task_id:Any, timeout:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `LCDataset`
    Metodo: `load(path:str, num_repeats:int=1, difficulty:Any='ALL', local_mode:Any=False) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `TimeOutException`
  - `WriteOnlyStringIO`
    Metodo: `read(self:Any, *args:Any, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `readline(self:Any, *args:Any, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `readlines(self:Any, *args:Any, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `readable(self:Any, *args:Any, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `redirect_stdin`
  - `LCEvaluator`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_process_answer(self:Any, text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_process_test(self:Any, test_case:Any, pred:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LCPassKEvaluator`
    Metodo: `__init__(self:Any, k:Any=(1, 10, 100)) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `estimate_pass_at_k(num_samples:Union[int, List[int], np.ndarray], num_correct:Union[List[int], np.ndarray], k:int) -> np.ndarray`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\lcsts.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `lcsts_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `LCSTSDataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\leval\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\leval\evaluators.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LEvalGPTEvaluator`
    Metodo: `__init__(self:Any, battle_model:str='turbo-16k-0613', evaluator_path:str='gpt-4-0613') -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `run_judge_pair(self:Any, prompt_template:Any, system_prompt:Any, question:Any, answer_a:Any, answer_b:Any, reference:Any) -> Any`
    Descrizione: Esegue pipeline o job completo.
    Metodo: `score(self:Any, predictions:List, references:List) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LEvalEMEvaluator`
    Metodo: `__init__(self:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\leval\leval_coursera.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LEvalCourseraDataset`
    Metodo: `load(**kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\leval\leval_financial_qa.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LEvalFinancialQADataset`
    Metodo: `load(**kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\leval\leval_gov_report_summ.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LEvalGovReportSummDataset`
    Metodo: `load(**kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\leval\leval_gsm100.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `gsm100_dataset_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `gsm100_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `LEvalGSM100Dataset`
    Metodo: `load(**kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\leval\leval_legal_contract_qa.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LEvalLegalContractQADataset`
    Metodo: `load(**kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\leval\leval_meeting_summ.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LEvalMeetingSummDataset`
    Metodo: `load(**kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\leval\leval_multidoc_qa.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LEvalMultidocQADataset`
    Metodo: `load(**kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\leval\leval_narrattive_qa.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LEvalNarrativeQADataset`
    Metodo: `load(**kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\leval\leval_natural_question.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LEvalNaturalQuestionDataset`
    Metodo: `load(**kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\leval\leval_news_summ.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LEvalNewsSummDataset`
    Metodo: `load(**kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\leval\leval_paper_assistant.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LEvalPaperAssistantDataset`
    Metodo: `load(**kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\leval\leval_patent_summ.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LEvalPatentSummDataset`
    Metodo: `load(**kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\leval\leval_quality.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LEvalQualityDataset`
    Metodo: `load(**kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\leval\leval_review_summ.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LEvalReviewSummDataset`
    Metodo: `load(**kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\leval\leval_scientific_qa.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LEvalScientificQADataset`
    Metodo: `load(**kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\leval\leval_topic_retrieval.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LEvalTopicRetrievalDataset`
    Metodo: `load(**kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\leval\leval_tpo.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LEvalTPODataset`
    Metodo: `load(**kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\leval\leval_tvshow_summ.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LEvalTVShowSummDataset`
    Metodo: `load(**kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\livecodebench\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\livecodebench\evaluator.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `codegen_check_correctness(sample:Any, generation:Any, timeout:Any, debug:Any=True) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `evaluate_generations_by_problem(problem_generations:list, sample:list, debug:bool, timeout:int) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `evaluate_generations(samples_list:list, generations_list:list[list[str]], debug:bool=False, num_process_evaluate:int=16, timeout:Any=6) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `codegen_metrics(samples_list:Any, generations_list:Any, k_list:Any=[1, 5, 10, 20, 40, 50, 75, 100, 125, 150, 200, 500, 1000], num_process_evaluate:Any=16, timeout:Any=6, debug:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `evaluate_score(args:Any) -> list[bool]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `code_execution_metrics(samples:Any, generations:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `parse_assert_statement(statement:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `check_testcase_output(testcase_str:Any, expected_output:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `test_output_metrics(samples:Any, generations:Any, k_list:Any=[1, 5]) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  Classi:
  - `LCBCodeGenerationEvaluator`
    Metodo: `__init__(self:Any, num_process_evaluate:Any, timeout:Any=6, release_version:Any='release_v1', extractor_version:Any='v1', start_date:Any=None, end_date:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_build_results(self:Any, extracted_predictions:Any, metrics:Any, eval_results:Any, final_metadata:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LCBCodeExecutionEvaluator`
    Metodo: `__init__(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LCBTestOutputEvaluator`
    Metodo: `__init__(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\livecodebench\execute_utils.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `codeexecute_check_correctness(check_program:Any, timeout:Any=3) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `unsafe_execute(check_program:Any, result:Any, timeout:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `time_limit(seconds:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `swallow_io() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `create_tempdir() -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
  - `chdir(root:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `reliability_guard(maximum_memory_bytes:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `TimeoutException`
  - `WriteOnlyStringIO`
    Metodo: `read(self:Any, *args:Any, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `readline(self:Any, *args:Any, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `readlines(self:Any, *args:Any, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `readable(self:Any, *args:Any, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `redirect_stdin`

- File: `vendor\llada\opencompass\opencompass\datasets\livecodebench\extract_utils.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `extract_code_generation(model_output:str, model_type:str='chat') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `extract_code_generation_v2(model_output:str, model_type:str='chat') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `extract_code_execution(model_output:str, cot:bool=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `extract_test_output_code(model_output:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\livecodebench\livecodebench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `Platform`
  - `Difficulty`
  - `TestType`
  - `Test`
    Metodo: `__post_init__(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LCBCodeGenerationDataset`
    Metodo: `load(path:str='opencompass/code_generation_lite', local_mode:bool=False, release_version:str='release_v1', start_date:str=None, end_date:str=None) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `LCBCodeExecutionDataset`
    Metodo: `load(path:str='opencompass/execution-v2', local_mode:bool=False, cot:bool=False) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `LCBTestOutputPredictionDataset`
    Metodo: `load(path:str='opencompass/test_generation', local_mode:bool=False) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `LCBSelfRepairDataset`
    Metodo: `load(path:str='livecodebench/code_generation_lite', local_mode:bool=False, release_version:str='release_v1') -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `CompassBenchCodeExecutionDataset`
    Metodo: `load(path:str='opencompass/execution-v2', local_mode:bool=False, cot:bool=False) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\livecodebench\pass_k_utils.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `estimate_pass_at_k(num_samples:Any, num_correct:Any, k:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `compute_metrics_from_results(results:Any, k_list:Any=[1, 5]) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `extract_instance_results(results:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\livecodebench\prompts.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `make_code_execution_prompt(code:Any, input:Any, cot:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_generic_question_template_test_completion(question_content:Any, starter_code:Any, testcase_input:str) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `get_generic_question_template_answer_self_repair(question:str, code:Any, metadata:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  Classi:
  - `CodeGenerationPromptConstants`
  - `TestOutputPromptConstants`
  - `SelfRepairPromptConstants`

- File: `vendor\llada\opencompass\opencompass\datasets\livecodebench\testing_util.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `truncatefn(s:Any, length:Any=300) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `timeout_handler(signum:Any, frame:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `only_int_check(val:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `string_int_check(val:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `combined_int_check(val:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `run_test(sample:Any, test:Any=None, debug:Any=False, timeout:Any=6) -> Any`
    Descrizione: Esegue pipeline o job completo.
  - `custom_compare_(output:Any, ground_truth:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `stripped_string_compare(s1:Any, s2:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `call_method(method:Any, inputs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `reliability_guard(maximum_memory_bytes:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `CODE_TYPE`
  - `TimeoutException`
  - `Capturing`
    Metodo: `__enter__(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__exit__(self:Any, *args:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\livemathbench\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\livemathbench\livemathbench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LiveMathBenchDataset`
    Metodo: `load(path:str, dataset_splits:List[str]=['CNMO', 'CCEE', 'AMC', 'WLPMC'], dataset_languages:List[str]=['cn', 'en'], cot:bool=True, version:str='202412') -> List[Dict[str, Any]]`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `LiveMathBenchEvaluator`
    Metodo: `__init__(self:Any, model_name:Any, url:Any, use_extract_model:Any=False, extract_url:Any=[], extract_model_name:Any='') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `batch_infer(self:Any, models:List[OpenAISDK], inputs:List[str], completed_indexes:set, output_handler:'LiveMathBenchOutputHandler', postprocess:Callable) -> List[str]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `extract(self:Any, questions:List[str], predictions:List[str], question_types:List[str], languages:List[str]) -> List[str]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `judge(self:Any, predictions:Any, references:Any, test_set:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `preprocess(self:Any, predictions:Any, references:Any, test_set:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any, test_set:Any) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LiveMathBenchOutputHandler`
    Metodo: `write_to_json(self:Any) -> Any`
    Descrizione: Serializza e salva output su disco.
    Metodo: `save(self:Any, idx:Any, **kwargs:Any) -> Any`
    Descrizione: Serializza e salva output su disco.

- File: `vendor\llada\opencompass\opencompass\datasets\livemathbench\prompts.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\livemathbench\utils.py`
  Logica d'uso: Runtime helpers, device/dtype control, tiny fallback model/tokenizer.
  Funzioni:
  - `extract_judge_label(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\livereasonbench\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\livereasonbench\livereasonbench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_final_results(judged_answers:Any, references:Any, origial_responses:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `_livereasonbench_postprocess(judgement:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `livereasonbench_postprocess(output:dict, output_path:str) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `LiveReasonBenchDataset`
    Metodo: `load(path:str, num_examples:int | None=None, n_repeats:int=1, version:str='livereasonbench-20241202', **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\livestembench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LiveStemBenchDataset`
    Metodo: `load(path:str, num_examples:int | None=None, n_repeats:int=1, version:str='livestembench-20241227', **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\llm_compression.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LLMCompressionDataset`
    Metodo: `load(path:str, name:List[str]=None, samples:int=None) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\lmeval.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LMEvalDataset`
    Metodo: `load(**kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\longbench\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\longbench\evaluators.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `normalize_answer(s:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `normalize_zh_answer(s:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `LongBenchF1Evaluator`
    Metodo: `__init__(self:Any, language:str='en') -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:List, references:List) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LongBenchCountEvaluator`
    Metodo: `score(self:Any, predictions:List, references:List) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LongBenchRetrievalEvaluator`
    Metodo: `__init__(self:Any, language:str='en') -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:List, references:List) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LongBenchRougeEvaluator`
    Metodo: `__init__(self:Any, language:str='en') -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:List, references:List) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LongBenchCodeSimEvaluator`
    Metodo: `score(self:Any, predictions:List, references:List) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LongBenchClassificationEvaluator`
    Metodo: `score(self:Any, predictions:List, references:List) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\longbench\longbench_2wikim_qa.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LongBench2wikimqaDataset`
    Metodo: `load(path:str, name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\longbench\longbench_dureader.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LongBenchdureaderDataset`
    Metodo: `load(path:str, name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\longbench\longbench_gov_report.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LongBenchgov_reportDataset`
    Metodo: `load(path:str, name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\longbench\longbench_hotpot_qa.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LongBenchhotpotqaDataset`
    Metodo: `load(path:str, name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\longbench\longbench_lcc.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LongBenchlccDataset`
    Metodo: `load(path:str, name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\longbench\longbench_lsht.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `lsht_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `LongBenchlshtDataset`
    Metodo: `load(path:str, name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\longbench\longbench_multi_news.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LongBenchmulti_newsDataset`
    Metodo: `load(path:str, name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\longbench\longbench_multifieldqa_en.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LongBenchmultifieldqa_enDataset`
    Metodo: `load(path:str, name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\longbench\longbench_multifieldqa_zh.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LongBenchmultifieldqa_zhDataset`
    Metodo: `load(path:str, name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\longbench\longbench_musique.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LongBenchmusiqueDataset`
    Metodo: `load(path:str, name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\longbench\longbench_narrative_qa.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LongBenchnarrativeqaDataset`
    Metodo: `load(path:str, name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\longbench\longbench_passage_count.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LongBenchpassage_countDataset`
    Metodo: `load(path:str, name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\longbench\longbench_passage_retrieval_en.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LongBenchpassage_retrieval_enDataset`
    Metodo: `load(path:str, name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\longbench\longbench_passage_retrieval_zh.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LongBenchpassage_retrieval_zhDataset`
    Metodo: `load(path:str, name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\longbench\longbench_qasper.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LongBenchqasperDataset`
    Metodo: `load(path:str, name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\longbench\longbench_qmsum.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LongBenchqmsumDataset`
    Metodo: `load(path:str, name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\longbench\longbench_repobench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LongBenchrepobenchDataset`
    Metodo: `load(path:str, name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\longbench\longbench_samsum.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `samsum_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `LongBenchsamsumDataset`
    Metodo: `load(path:str, name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\longbench\longbench_trec.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `trec_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `LongBenchtrecDataset`
    Metodo: `load(path:str, name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\longbench\longbench_trivia_qa.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `triviaqa_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `LongBenchtriviaqaDataset`
    Metodo: `load(path:str, name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\longbench\longbench_vcsum.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LongBenchvcsumDataset`
    Metodo: `load(path:str, name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\longbenchv2.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LongBenchv2Dataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `LongBenchv2Evaluator`
    Metodo: `__init__(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any, test_set:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\lveval\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\lveval\evaluators.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `normalize_answer(s:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `normalize_zh_answer(s:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `LVEvalF1Evaluator`
    Metodo: `__init__(self:Any, language:str='en') -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:List, references:List) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LVEvalOPTF1Evaluator`
    Metodo: `__init__(self:Any, language:str='en') -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:List, references:List) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LVEvalOPTRougeEvaluator`
    Metodo: `__init__(self:Any, language:str='en') -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:List, references:List) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\lveval\lveval_cmrc_mixup.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LVEvalcmrcDataset`
    Metodo: `load(**kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\lveval\lveval_dureader_mixup.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LVEvaldureaderDataset`
    Metodo: `load(**kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\lveval\lveval_factrecall_en.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LVEvalfactrecallenDataset`
    Metodo: `load(**kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\lveval\lveval_factrecall_zh.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LVEvalfactrecallzhDataset`
    Metodo: `load(**kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\lveval\lveval_hotpotwikiqa_mixup.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LVEvalhotpotwikiqaDataset`
    Metodo: `load(**kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\lveval\lveval_lic_mixup.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LVEvallicDataset`
    Metodo: `load(**kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\lveval\lveval_loogle_CR_mixup.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LVEvallooglecrDataset`
    Metodo: `load(**kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\lveval\lveval_loogle_MIR_mixup.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LVEvallooglemirDataset`
    Metodo: `load(**kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\lveval\lveval_loogle_SD_mixup.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LVEvallooglesdDataset`
    Metodo: `load(**kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\lveval\lveval_multifieldqa_en_mixup.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LVEvalmultifieldqaenDataset`
    Metodo: `load(**kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\lveval\lveval_multifieldqa_zh_mixup.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LVEvalmultifieldqazhDataset`
    Metodo: `load(**kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\mastermath2024v1.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `MastermathDatasetv1`
    Metodo: `load(path:str, name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `MastermathDatasetv1Evaluator`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\matbench\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\matbench\matbench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `MatbenchDataset`
    Metodo: `load(path:Any, task:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `MatbenchEvaluator_regression`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `MatbenchEvaluator_classification`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `MatbenchEvaluator_classification_glass`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\matbench\post_process.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_numerical_final_results(judged_answers:Any, references:Any, origial_responses:Any, metric_name:Any='mae') -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `_numerical_postprocess(judgement:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `numerical_llmjudge_postprocess(output:dict, output_path:str) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `contains_elements_and_matches(sentence:Any, chem_elts:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `remove_formula(sentence:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `verify_float(number:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `parse_float_answer(sentence:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `parse_true_false_answer(raw_string:Any, option:Any='') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `parse_has_hasnot_answer(raw_string:Any, option:Any='') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\math.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `last_boxed_only_string(string:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `remove_boxed(s:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `extract_boxed_answer(pred_str:Any, strip_double_curly_brace:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `normalize_final_answer(final_answer:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `extract_answer(response_text:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `math_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `math_judement_preprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `math_postprocess_v2(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `MATHDataset`
    Metodo: `load(path:str, file_name:str='math.json', **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `MATHEvaluator`
    Metodo: `__init__(self:Any, version:Any='v1', pred_postprocessor:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_fix_fracs(self:Any, string:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_fix_a_slash_b(self:Any, string:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_remove_right_units(self:Any, string:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_fix_sqrt(self:Any, string:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_fix_sqrt_v2(self:Any, string:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_strip_string(self:Any, string:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_strip_string_v2(self:Any, string:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `is_equiv(self:Any, str1:Any, str2:Any, verbose:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `MATHAgentEvaluator`
    Metodo: `__init__(self:Any, action:str='PythonInterpreter', version:Any='v1') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `soft_equal(self:Any, pred:Any, refer:Any, step:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_action(self:Any, step:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `score(self:Any, predictions:Any, references:Any, steps:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\math401.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `check(a:Any, b:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `Math401Evaluator`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\math_intern.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `last_boxed_only_string(string:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `remove_boxed(s:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `extract_boxed_answer(pred_str:Any, strip_double_curly_brace:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `math_intern_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `fix_fracs(string:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `fix_a_slash_b(string:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `remove_right_units(string:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `fix_sqrt(string:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `strip_string(string:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `is_equiv(str1:Any, str2:Any, verbose:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `MATHInternDataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `MATHInternEvaluator`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `Extractor`
    Metodo: `extract_matching_bracket(cls:Any, target_str:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `clean(cls:Any, target_str:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `extract_answer(cls:Any, pred:str, extract_last_num:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\mathbench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_number(options:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `get_circular_example(entry:Any, id:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `mathbench_postprocess(text:str, name:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `first_option_postprocess(text:str, options:str, cushion:Any=True) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `remove_invisible_chars(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `MathBenchDataset`
    Metodo: `load(path:str, name:str, with_circular:bool=True) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `MathBenchBuggyDataset`
    Metodo: `load(path:str, name:str, with_circular:bool=True) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `MathBenchCircularEvaluator`
    Metodo: `__init__(self:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any, test_set:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\mbpp.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `swallow_io() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `time_limit(seconds:float) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_execution(programs:Any, timeout:Any, key:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `execution(programs:Any, task_id:Any, timeout:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `MBPPDataset`
    Metodo: `load(path:str, local_mode:bool=False) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `MBPPDatasetV2`
    Metodo: `load(path:str, num_repeats:int=1) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `SanitizedMBPPDataset`
    Metodo: `load(path:str, num_repeats:int=1) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `MBPPPlusDataset`
    Metodo: `load(path:str, num_repeats:int=1) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `TimeOutException`
  - `WriteOnlyStringIO`
    Metodo: `read(self:Any, *args:Any, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `readline(self:Any, *args:Any, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `readlines(self:Any, *args:Any, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `readable(self:Any, *args:Any, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `redirect_stdin`
  - `MBPPEvaluator`
    Metodo: `__init__(self:Any, metric:str='MBPP') -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_process_answer(self:Any, text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_process_test(self:Any, test_case:Any, pred:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `MBPPEvaluator2`
    Metodo: `_process_answer(self:Any, text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `MBPPPassKEvaluator`
    Metodo: `__init__(self:Any, k:Any=(1, 10, 100)) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `estimate_pass_at_k(num_samples:Union[int, List[int], np.ndarray], num_correct:Union[List[int], np.ndarray], k:int) -> np.ndarray`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\mbpp_pro.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `MBPPProDataset`
    Metodo: `load(path:Any, local_mode:Any=False) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `MBPPProEvaluator`
    Metodo: `score(self:Any, predictions:List, references:List, test_set:Dataset) -> Dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\medbench\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\medbench\constructions.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `TaskSchema`
    Metodo: `__init__(self:Any, passage:Any=None, question:Any=None, options:Any=None, label:Any=None, answer:Any=None, other:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `to_dict(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `MedBenchInstance`
    Metodo: `__init__(self:Any, task_description:Any, data_source:Any, task_schema:Any, output:Any, evaluation_metric:Any, task_example:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `to_dict(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `ChatGPTSchema`
    Metodo: `__init__(self:Any, context:Any=None, metadata:Any='') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `to_dict(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `ResultsForHumanSchema`
    Metodo: `__init__(self:Any, index:Any, problem_input:Any, label:Any, model_input:Any='', model_output:Any='', parse_result:Any='', first_stage_output:Any='', second_stage_input:Any='', is_correct:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `to_dict(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `to_tsv(result_list:Any, path:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\medbench\dataset_loader.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `convert_zero_shot(line:Any, dataset_name:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `combine_prompt(prompt_path:Any, dataset_name:Any, load_explanation:Any=True, chat_mode:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_lazy_load_enc() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `concat_prompt(demos:Any, dataset_name:Any, max_tokens:Any, end_of_example:Any='\n', verbose:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `concat_prompt_chat_mode(demos:Any, dataset_name:Any, max_tokens:Any, end_of_example:Any='\n', verbose:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `convert_few_shot(line:Any, dataset_name:Any, demo:Any, n_shot:Any, chat_mode:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `load_dataset(dataset_name:Any, setting_name:Any, parent_path:Any, prompt_path:Any=None, max_tokens:Any=None, end_of_example:Any='\n', chat_mode:Any=False, verbose:Any=False) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `generate_second_stage_input(dataset_name:Any, input_list:Any, output_list:Any, with_format_prompt:Any=False) -> Any`
    Descrizione: Genera output testo o sequenze.
  - `load_dataset_as_result_schema(dataset_name:Any, parent_path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\medbench\evaluation.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `convert_to_set(item:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `evaluate_single_sample(dataset_name:Any, prediction:Any, label:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\medbench\math_equivalence.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_fix_fracs(string:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_fix_a_slash_b(string:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_remove_right_units(string:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_fix_sqrt(string:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_strip_string(string:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `is_equiv(str1:Any, str2:Any, verbose:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\medbench\medbench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `process_generated_results_CMeEE(pred_file:Any) -> Any`
    Descrizione: Genera output testo o sequenze.
  - `process_generated_results_EMR(pred_file:Any) -> Any`
    Descrizione: Genera output testo o sequenze.
  - `process_generated_results_CMeIE(pred_file:Any) -> Any`
    Descrizione: Genera output testo o sequenze.
  - `process_generated_results_CDN(pred_file:Any) -> Any`
    Descrizione: Genera output testo o sequenze.
  - `process_generated_results_CDEE(pred_file:Any) -> Any`
    Descrizione: Genera output testo o sequenze.
  - `process_generated_results_CTC(pred_file:Any) -> Any`
    Descrizione: Genera output testo o sequenze.
  - `process_generated_results_doc_parsing(pred_file:Any) -> Any`
    Descrizione: Genera output testo o sequenze.
  - `process_generated_results_mrg(pred_file:Any) -> Any`
    Descrizione: Genera output testo o sequenze.
  - `calc_info_extract_task_scores(list_structured_predict:Any, list_structured_golden:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `calc_cls_task_scores(list_structured_golden:Any, list_structured_predict:Any, list_labels:Any=None, return_macro:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `calc_nlg_task_scores(list_structured_golden:Any, list_structured_predict:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `calc_scores_f1(dict_gt:Any, dict_pred:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `calc_scores_ctc(dict_gt:Any, dict_pred:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `calc_scores_nlg(dict_gt:Any, dict_pred:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `MedBenchDataset`
    Metodo: `load(path:str, name:str, setting_name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `MedBenchEvaluator`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `MedBenchEvaluator_CMeEE`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `MedBenchEvaluator_DBMHG`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `MedBenchEvaluator_IMCS_V2_MRG`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `MedBenchEvaluator_CMeIE`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `MedBenchEvaluator_CHIP_CDEE`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `MedBenchEvaluator_CHIP_CDN`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `MedBenchEvaluator_CHIP_CTC`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `MedBenchEvaluator_Doc_parsing`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `MedBenchEvaluator_NLG`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `MedBenchEvaluator_Cloze`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `MedBenchEvaluator_TF`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\medbench\post_process.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `extract_last_line(string:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `remove_few_shot_prefix(string:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `try_parse_few_shot_qa_single_answer(string:Any, setting_name:Any, language:Any='en') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `try_parse_few_shot_pattern(string:str, dataset_name:Any, setting_name:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `parse_few_shot_qa_single_answer(string:Any, setting_name:Any, language:Any='en') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `find_first_capital_letter(answer:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `extract_answer_in_bracket(answer:Any, prefix:Any='【', suffix:Any='】') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `parse_math_answer(setting_name:Any, raw_string:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `parse_qa_multiple_answer(string:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `post_process(dataset_name:Any, setting_name:Any, prediction:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\medbench\utils.py`
  Logica d'uso: Runtime helpers, device/dtype control, tiny fallback model/tokenizer.
  Funzioni:
  - `read_jsonl(path:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `save_jsonl(lines:Any, directory:Any) -> Any`
    Descrizione: Serializza e salva output su disco.
  - `extract_answer(js:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\Medbullets.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_parse(item:dict, prompt_mode:str) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `answer_cleansing(method:str, prediction:str, options:list, label:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_generic_llmjudge_postprocess(judgement:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `medbullets_llmjudge_postprocess(output:dict, output_path:str, dataset:Dataset) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `MedbulletsDataset`
    Metodo: `load(path:str, prompt_mode:str='zero-shot', **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `MedbulletsEvaluator`
    Metodo: `score(self:Any, predictions:Any, references:Any, test_set:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\MedCalc_Bench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `check_correctness(answer:str, ground_truth:Any, calid:Any, upper_limit:Any, lower_limit:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `extract_answer(answer:Any, calid:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_parse(item:Any, prompt_mode:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `MedCalc_BenchDataset`
    Metodo: `load(path:str, prompt_mode:str, **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `MedCalcOfficial_Evaluator`
    Metodo: `score(self:Any, predictions:Any, references:Any, test_set:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\medmcqa.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_parse(item:Any, prompt_mode:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `answer_cleansing(method:str, prediction:str, options:list, label:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_generic_llmjudge_postprocess(judgement:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `medmcqa_llmjudge_postprocess(output:dict, output_path:str, dataset:Dataset) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `MedmcqaDataset`
    Metodo: `load(path:str, prompt_mode:str='zero-shot', **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `MedmcqaEvaluator`
    Metodo: `score(self:Any, predictions:Any, references:Any, test_set:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\MedQA.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `MedQADataset`
    Metodo: `load_single(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\MedXpertQA.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_parse(item:Any, prompt_mode:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `answer_cleansing(method:str, prediction:str, options:list, label:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_generic_llmjudge_postprocess(judgement:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `MedXpertQA_llmjudge_postprocess(output:dict, output_path:str, dataset:Dataset) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `MedXpertQADataset`
    Metodo: `load(path:str, prompt_mode:str, **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `MedXpertQAEvaluator`
    Metodo: `score(self:Any, predictions:Any, references:Any, test_set:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\mgsm.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `mgsm_postprocess(text:str, lang:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `MGSMSDataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `MGSM_Evaluator`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\mmlu.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `MMLUDataset`
    Metodo: `load(path:str, name:str, **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `MMLUDatasetClean`
    Metodo: `load_contamination_annotations(path:Any, split:Any='val') -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
    Metodo: `load(path:str, name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\mmlu_cf.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `MMLUCFDataset`
    Metodo: `load(path:str, name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\mmlu_pro.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_parse(item:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `MMLUProDataset`
    Metodo: `load(path:str, category:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `MMLUProBaseEvaluator`
    Metodo: `is_equal(self:Any, pred:Any, refer:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\MMLUArabic.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `MMLUArabicDataset`
    Metodo: `load(path:str, name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\mmmlu.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `MMMLUDataset`
    Metodo: `load(path:str, name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `MMMLULiteDataset`
    Metodo: `load(path:str, name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\multipl_e.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `MultiplEDataset`
    Metodo: `load(path:str, language:str, tag:str='humaneval', local_mode:bool=False) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `MultiplEEvaluator`
    Metodo: `_stop_at_stop_token(self:Any, decoded_string:Any, stop_tokens:Any) -> Any`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
    Metodo: `_remove_prefix(self:Any, prompt:str, completion:str, threshold:float=0.95) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_process_completions(self:Any, test_case:Any, completion:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\multirc.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `MultiRCDataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `MultiRCDatasetV2`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\musr\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\musr\murder_mystery_solved_ex.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\musr\musr.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `MusrDataset`
    Metodo: `load(path:Any, name:Any, self_consistency_n:Any=1, exclude_contrastive_examples:Any=False, reverse_contrastive_sample:Any=False, skip_ablated:Any=False, randomize:Any=False, offset:Any=0, sample_size:Any=None, **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `MusrEvaluator`
    Metodo: `__init__(self:Any, answer_index_modifier:Any=1, self_consistency_n:Any=1, pred_postprocessor:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\musr\object_placements_solved_ex.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\musr\team_allocation_solved_ex.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\musr\tree.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LogicNodeOperatorType`
  - `LogicNodeFactType`
  - `LogicNodeConstraints`
  - `LogicNodeDeductionType`
  - `LogicNode`
    Metodo: `__init__(self:Any, value:str='', children:List['LogicNode']=None, operator:str=LogicNodeOperatorType.OR, fact_type:str=LogicNodeFactType.EXPLICIT, constraints:List[str]=(), deduction_type:str=None, prunable:bool=True, can_be_leaf:bool=False, frozen:bool=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `children(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `children(self:Any, children:List['LogicNode']) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__str__(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__repr__(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `to_json(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `from_json(cls:Any, js:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LogicTree`
    Metodo: `__init__(self:Any, chance_of_or:float=0.3, chance_of_cs_fact:float=0.1, depth:int=2, chance_to_prune:float=0.6, chance_to_prune_all:float=0.2, bf_factor:Dict[int, float]=None, deduction_type_sample_rate:Dict[LogicNodeDeductionType, float]=None, enforce_cs_fact_per_level:bool=False, root_structure:List[Any]=(), nodes:List[LogicNode]=(), populate:bool=True, prune:bool=True) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__str__(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_facts(self:Any, include_cs:bool=False, include_deductions_from_level:int=-1, no_facts_after_depth:int=-1) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `print_tree(self:Any, node:Any=None, level:Any=0) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `print_for_gpt(self:Any, node:Any=None, level:Any=0, pad_char:Any=' ', pad_space:Any=4, print_forward:Any=True, print_conjection_types:bool=False, print_reasoning_types:bool=False, ignore_value_after_depth:int=-1, print_only_nodes_with_value:bool=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `populate(self:Any, node:LogicNode, current_depth:int=1) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `prune(self:Any, node:LogicNode, current_depth:int=1) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `to_json(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `from_json(cls:Any, _js:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\narrativeqa.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `NarrativeQADataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\natural_question.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `NaturalQuestionDataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `NQOpenDataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `NQEvaluator`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\natural_question_cn.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `NaturalQuestionDatasetCN`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `NQEvaluatorCN`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\needlebench\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\needlebench\atc.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `NeedleBenchATCDataset`
    Metodo: `load(path:Any, file_name:str, num_needles:int, language:str, repeats:int) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `NeedleBenchATCOrderedDataset`
    Metodo: `load(path:Any, file_name:Any, num_needles:int, language:str, repeats:int) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\needlebench\atc_choice.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_number(options:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `get_circular_example(entry:Any, id:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  Classi:
  - `NeedleBenchATCDataset`
    Metodo: `load(path:str, file_name:str, num_needles:int, language:str, repeats:int, with_circular:bool=True) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\needlebench\multi.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_random_needles(counter:Any, file_path:Any, needle_count:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  Classi:
  - `NeedleBenchMultiDataset`
    Metodo: `load(path:str, length:int, depth:int, tokenizer_model:str, file_list:'list[str]', num_repeats_per_file:int, length_buffer:int, guide:bool, language:str, needle_file_name:str, num_needles:int, diff:int, position:str='End') -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `NeedleBenchMultiEvaluator`
    Metodo: `levenshtein_distance(self:Any, s1:Any, s2:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, gold:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\needlebench\origin.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_random_line_by_language(counter:Any, file_path:Any, language:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `needlebench_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `needlebench_dataset_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `NeedleBenchOriginDataset`
    Metodo: `load(path:str, length:int, depth:int, tokenizer_model:str, file_list:list[str], num_repeats_per_file:int, length_buffer:int, guide:bool, language:str, needle_file_name:str, position:str='End') -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `NeedleBenchOriginEvaluator`
    Metodo: `__init__(self:Any, use_trim:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_trim_prediction(prediction:Any, reference:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `levenshtein_distance(self:Any, s1:Any, s2:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, gold:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\needlebench\parallel.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_unique_entries(file_path:Any, n:Any, language:Any, unique_arg1:Any=False, unique_arg2:Any=False, unique_combination:Any=False) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  Classi:
  - `NeedleBenchParallelDataset`
    Metodo: `load(path:str, needle_file_name:str, length:int, depths:list[int], tokenizer_model:str, file_list:list[str], num_repeats_per_file:int, length_buffer:int, guide:bool, language:str, position:str='End') -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `NeedleBenchParallelEvaluator`
    Metodo: `levenshtein_distance(self:Any, s1:Any, s2:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, gold:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\needlebench_v2\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\needlebench_v2\atc.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `QuestionType`
  - `NeedleBenchATCDataset`
    Metodo: `load(path:Any, file_name:str, num_needles:int, language:str, repeats:int, question_types:list[QuestionType]=[QuestionType.ELDEST_ANCESTOR, QuestionType.NTH_ANCESTOR, QuestionType.NTH_DESCENDANT, QuestionType.RELATIONSHIP_DISTANCE]) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\needlebench_v2\atc_elder_only.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `clean_atc_answer(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `needlebench_atc_postprocess_v2(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `NeedleBenchATCDataset`
    Metodo: `load(path:Any, file_name:str, num_needles:int, language:str, repeats:int) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `NeedleBenchATCEvaluator`
    Metodo: `score(self:Any, predictions:Any, gold:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\needlebench_v2\multi.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_random_needles(counter:Any, file_path:Any, num_needles:Any, language:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  Classi:
  - `NeedleBenchMultiDataset`
    Metodo: `load(path:str, length:int, depth:int, tokenizer_model:str, file_list:'list[str]', num_repeats_per_file:int, length_buffer:int, language:str, needle_file_name:str, num_needles:int, diff:int, quesiton_position:str='End') -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\needlebench_v2\origin.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_random_line_by_language(counter:Any, file_path:Any, language:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `needlebench_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `needlebench_dataset_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `NeedleBenchOriginDataset`
    Metodo: `load(path:str, length:int, depth:int, tokenizer_model:str, file_list:list[str], num_repeats_per_file:int, length_buffer:int, language:str, needle_file_name:str, quesiton_position:str='End') -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `NeedleBenchOriginEvaluator`
    Metodo: `score(self:Any, predictions:Any, gold:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\needlebench_v2\parallel.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_unique_entries(file_path:Any, n:Any, language:Any, unique_arg1:Any=False, unique_arg2:Any=False, unique_combination:Any=False) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  Classi:
  - `NeedleBenchParallelDataset`
    Metodo: `load(path:str, needle_file_name:str, length:int, depths:list[int], tokenizer_model:str, file_list:list[str], num_repeats_per_file:int, length_buffer:int, language:str, quesiton_position:str='End') -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `NeedleBenchParallelEvaluator`
    Metodo: `score(self:Any, predictions:Any, gold:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\nejmaibench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_parse(item:Any, prompt_mode:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `answer_cleansing(method:str, prediction:str, options:list, label:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `NejmaibenchDataset`
    Metodo: `load(path:str, prompt_mode:str='zero-shot', **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `NejmaibenchEvaluator`
    Metodo: `score(self:Any, predictions:Any, references:Any, test_set:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\NPHardEval\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\NPHardEval\cmp_GCP_D.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `q2text(q:Any, p:Any=gcp_dPrompts) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `CMP_GCP_D_Dataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `CMP_GCP_D_Evaluator`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `parse_xml_to_dict(self:Any, xml_string:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `read_dimacs_format(self:Any, dimacs_str:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `gcp_greedy_solution(self:Any, adjacency_list:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `gcp_decision_check(self:Any, dimacs_str:Any, answer:Any, k_colors:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\NPHardEval\cmp_KSP.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `q2text(q:Any, p:Any=kspPrompts) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `CMP_KSP_Dataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `CMP_KSP_Evaluator`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `parse_xml_to_dict(self:Any, xml_string:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `ksp_optimal_solution(self:Any, knapsacks:Any, capacity:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `kspCheck(self:Any, instance:Any, solution:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\NPHardEval\cmp_TSP_D.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `q2text(adj_matrix:Any, distance_limit:Any, p:Any=tsp_dPrompts) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `CMP_TSP_D_Dataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `CMP_TSP_D_Evaluator`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `parse_xml_to_dict(self:Any, xml_string:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `tsp_approx(self:Any, distance_matrix:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `tsp_decision_check(self:Any, distance_matrix:Any, threshold:Any, tour:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\NPHardEval\hard_GCP.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `q2text(q:Any, p:Any=gcpPrompts) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `HardGCPDataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `HardGCPEvaluator`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `parse_xml_to_dict(self:Any, xml_string:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `gcpCheck(self:Any, dimacs_str:Any, answer_str:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `read_dimacs_format(self:Any, dimacs_str:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `parse_answer(self:Any, llm_string:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\NPHardEval\hard_MSP.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `q2text(q:Any, p:Any=mspPrompts) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `Hard_MSP_Dataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `Hard_MSP_Evaluator`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `mspCheck(self:Any, instance:Any, llm_string:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `parse_xml_to_dict(self:Any, xml_string:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\NPHardEval\hard_TSP.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `q2text(q:Any, p:Any=tspPrompts) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `Hard_TSP_Dataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `Hard_TSP_Evaluator`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `parse_xml_to_dict(self:Any, xml_string:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `tspCheck(self:Any, distance_matrix:Any, llm_string:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `greedy_tsp(self:Any, distance_matrix:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\NPHardEval\p_BSP.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `q2text(q:Any, p:Any=bspPrompts) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `P_BSP_Dataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `P_BSP_Evaluator`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `parse_xml_to_dict(self:Any, xml_string:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `bsp_check(self:Any, instance:Any, solution:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\NPHardEval\p_EDP.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `q2text(q:Any, p:Any=edpPrompts) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `P_EDP_Dataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `P_EDP_Evaluator`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `compute_min_edit_distance(self:Any, string_a:Any, string_b:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `edp_check(self:Any, instance:Any, solution:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `parse_xml_to_dict(self:Any, xml_string:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\NPHardEval\p_SPP.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `q2text(q:Any, p:Any=sppPrompts) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `P_SPP_Dataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `P_SPP_Evaluator`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `parse_xml_to_dict(self:Any, xml_string:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `ssp_optimal_solution(self:Any, instance:Any, source:Any, target:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `spp_check(self:Any, instance:Any, solution:Any, start_node:Any=None, end_node:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\NPHardEval\prompts.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\NPHardEval\utils.py`
  Logica d'uso: Runtime helpers, device/dtype control, tiny fallback model/tokenizer.
  Funzioni:
  - `append_root_tags(string:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `parse_xml_to_dict(xml_string:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\obqa.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `OBQADataset`
    Metodo: `load(path:Any, name:Any='main') -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `OBQADatasetV2`
    Metodo: `load(path:Any, name:Any='main') -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\olymmath.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `OlymMATHDataset`
    Metodo: `load(path:str, subset:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\OlympiadBench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_single_answer_type_text(answer_type:Any, is_chinese:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `get_answer_type_text(answer_type:Any, is_chinese:Any, multiple_answer:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `olympiadbench_postprocess_v2(text:str, is_chinese:bool=False, is_deepseek:bool=False) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `OlympiadBenchDataset`
    Metodo: `load(path:str, name:str=None, **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `OlympiadBenchPrompter`
    Metodo: `__init__(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `make_prompt(self:Any, language:Any, subject:Any, question_type:Any, answer_type:Any, is_multiple_answer:Any, unit:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `MathJudger`
    Metodo: `__init__(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `split_by_comma(self:Any, expr:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `trans_plus_minus_sign(self:Any, expr_list:list) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `judge(self:Any, expression1:Any, expression2:Any, precision:Any=1e-08) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `is_interval(self:Any, epr:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `sympy_sub_pi(self:Any, expression_sympy:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `is_equal(self:Any, expression1:Any, expression2:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `numerical_equal(self:Any, expression1:str, expression2:str, include_percentage:bool=True) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `expression_equal(self:Any, exp1:Any, exp2:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `equation_equal(self:Any, expression1:Any, expression2:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `interval_equal(self:Any, expression1:Any, expression2:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `preprocess(self:Any, expression1:Any, expression2:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `can_compute_power(self:Any, expr:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `OlympiadBenchEvaluator`
    Metodo: `__init__(self:Any, version:Any='v1') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `OlympiadBenchTemplate`
    Metodo: `__init__(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate_item(self:Any, entry:Dict, *args:Any, **kwargs:Any) -> str`
    Descrizione: Genera output testo o sequenze.

- File: `vendor\llada\opencompass\opencompass\datasets\omni_math.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `OmniMathDataset`
    Metodo: `load() -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `OmniMathEvaluator`
    Metodo: `__init__(self:Any, url:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `batch_infer(self:Any, models:List[TurboMindAPIModel], inputs:List[str]) -> List[str]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `parse_response(self:Any, response:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any, origin_prompt:Any, test_set:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\OpenFinData.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `OpenFinDataDataset`
    Metodo: `load(path:str, name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `OpenFinDataKWEvaluator`
    Metodo: `__init__(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\phybench\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\phybench\box_extract.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `extract_boxed_latex(prediction:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\phybench\EED.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `update_func(x:Any, y:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `remove_func(x:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `remove_tree_func(x:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `insert_func(x:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `insert_tree_func(x:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `calc_tree_size(node:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `score_calc(tree_dist:Any, tree_size:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `simplify_with_timeout(expr:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `time_simplify(expr:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `equal_with_timeout(expr1:Any, expr2:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `time_equal(expr1:Any, expr2:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `sympy_to_tree(expr:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `print_tree(node:Any, indent:Any=0) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `EED(answer_latex:Any, test_latex:Any, debug_mode:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `TreeNode`
    Metodo: `__init__(self:Any, label:Any, children:Any=None, node_type:Any='other') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_children(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `__str__(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LaTeXError`
    Metodo: `__init__(self:Any, message:Any='LaTeXError') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `SymPyError`
    Metodo: `__init__(self:Any, message:Any='SymPyError') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `TreeError`
    Metodo: `__init__(self:Any, message:Any='TreeError') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `DistError`
    Metodo: `__init__(self:Any, message:Any='DistanceError') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\phybench\extended_zss.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `ext_distance(A:Any, B:Any, get_children:Any, single_insert_cost:Any, insert_cost:Any, single_remove_cost:Any, remove_cost:Any, update_cost:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `Node`
    Metodo: `__init__(self:Any, label:Any, children:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_children(node:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_label(node:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `addkid(self:Any, node:Any, before:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get(self:Any, label:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `AnnotatedTree`
    Metodo: `__init__(self:Any, root:Any, get_children:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\phybench\latex_pre_process.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `brackets_balanced(s:str) -> bool`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `remove_non_ascii(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `extract_bracket_content(s:str, bracket_position:int) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `find_first_unescaped_brace(s:str) -> int`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `extract_command(s:str, brace_pos:int) -> str | None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `remove_command(s:Any, command:Any, keep_inside:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `convert_latex_fractions(latex_str:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_first_brace_command(s:str) -> str | None`
    Descrizione: Recupera valore/stato calcolato.
  - `remove_overall_brace(s:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `exp_frac(s:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `find_all(s:Any, sub_str:Any, allow_overlap:Any=True) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `bar_inside_vec(s:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `vec_lower_idx(input_str:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `convert_vec_syntax(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `remove_outer_braces(tex_str:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `extract_last_equal_content(s:str, strip_whitespace:bool=True) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `first_pre_process(s:Any, extrac_box:Any=True) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `second_pre_process(s:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `master_convert(s:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `MyConfig`
  - `MyNormalization`

- File: `vendor\llada\opencompass\opencompass\datasets\phybench\phybench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `PhyBenchDataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `MathEEDEvaluator`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\physics.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `PHYSICSDataset`
    Metodo: `load(path:str, name:str, **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\piqa.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `PIQADataset`
    Metodo: `load_single(path:Any, data_filename:Any, label_filename:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `PIQADatasetV2`
    Metodo: `load_single(path:Any, data_filename:Any, label_filename:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `PIQADatasetV3`
    Metodo: `load_single(path:Any, data_filename:Any, label_filename:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\PMMEval\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\PMMEval\flores.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `wmt_postprocess(text:str, lang:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `compute_maximum_bleu_value(gen:str, ref:str, lang:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `trim_multiple_space(tokes:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `build_tokenizer(lang:str) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
  - `tokenize(sent:Any, lang:Any) -> Any`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
  - `pmmeval_flores_postprocess(text:str, lang_fullname:str) -> Tuple[str]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `SpaceTokenizer`
    Metodo: `__call__(self:Any, sent:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `NonASCIITokenizer`
    Metodo: `__init__(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__call__(self:Any, sent:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `PMMEvalFloresDataset`
    Metodo: `load(path:str, lang_fullname:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `PMMEvalFloresEvaluator`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\PMMEval\humanevalxl.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_clean_up_code(text:str, language_type:str, reference:Any) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `PMMEvalHumanEvalXLDataset`
    Metodo: `load(path:str, lang:str, program_lang:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `PMMEvalHumanEvalXLEvaluator`
    Metodo: `__init__(self:Any, language:Any, ip_address:Any='localhost', text_language:Any='', port:Any='', retry:Any=2, timeout:Any=600) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_code_eval_service(self:Any, file_path:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\PMMEval\mgsm.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_get_last_digit(s:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `PMMEvalMGSMDataset`
    Metodo: `load(path:str, lang:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `PMMEvalMGSMEvaluator`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\PMMEval\mhellaswag.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `extract_choice(gen:Any, lang:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `extract_choice_fuzzy(gen:Any, lang:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `pmmeval_mhellaswag_postprocess(text:str, lang_code:str) -> Tuple[str]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `PMMEvalMHellaswagDataset`
    Metodo: `load(path:str, lang:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `PMMEvalMHellaswagEvaluator`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\PMMEval\mifeval.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `test_instruction_following_strict(inp:Any, response:Any, lang_code:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `test_instruction_following_loose(inp:Any, response:Any, lang_code:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `pmmeval_mifeval_postprocess(text:str, lang_code:str) -> Tuple[str]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `PMMEvalMIFEvalDataset`
    Metodo: `load(path:str, lang:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `PMMEvalMIFEvalEvaluator`
    Metodo: `score(self:Any, predictions:Any, references:Any, test_set:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\PMMEval\mifeval_utils\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\PMMEval\mifeval_utils\combination_checker.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `repeat_prompt_checker(input_string:str, prompt_to_repeat:str, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `two_responses_checker(input_string:str, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\PMMEval\mifeval_utils\detectable_content_checker.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `number_placeholders_checker(input_string:str, num_placeholders:int, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `postscript_checker(input_string:str, postscript_marker:str, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\PMMEval\mifeval_utils\detectable_format_checker.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `removeprefix(s:Any, prefix:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `removesuffix(s:Any, suffix:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `constrained_response_checker(input_string:str, lang_code:str, **kwargs:Any) -> Any`
    Descrizione: Esegue passaggi di training/update.
  - `number_bullet_lists_checker(input_string:str, num_bullets:int, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `number_highlighted_sections_checker(input_string:str, num_highlights:int, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `title_checker(input_string:str, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `json_format_checker(input_string:str, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\PMMEval\mifeval_utils\keywords_checker.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `forbidden_words_checker(input_string:str, forbidden_words:list, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\PMMEval\mifeval_utils\length_constraints_checker.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `nth_paragraph_first_word_checker(input_string:str, num_paragraphs:int, nth_paragraph:int, first_word:str, lang_code:str, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `number_paragraphs_checker(input_string:str, num_paragraphs:int, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `number_sentences_checker(input_string:str, relation:str, num_sentences:int, lang_code:str, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `number_words_checker(input_string:str, relation:str, num_words:int, lang_code:str, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\PMMEval\mifeval_utils\punctuation_checker.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `no_comma_checker(input_string:str, lang_code:str, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\PMMEval\mifeval_utils\startend_checker.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `end_checker_checker(input_string:str, end_phrase:str, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `quotation_checker(input_string:str, lang_code:str, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\PMMEval\mlogiqa.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `extract_choice(gen:Any, lang:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `extract_choice_fuzzy(gen:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `pmmeval_mlogiqa_postprocess(text:str, lang_code:str) -> Tuple[str]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `PMMEvalMLogiQADataset`
    Metodo: `load(path:str, lang:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `PMMEvalMLogiQAEvaluator`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\PMMEval\mmmlu.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `extract_choice(gen:Any, lang:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `extract_choice_fuzzy(gen:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `pmmeval_mmmlu_postprocess(text:str, lang_code:str) -> Tuple[str]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `PMMEvalMMMLUDataset`
    Metodo: `load(path:str, lang:str, difficulty:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `PMMEvalMMMLUEvaluator`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\PMMEval\xnli.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `extract_choice(gen:Any, lang:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `extract_choice_fuzzy(gen:Any, lang:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `pmmeval_xnli_postprocess(text:str, lang_code:str) -> Tuple[str]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `PMMEvalXNLIDataset`
    Metodo: `load(path:str, lang:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `PMMEvalXNLIEvaluator`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\ProteinLMBench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_parse(item:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `ProteinLMBenchDataset`
    Metodo: `load(path:str, **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `ProteinLMBenchEvaluator`
    Metodo: `score(self:Any, predictions:Any, references:Any, test_set:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\PubMedQA.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `PubMedQADataset`
    Metodo: `load_single(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\py150.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `py150_post_process(code:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `Py150Dataset`
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\qasper.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `QASPERDataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\qaspercut.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `QASPERCUTDataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\QuALITY.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `QuALITYDataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `QuALITYEvaluator`
    Metodo: `score(self:Any, predictions:Any, references:Any, test_set:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\race.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `RaceDataset`
    Metodo: `load(path:str, name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\rbench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `RBenchDataset`
    Metodo: `load_single(path:Any, subset:Any='en') -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
    Metodo: `load(path:Any, subset:Any='en', **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\realtoxicprompts.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `RealToxicPromptsDataset`
    Metodo: `load(**kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\reasonbench\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\reasonbench\ReasonBenchDataset.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `ReasonBenchDataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\record.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `ReCoRD_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `ReCoRDDataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `ReCoRDDatasetV2`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\rolebench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `RoleBenchBaseDataset`
    Metodo: `load_single(source_file:Any, desc_list:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
    Metodo: `load_desc(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
    Metodo: `load_dataset(path:Any, desc_list:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `InstructionGeneralizationEnglishDataset`
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `RoleGeneralizationEnglishDataset`
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `InstructionGeneralizationChineseDataset`
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\ruler\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\ruler\ruler_cwe.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `RulerCweDataset`
    Metodo: `load(max_seq_length:int=4096, tokenizer_model:str='gpt-4', template:str='Below is a numbered list of words. In these words, some appear more often than others. Memorize the ones that appear most often.\n{context}\nQuestion: What are the 10 most common words in the above list? Answer: The top 10 words that appear most often in the list are:', tokens_to_generate:int=120, freq_cw:int=30, freq_ucw:int=3, num_cw:int=10, num_samples:int=500, random_seed:int=42, remove_newline_tab:str='') -> Dataset`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `RulerCweEvaluator`
    Metodo: `score(self:Any, predictions:Any, gold:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\ruler\ruler_fwe.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `RulerFweDataset`
    Metodo: `load(max_seq_length:int=4096, tokenizer_model:str='gpt-4', template:str="Read the following coded text and track the frequency of each coded word. Find the three most frequently appeared coded words. {context}\nQuestion: Do not provide any explanation. Please ignore the dots '....'. What are the three most frequently appeared words in the above coded text? Answer: According to the coded text above, the three most frequently appeared words are:", tokens_to_generate:int=50, alpha:float=2.0, coded_wordlen:int=6, num_samples:int=500, random_seed:int=42, remove_newline_tab:str='', vocab_size:int=-1) -> Dataset`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `RulerFweEvaluator`
    Metodo: `score(self:Any, predictions:Any, gold:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\ruler\ruler_niah.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `RulerNiahDataset`
    Metodo: `load(base_path:str, file_path:str, tokens_to_generate:int=128, max_seq_length:int=4096, tokenizer_model:str='gpt-4', num_samples:int=500, random_seed:int=42, template:str='Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n{context}\nWhat are all the special magic {type_needle_v} for {query} mentioned in the provided text? The special magic {type_needle_v} for {query} mentioned in the provided text are', num_needle_k:int=1, num_needle_v:int=1, num_needle_q:int=1, type_haystack:str='essay', type_needle_k:str='words', type_needle_v:str='numbers', remove_newline_tab:str='') -> Dataset`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `RulerNiahEvaluator`
    Metodo: `score(self:Any, predictions:Any, gold:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\ruler\ruler_qa.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `RulerQaDataset`
    Metodo: `load(path:str, dataset:str='squad', max_seq_length:int=4096, tokenizer_model:str='gpt-4', template:str='Answer the question based on the given documents. Only give me the answer and do not output any other words.\n\nThe following are given documents.\n\n{context}\n\nAnswer the question based on the given documents. Only give me the answer and do not output any other words.\n\nQuestion: {query} Answer:', tokens_to_generate:int=32, num_samples:int=500, pre_samples:int=0, random_seed:int=42, remove_newline_tab:str='') -> Dataset`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `RulerQaEvaluator`
    Metodo: `score(self:Any, predictions:Any, gold:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\ruler\ruler_vt.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `RulerVtDataset`
    Metodo: `load(max_seq_length:int=4096, tokenizer_model:str='gpt-4', template:str='Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n{context}\nQuestion: Find all variables that are assigned the value {query} in the text above. Answer: According to the chain(s) of variable assignment in the text above, {num_v} variables are assigned the value {query}, they are: ', tokens_to_generate:int=30, num_chains:int=1, num_hops:int=4, num_samples:int=500, random_seed:int=42, remove_newline_tab:str='') -> Dataset`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `RulerVtEvaluator`
    Metodo: `score(self:Any, predictions:Any, gold:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\s3eval.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `S3EvalDataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `S3EvalEvaluator`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\safety.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `SafetyDataset`
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\sage\dataset_loader.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `SAGEDataset`
    Metodo: `load(split:str='val') -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\sage\evaluation.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `fix_json_slash(s:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `sage_pred_postprocess(prediction:str, think_tags:Tuple[str, str]=('<think>', '</think>')) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_final_results(parsed_judges:List[List[Dict]], references:List[List[str]], origial_judges:List[List[str]]) -> Dict`
    Descrizione: Recupera valore/stato calcolato.
  - `process_judge_output(output:Dict, think_tags:Tuple[str, str]=('<think>', '</think>')) -> Tuple[List[str], List[Dict], List[str]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `sage_judge_postprocess(output:List[Dict], output_path:str, think_tags:Tuple[str, str]=('<think>', '</think>')) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `SAGELLMEvaluator`
    Metodo: `__init__(self:Any, prompt_template:ConfigDict, judge_cfg:List[ConfigDict], dataset_cfg:Optional[ConfigDict]=None, pred_postprocessor:Optional[ConfigDict]=None, dict_postprocessor:Optional[ConfigDict]=None, keep_predictions:bool=False) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `build_inferencer(self:Any) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
    Metodo: `score(self:Any, predictions:Any, references:Optional[List]=None, test_set:Optional[Dataset]=None) -> Dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `output_postprocess(self:Any, output:Dict, dataset:Any=None) -> Dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `default_judge_cfg(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\sage\prompt.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\scibench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `scibench_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `ScibenchDataset`
    Metodo: `load(path:str, name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\scicode.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `process_hdf5_list(group:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `process_hdf5_dict(group:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `process_hdf5_sparse_matrix(group:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `process_hdf5_datagroup(group:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `process_hdf5_to_tuple(step_id:Any, test_num:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `are_dicts_close(dict1:Any, dict2:Any, atol:Any=1e-08, rtol:Any=1e-05) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `process_symbol_in_dict(dict:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `are_csc_matrix_close(matrix1:Any, matrix2:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `cmp_tuple_or_list(var1:Any, var2:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `SciCodeDataset`
    Metodo: `load(path:Any, with_bg:Any, **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
    Metodo: `return_dataset(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `SciCodeEvaluator`
    Metodo: `__init__(self:Any, dataset_path:Any, with_bg:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `extract_python_script(self:Any, response:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `run_script(self:Any, script_path:Any) -> Any`
    Descrizione: Esegue pipeline o job completo.
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\ScienceQA.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `ScienceQADataset`
    Metodo: `load_single(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\SciEval.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `SciEvalDataset`
    Metodo: `load(path:str, name:str, **kwargs:Any) -> DatasetDict`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\SciKnowEval.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_parse(item:Any, prompt_mode:Any, discipline:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `answer_cleansing(method:str, prediction:str, options:list, label:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `SciKnowEvalDataset`
    Metodo: `load(path:str, prompt_mode:str, **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `SciKnowEvalEvaluator`
    Metodo: `score(self:Any, predictions:Any, references:Any, test_set:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\simpleqa.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_final_results(judged_answers:Any, references:Any, origial_responses:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `_single_simpleqa_postprocess(judgement:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `simpleqa_postprocess(output:dict, output_path:str) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `SimpleQADataset`
    Metodo: `load(path:str, num_examples:int | None=None, n_repeats:int=1, **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\siqa.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `siqaDataset`
    Metodo: `load_single(path:Any, data_filename:Any, label_filename:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `siqaDataset_V2`
    Metodo: `load_single(path:Any, data_filename:Any, label_filename:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `SiqaDatasetV3`
    Metodo: `load_single(path:Any, data_filename:Any, label_filename:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\smolinstruct.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `extract_chemical_data(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `parse_molecule(molecular_formula:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `calculate_single_element_match_for_list(predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `calculate_single_element_match(predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `extract_number(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `smolinstruct_acc_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `smolinstruct_acc_0shot_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `SmolInstructDataset`
    Metodo: `load(path:str, name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `NCElementMatchEvaluator`
    Metodo: `__init__(self:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `NCExactMatchEvaluator`
    Metodo: `__init__(self:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `RMSEEvaluator`
    Metodo: `__init__(self:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `FTSEvaluator`
    Metodo: `__init__(self:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `MeteorEvaluator`
    Metodo: `__init__(self:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\squad20.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `SQuAD20Dataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `SQuAD20Evaluator`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\srbench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `mydataset_postprocess(formula_str:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `change_data_to_prompt(points:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `SRbenchDataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `SRbenchDatasetEvaluator`
    Metodo: `__init__(self:Any, path:Any='') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `parse_formula(self:Any, formula_str:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate_samples(self:Any, x0_range:Any=(-10, 10), x1_range:Any=(-10, 10), num_points:Any=1000) -> Any`
    Descrizione: Genera output testo o sequenze.
    Metodo: `is_symbolically_equivalent(self:Any, formula1:Any, formula2:Any, n_var:Any=2) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\storycloze.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `StoryClozeDataset`
    Metodo: `load(path:Any, lang:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `StoryClozeDatasetV2`
    Metodo: `load(path:Any, lang:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\strategyqa.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `strategyqa_pred_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `strategyqa_dataset_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `StrategyQADataset`
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\subjective\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\subjective\alignbench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `prompt_construct(sample:Any, config:Config) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `detect_mapping(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `extract_missing_rating(text:Any, search_type:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `extract_rating(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `check_rating(rating:Any, all_dimensions:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `post_process_alignbench(judgement:dict, all_dimensions:Any=All_Dimensions, possible_keys:Any=['综合得分']) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_dimension_results(judged_answers:Any, references:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `get_capability_results(judged_answers:Any, references:Any, categories:Any=CATEGORIES) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `alignbench_postprocess(output:dict, output_path:str, judge_type:Optional[str]='general') -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `Config`
    Metodo: `__init__(self:Any, alignment_bench_config_path:Any, alignment_bench_config_name:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `category2dimensions(self:Any, category:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `dimension2def(self:Any, dimension:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `category2type(self:Any, category:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `AlignmentBenchDataset`
    Metodo: `load(self:Any, path:str, name:str, alignment_bench_config_path:Optional[str]='', alignment_bench_config_name:Optional[str]='', *args:Any, **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\subjective\alpacaeval.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `post_process_alpacav2(completion:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `alpacaeval_postprocess(output:dict, output_path:str) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `alpacaeval_bradleyterry_postprocess(output:dict, output_path:str) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `AlpacaEvalDataset`
    Metodo: `load(self:Any, path:str, name:str, *args:Any, **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\subjective\arena_hard.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `post_process_arenahard(completion:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_battles_from_judgment(judged_answers:Any, references:Any, WEIGHT:Any=3) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `compute_mle_elo(df:Any, SCALE:Any=400, BASE:Any=10, INIT_RATING:Any=1000) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_bootstrap_result(battles:Any, func_compute_elo:Any, num_round:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `preety_print_two_ratings(ratings_1:Any, ratings_2:Any, column_names:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `predict_win_rate(elo_ratings:Any, SCALE:Any=400, BASE:Any=10, INIT_RATING:Any=1000) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_win_rate_column(df:Any, column:Any, baseline:Any='gpt4-0314') -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `arenahard_postprocess(output:dict, output_path:str) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `arenahard_bradleyterry_postprocess(output:dict, output_path:str) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `ArenaHardDataset`
    Metodo: `load(self:Any, path:str, name:str, *args:Any, **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\subjective\commonbench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `post_process(judgement:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_capability_results(judged_answers:Any, references:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `commonbench_postprocess(output:dict, output_path:str, post_process:Optional[callable]=post_process) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\subjective\compass_arena.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `check_position_bias(judged_answers:Any, references:Any, banned_choice:Any=['C']) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `post_process_compassarena(item:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `compassarena_postprocess(output:dict, output_path:str, summary_type:Any='single', check_pos_bias:Any=True) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `compassarena_bradleyterry_postprocess(output:dict, output_path:str, count_ties:bool=True) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `CompassArenaDataset`
    Metodo: `load(self:Any, path:str, name:str, *args:Any, **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\subjective\compass_arena_subjective_bench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `post_process_pairwise(completion:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `post_process_pointwise(completion:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `compassarena_subjectiveeval_pointwise_postprocess(output:dict, output_path:str) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `compassarena_subjectiveeval_pairwise_postprocess(output:dict, output_path:str) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `count_style_elements(text:str, suffix:str='', encoder_model:str='gpt-3.5-turbo', code_pattern:str='```([^`]*)```') -> Dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `process_convo_for_style_elements(conversation:Union[str, List], code_pattern:str='```([^`]*)```', suffix:str='') -> Dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_element_counts(data:List[Dict], column:str, suffix:str='', code_pattern:str='```([^`]*)```') -> List[Dict]`
    Descrizione: Recupera valore/stato calcolato.
  - `compassarena_subjectiveeval_bradleyterry_postprocess(output:dict, output_path:str) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `CompassArenaSubjectiveBench`
    Metodo: `load(self:Any, path:str, name:str, *args:Any, **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\subjective\compassbench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `CompassBenchDataset`
    Metodo: `load(self:Any, path:str, name:str, *args:Any, **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\subjective\compassbench_checklist.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `CompassBenchCheklistDataset`
    Metodo: `load(self:Any, path:str, name:str, *args:Any, **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\subjective\compassbench_control_length_bias.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `CompassBenchControlLengthBiasDataset`
    Metodo: `load(self:Any, path:str, name:str, *args:Any, **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\subjective\corev2.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `cn_string(s:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `build_prompt_cn(item:Any, prompt:Any, ics:Any) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
  - `build_prompt_en(item:Any, prompt:Any, ics:Any) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
  - `build_prompt(item:Any, nopt:Any=4, multi_lang:Any=True) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
  Classi:
  - `Corev2Dataset`
    Metodo: `load(self:Any, path:str, name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\subjective\creationbench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `prompt_construct(sample:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `prompt_construct_score_with_ref(sample:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `prompt_construct_compare(sample:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `prompt_construct_compare_4opt(sample:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `CreationBenchDataset`
    Metodo: `load(self:Any, path:str, name:str, multi_dimension:Optional[bool]=False) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\subjective\flames.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `prompt_construct(sample:Any, config:Config) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `Config`
    Metodo: `__init__(self:Any, flames_config_path:Any, flames_bench_config_name:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `FlamesDataset`
    Metodo: `load(self:Any, path:str, name:str, *args:Any, **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\subjective\fofo.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `post_process_fofo(judgement:dict) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `fofo_postprocess(output:dict, output_path:str) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `FofoDataset`
    Metodo: `load(self:Any, path:str, name:str, *args:Any, **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\subjective\followbench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `post_process_followbench(item:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_scores(judged_answers:Any, references:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `followbench_postprocess(output:dict, output_path:str) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `FollowBenchDataset`
    Metodo: `load(self:Any, path:str, name:str, cate:str, *args:Any, **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\subjective\hellobench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `post_process_hellobench(judgement:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_judgeanswer(result:Any, filename:Any, post_process:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `hellobench_postprocess(output:dict, output_path:str) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `HelloBenchDataset`
    Metodo: `load(self:Any, path:str, category_name:str, *args:Any, **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\subjective\judgerbench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `parse_conversation(conversation:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `JudgerBenchDataset`
    Metodo: `load(self:Any, path:str, name:str, *args:Any, **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `JudgerBenchEvaluator`
    Metodo: `__init__(self:Any, num_workers:Any=16) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_judge_result(self:Any, judge:Any, dataset_name:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\subjective\mtbench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `prompt_construct(problem:Any, multi_turn:Any=False, judge_type:Any='single') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `post_process_mtbench(judgement:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_capability_results(judged_answers:Any, references:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `mtbench_postprocess(output:dict, output_path:str) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `MTBenchDataset`
    Metodo: `load(self:Any, path:str, name:str, judge_type:Any='single', multi_turn:Any=True, *args:Any, **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\subjective\mtbench101.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `eval_prompt_construct(task:Any, ref_answer:Any, history:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `add_format(question:Any, answer:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `post_process_mtbench101(judgement:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_final_results(judged_answers:Any, references:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `mtbench101_postprocess(output:dict, output_path:str) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `MTBench101Dataset`
    Metodo: `load(self:Any, path:str, name:str, *args:Any, **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\subjective\multiround.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `prompt_construct(sample:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `MultiroundDataset`
    Metodo: `load(self:Any, path:str, name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\subjective\subjective_cmp.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `SubjectiveCmpDataset`
    Metodo: `load(self:Any, path:str, name:str, *args:Any, **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\subjective\utils.py`
  Logica d'uso: Runtime helpers, device/dtype control, tiny fallback model/tokenizer.
  Funzioni:
  - `get_judgeanswer_and_reference(result:Any, filename:Any, post_process:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\datasets\subjective\wildbench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `parse_conversation(conversation:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `post_process_wildbench_pair(judgement:dict) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `post_process_wildbench_single(judgement:dict) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `wildbench_postprocess(output:dict, output_path:str) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `wildbench_bradleyterry_postprocess(output:dict, output_path:str) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `WildBenchDataset`
    Metodo: `load(self:Any, path:str, K:Any=-1, eval_mode:Any='pair', *args:Any, **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\subjective\writingbench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `post_process_writingbench(judgement:dict) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `writingbench_postprocess(output:dict, output_path:str) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `WritingBenchDataset`
    Metodo: `load(self:Any, path:str, name:str, *args:Any, **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\summedits.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `SummeditsDataset_V2`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\summscreen.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `SummScreenDataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\supergpqa\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\supergpqa\supergpqa.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_parse(item:Any, template:Any, prompt_mode:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_generic_llmjudge_postprocess(judgement:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `supergpqa_llmjudge_postprocess(output:dict, output_path:str, dataset:Dataset) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `SuperGPQADataset`
    Metodo: `load(path:str, prompt_mode:str, discipline:str=None, field:str=None, subfield:str=None, **kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `SuperGPQAEvaluator`
    Metodo: `__init__(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any, test_set:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\supergpqa\supergpqa_dataset_config\config_wrapper.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `initialize_config(config_path:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_config_wrapper() -> Any`
    Descrizione: Recupera valore/stato calcolato.
  Classi:
  - `ConfigWrapper`
    Metodo: `__init__(self:Any, config_path:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__setattr__(self:Any, key:Any, value:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__getattr__(self:Any, key:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_id(self:Any, data:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `print_all_keys(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\supergpqa\supergpqa_eval.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `safe_regex_search(pattern:Any, text:Any, flags:Any=0) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `extract_option_labels(text:Any, options:Any='ABCDEFGHIJ') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `extract_option_content(text:Any, options_content:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\supergpqa\supergpqa_utils.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `load_yaml(yaml_path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `load_json_or_jsonl(file_path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `find_file(base_path:Any, sub_path:Any, extensions:Any=('json', 'jsonl')) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `load_json_or_jsonl_with_idx(data_path:Any, split:Any='', idx:Any=None) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `load_split_data(base_path:Any, split_name:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `process_mixed_data(base_path:Any, mode:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `initialize_config(config_path:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_config_wrapper() -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `read_yaml(config:Any='default') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `write_jsonl_lines(file:Any, data:Any) -> Any`
    Descrizione: Serializza e salva output su disco.
  - `print_info(info:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `read_json_or_jsonl(data_path:Any, split:Any='', mapping_key:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `read_json_or_jsonl_with_idx(data_path:Any, split:Any='', idx:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `clean_json_string(json_str:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `is_in_idx_ranges(idx:Any, idx_ranges:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `extract_json(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `extract_all_responses_from_json(response_json:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `clean_latex(latex_expr:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `extract_text_from_brackets(text:Any, clean_level:Any='basic') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `extract_inner_text_from_brackets(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `extract_numbers(str:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `extract_and_sort_inequalities(latex_expr:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `rule5_normalize_content(content:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `normalize_string(s:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `remove_commas_and_spaces(s:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `remove_non_alphanumeric(s:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `contains_or(answer:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `compare_multi_results(response:Any, answer:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `split_or_expression(expression:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `compare_math_expressions(response:Any, answer:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `method_equal(response_text:Any, answer:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `method_1(response_text:Any, answer:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `method_2(response_text:Any, answer:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `method_3(response_text:Any, answer:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `method_4(response_text:Any, answer:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `method_5(response_text:Any, answer:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `method_9(response_text:Any, answer:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `method_10(response_text:Any, answer:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `method_18(response_text:Any, answer:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `method_general(response_text:Any, answer:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `evaluate_response_vs_answer(response:Any, answer:Any, question_type:Any, rule_id:Any, idx:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `compute_one_mixed_question_pass_rate(idx:Any, question_list:Any, response_json:Any, base_path:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `evaluate_responses(data:Any, mode:Any, base_path:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `ConfigWrapper`
    Metodo: `__init__(self:Any, config_path:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__setattr__(self:Any, key:Any, value:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__getattr__(self:Any, key:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_id(self:Any, data:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `print_all_keys(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\svamp.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `SVAMPDataset`
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\tabmwp.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_table_text(problem:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `get_question_text(problem:Any, option_inds:Any='ABCDEFGH') -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `get_answer(problem:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `get_choices(problem:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `get_unit(problem:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `get_solution_text(problem:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `normalize_answer(text:Any, unit:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `score_string_similarity(str1:Any, str2:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `extract_prediction(output:Any, options:Any=None, option_inds:Any='ABCDEFGH') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `TabMWPEvaluator`
    Metodo: `_preprocess(self:Any, predictions:List, references:List) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `TabMWPDataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\taco.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `timeout_handler(signum:Any, frame:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `run_test(sample:Any, test:Any=None, debug:Any=False) -> Any`
    Descrizione: Esegue pipeline o job completo.
  - `custom_compare_(output:Any, ground_truth:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `stripped_string_compare(s1:Any, s2:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `call_method(method:Any, inputs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `reliability_guard(maximum_memory_bytes:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `TACODataset`
    Metodo: `load(path:str, num_repeats:int=1, difficulty:Any='ALL') -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `TACOEvaluator`
    Metodo: `post_process(self:Any, text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `check_correctness(self:Any, sample:Any, generation:Any, timeout:Any, debug:Any=True) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `evaluate_generations(self:Any, generations:Any, samples:Any, idx:Any=None, debug:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `estimate_pass_at_k(self:Any, num_samples:Any, num_correct:Any, k:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `compute_metrics(self:Any, results:Any, k_list:Any=[1, 10, 100]) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any, test_set:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `CODE_TYPE`
  - `TimeoutException`
  - `Capturing`
    Metodo: `__enter__(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__exit__(self:Any, *args:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\teval\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `teval_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `TEvalDataset`
    Metodo: `__init__(self:Any, reader_cfg:Optional[Dict]={}, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `load(self:Any, path:str, name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\teval\evaluators\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\teval\evaluators\instruct_evaluator.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `InstructEvaluator`
    Metodo: `__init__(self:Any, dataset_path:str, **kwargs:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_load_dataset(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_process_response(self:Any, datum:dict) -> ResponseDataSample`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_evaluate(self:Any, data_sample:dict) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `compute_args_em_metric(self:Any, gt_action:Any, pred_action:Any, gt_args:Any, pred_args:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `string_format_parse(self:Any, data_sample:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `json_format_parse(self:Any, data_sample:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `evaluate(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_post_process(self:Any, results_list:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\teval\evaluators\planning_evaluator.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `PlanningEvaluator`
    Metodo: `__init__(self:Any, dataset_path:str, name_weight:Any=0.75, args_weight:Any=0.25, match_threshold:Any=0.7, match_strategy:str='bertscore', bert_score_model:str='all-mpnet-base-v2', default_prompt_type:str='json', **kwargs:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_load_dataset(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `format_load(self:Any, data:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_process_response(self:Any, datum:Any) -> ResponseDataSample`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_evaluate(self:Any, data_sample:Any) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `evaluate(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `permutation_match(self:Any, pred_plan:Any, gt_plan:Any) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `bertscore_match(self:Any, pred_plan:Any, gt_plan:Any) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_post_process(self:Any, results_list:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\teval\evaluators\reason_retrieve_understand_evaluator.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `input_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `ReasonRetrieveUnderstandEvaluator`
    Metodo: `__init__(self:Any, dataset_path:str, bert_score_model:str='all-mpnet-base-v2', default_prompt_type:str='json', eval_type:str='reason', **kwargs:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_load_dataset(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `format_load(self:Any, data:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_process_response(self:Any, datum:Any) -> ResponseDataSample`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_evaluate(self:Any, data_sample:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `evaluate(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `find_a_dot_b_structure(self:Any, text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `find_FinishAction(self:Any, text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_post_process(self:Any, results_list:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `ReasonRetrieveUnderstandEvaluatorNoBatch`
    Metodo: `__init__(self:Any, dataset_path:str, bert_score_model:str='all-mpnet-base-v2', default_prompt_type:str='json', eval_type:str='reason') -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_load_dataset(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `format_load(self:Any, data:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_process_response(self:Any, datum:Any) -> ResponseDataSample`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_evaluate(self:Any, data_sample:Any) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `evaluate(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_post_process(self:Any, results_list:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\teval\evaluators\review_evaluator.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `ReviewEvaluator`
    Metodo: `__init__(self:Any, dataset_path:str, **kwargs:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_load_dataset(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_process_response(self:Any, datum:dict) -> ResponseDataSample`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_evaluate(self:Any, data_sample:Any) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `json_format_parse(self:Any, pred_data:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `evaluate(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_post_process(self:Any, results_list:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\teval\schema.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `ResponseDataSample`

- File: `vendor\llada\opencompass\opencompass\datasets\teval\utils\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\teval\utils\convert_results.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `parse_args() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `convert_results(result_path:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\teval\utils\format_load.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `format_load(raw_data:str, start_character:str='', end_character:str='') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\teval\utils\meta_template.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\teval\utils\template.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `format_string(template:str, input_data:dict) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `parse_string(template:str, input_string:str, allow_newline:bool=False) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\TheoremQA\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\datasets\TheoremQA\legacy.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `TheoremQA_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `TheoremQA_postprocess_v2(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `TheoremQADataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\TheoremQA\main.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `TheoremQA_postprocess_v3(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `TheoremQA_postprocess_v4(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `TheoremQADatasetV3`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `TheoremQAEvaluatorV3`
    Metodo: `score(self:Any, predictions:Any, references:Any, test_set:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\TheoremQA\number_utils.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `floatify(num:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `within_eps(pred:float, gt:float) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `clean_units(pred_str:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `number_it(num:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `compare_two_numbers(p:Any, gt:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `compare_two_list(pred:Any, gt:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\TheoremQA\utils.py`
  Logica d'uso: Runtime helpers, device/dtype control, tiny fallback model/tokenizer.
  Funzioni:
  - `time_limit(seconds:float) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `extract_theoremqa_answer(pred:str, answer_flag:bool=True) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `answer_clean(direct_answer_trigger_for_fewshot:tuple, pred:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `compare_answer_with_groundtruth(answer:str, groundtruth_str:str, groundtruth_num:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\tnews.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `TNewsDataset`
    Metodo: `load(**kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `TNewsDatasetV2`
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\triviaqa.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `TriviaQADataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `TriviaQADatasetV2`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `TriviaQADatasetV3`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `TriviaQAEvaluator`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\triviaqarc.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `TriviaQArcDataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\truthfulqa.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `TruthfulQADataset`
    Metodo: `load(**kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `TruthfulQAEvaluator`
    Metodo: `__init__(self:Any, truth_model:str='allenai/truthfulqa-truth-judge-llama2-7B', info_model:str='allenai/truthfulqa-info-judge-llama2-7B', metrics:Any='truth', key:Any='ENV') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `basic_score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `prompt(self:Any, pred:Any, refer:Any, metric:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `postprocess(self:Any, generated_token:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `api_score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\tydiqa.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `TydiQADataset`
    Metodo: `load(path:Any, lang:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `TydiQAEvaluator`
    Metodo: `f1_score(self:Any, prediction:Any, ground_truth:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `exact_match_score(self:Any, prediction:Any, ground_truth:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `metric_max_over_ground_truths(self:Any, metric_fn:Any, prediction:Any, ground_truths:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\wic.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `WiCDataset`
    Metodo: `load(**kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `WiCDatasetV2`
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\wikibench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_number(options:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  Classi:
  - `WikiBenchDataset`
    Metodo: `load(path:str, filename:str, name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\winograd.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `WinogradDataset`
    Metodo: `load(**kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\winogrande.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `WinograndeDataset`
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `WinograndeDatasetV2`
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `WinograndeDatasetV3`
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\wnli.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `wnliDataset`
    Metodo: `load(**kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\wsc.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `WSCDataset`
    Metodo: `load(**kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `WSCDatasetV2`
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `WSCDatasetV3`
    Metodo: `load(path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\xcopa.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `XCOPADataset`
    Metodo: `load(**kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\xiezhi.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `XiezhiDataset`
    Metodo: `load(path:str, name:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `XiezhiRetriever`
    Metodo: `__init__(self:Any, dataset:Any, ice_separator:Optional[str]='\n', ice_eos_token:Optional[str]='\n', ice_num:Optional[int]=1) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `retrieve(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\datasets\xlsum.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `XLSUMDataset`
    Metodo: `load(**kwargs:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\datasets\xsum.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `Xsum_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `XsumDataset`
    Metodo: `load(path:str) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.

- File: `vendor\llada\opencompass\opencompass\evaluator\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\evaluator\cascade_evaluator.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `CascadeEvaluator`
    Metodo: `__init__(self:Any, llm_evaluator:Dict, rule_evaluator:Optional[Dict]=None, sample_score_fn:Optional[Callable]=None, parallel:bool=True) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `sample_score(self:Any, prediction:str, reference:str, test_set:Any=None) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_get_llm_correctness(self:Any, llm_detail:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:List[str], references:List[str], test_set:Optional[Dataset]=None) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\evaluator\generic_llm_evaluator.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `GenericLLMEvaluator`
    Metodo: `__init__(self:Any, prompt_template:ConfigDict, judge_cfg:ConfigDict, dataset_cfg:Optional[ConfigDict]=None, pred_postprocessor:Optional[ConfigDict]=None, dict_postprocessor:Optional[ConfigDict]=None, keep_predictions:bool=False) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `build_inferencer(self:Any) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
    Metodo: `score(self:Any, predictions:Any, references:Optional[List]=None, test_set:Optional[Dataset]=None) -> Dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `pred_postprocess(self:Any, predictions:List) -> Dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `output_postprocess(self:Any, output:Dict, dataset:Any=None) -> Dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `default_judge_cfg(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\evaluator\math_evaluator.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `MATHVerifyEvaluator`
    Metodo: `score(self:Any, predictions:Any, references:Any, test_set:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\lagent\actions\ipython_interpreter.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `extract_code(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `escape_ansi(line:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `publish_image_to_local(image_base64:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_multiline_input(hint:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  Classi:
  - `TimeoutError`
  - `IPythonInterpreter`
    Metodo: `__init__(self:Any, description:str=DEFAULT_DESCRIPTION, name:Optional[str]=None, enable:bool=True, disable_description:Optional[str]=None, timeout:int=20, trim_output:Optional[int]=1024, user_data_dir:str='ENV', force_user_data:bool=True) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `start_kernel() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `initialize(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `reset(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_call(self:Any, command:str, timeout:Optional[int]=None) -> Tuple[str, bool]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__call__(self:Any, command:str, timeout:Optional[int]=None) -> ActionReturn`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\lagent\actions\python_interpreter.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `GenericRuntime`
    Metodo: `__init__(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `exec_code(self:Any, code_piece:str) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `eval_code(self:Any, expr:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `PythonInterpreter`
    Metodo: `__init__(self:Any, description:str=DEFAULT_DESCRIPTION, answer_symbol:Optional[str]=None, answer_expr:Optional[str]='solution()', answer_from_stdout:bool=False, name:Optional[str]=None, enable:bool=True, disable_description:Optional[str]=None, timeout:int=20) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `extract_code(command:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__call__(self:Any, command:str) -> ActionReturn`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_call(self:Any, command:str) -> ActionReturn`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\lagent\agents\react.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `ReActProtocol`
    Metodo: `__init__(self:Any, **kwargs:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `format(self:Any, chat_history:List[Dict], inner_step:List[Dict], action_executor:ActionExecutor, force_stop:bool=False) -> list`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `ReAct`
    Metodo: `__init__(self:Any, use_system_role:bool=True, first_system_role:bool=True, merge_adjacent_role:bool=False, **kwargs:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `chat(self:Any, message:str) -> AgentReturn`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `CIReAct`
    Metodo: `reset(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `chat(self:Any, message:str) -> AgentReturn`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `CIReActMergeRole`
    Metodo: `chat(self:Any, message:str) -> AgentReturn`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `merge_role(self:Any, inputs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\metrics\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\metrics\dump_results.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `DumpResults`
    Metodo: `__init__(self:Any, save_path:str, collect_device:str='cpu', prefix:Optional[str]=None) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `process(self:Any, data_batch:Any, data_samples:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `compute_metrics(self:Any, results:list) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\metrics\mme_score.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `MMEMetric`
    Metodo: `__init__(self:Any, collect_device:str='cpu', prefix:Optional[str]=None) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `process(self:Any, data_batch:Any, data_samples:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `compute_metrics(self:Any, results:list) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\metrics\seedbench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `SEEDBenchAcc`
    Metodo: `process(self:Any, data_batch:Any, data_samples:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `compute_metrics(self:Any, results:list) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\models\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\models\accessory.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LLaMA2AccessoryModel`
    Metodo: `__init__(self:Any, tokenizer_only:bool=False, meta_template:Optional[Dict]=None, additional_stop_symbols:Iterable[str]=(), **from_pretrained_kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_load_model(self:Any, from_pretrained_kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_load_tokenizer(self:Any, from_pretrained_kwargs:Any) -> Any`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
    Metodo: `generate(self:Any, inputs:List[str], max_out_len:int) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `get_ppl(self:Any, inputs:List[str], mask_length:Optional[List[int]]=None) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_token_len(self:Any, prompt:str) -> int`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\models\ai360_api.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `AI360GPT`
    Metodo: `__init__(self:Any, path:str, key:str, url:str='https://api.360.cn/v1/chat/completions', query_per_second:int=2, max_seq_len:int=2048, meta_template:Optional[Dict]=None, retry:int=2, generation_kwargs:Dict={'temperature': 0.9, 'max_tokens': 2048, 'top_p': 0.5, 'tok_k': 0, 'repetition_penalty': 1.05}) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, inputs:List[PromptType], max_out_len:int=512) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `_generate(self:Any, input:PromptType, max_out_len:int=512) -> str`
    Descrizione: Genera output testo o sequenze.

- File: `vendor\llada\opencompass\opencompass\models\alaya.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `AlayaLM`
    Metodo: `__init__(self:Any, path:str, max_seq_len:int=2048, tokenizer_only:bool=False, meta_template:Optional[Dict]=None, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `do_inference(self:Any, instruction:Any, history:Any=[]) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, inputs:Any, max_out_len:int=1000) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `get_token_len(self:Any, prompt:str) -> int`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_ppl(self:Any, inputs:List[str], mask_length:Optional[List[int]]=None) -> List[float]`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\models\baichuan_api.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `BaiChuan`
    Metodo: `__init__(self:Any, path:str, key:str, url:str, query_per_second:int=2, max_seq_len:int=2048, meta_template:Optional[Dict]=None, retry:int=2, system_prompt:str='') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, inputs:List[PromptType], max_out_len:int=512) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `_generate(self:Any, input:PromptType, max_out_len:int=512) -> str`
    Descrizione: Genera output testo o sequenze.

- File: `vendor\llada\opencompass\opencompass\models\baidu_api.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `ERNIEBot`
    Metodo: `__init__(self:Any, path:str, key:str, secretkey:str, url:str, query_per_second:int=2, max_seq_len:int=2048, meta_template:Optional[Dict]=None, retry:int=2, generation_kwargs:Dict={'temperature': 0.8}) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_generate_access_token(self:Any) -> Any`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
    Metodo: `generate(self:Any, inputs:List[PromptType], max_out_len:int=512) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `_generate(self:Any, input:PromptType, max_out_len:int=512) -> str`
    Descrizione: Genera output testo o sequenze.

- File: `vendor\llada\opencompass\opencompass\models\bailing_api_oc.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `HTTPAdapterWithSocketOptions`
    Metodo: `__init__(self:Any, *args:Any, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `init_poolmanager(self:Any, *args:Any, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `BailingAPI`
    Metodo: `__init__(self:Any, path:str, token:str, url:str, meta_template:Optional[Dict]=None, query_per_second:int=1, retry:int=3, generation_kwargs:Dict={}, max_seq_len:Any=4096) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, inputs:Union[List[str], PromptList], max_out_len:int=11264) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `_generate(self:Any, sess:Any, input:Union[str, PromptList], max_out_len:int) -> str`
    Descrizione: Genera output testo o sequenze.
    Metodo: `_infer_result(self:Any, request:Any, sess:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\models\base.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `BaseModel`
    Metodo: `__init__(self:Any, path:str, max_seq_len:int=2048, tokenizer_only:bool=False, meta_template:Optional[Dict]=None, generation_kwargs:Optional[Dict]=dict(), sync_rank:bool=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, inputs:List[str], max_out_len:int) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `get_ppl(self:Any, inputs:List[str], mask_length:Optional[List[int]]=None) -> List[float]`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_ppl_tokenwise(self:Any, inputs:List[str], mask_length:Optional[List[int]]=None) -> List[float]`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `encode(self:Any, prompt:str) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `decode(self:Any, tokens:torch.Tensor) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_token_len(self:Any, prompt:str) -> int`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `parse_template(self:Any, prompt_template:PromptType, mode:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_ppl_from_template(self:Any, templates:List[PromptType], mask_length:Any=None) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_ppl_tokenwise_from_template(self:Any, templates:List[PromptType], label:List[List[int]], mask_length:Any=None) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `generate_from_template(self:Any, templates:List[PromptType], max_out_len:int, **kwargs:Any) -> Any`
    Descrizione: Genera output testo o sequenze.
    Metodo: `get_token_len_from_template(self:Any, templates:Union[PromptType, List[PromptType]], mode:str='ppl') -> Union[List[int], int]`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `sync_inputs(self:Any, inputs:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `to(self:Any, device:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LMTemplateParser`
    Metodo: `__init__(self:Any, meta_template:Optional[Dict]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `parse_template(self:Any, prompt_template:PromptType, mode:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_split_rounds(self:Any, prompt_template:List[Union[str, Dict]], single_round_template:List[Union[str, Dict]]) -> List[int]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_update_role_dict(self:Any, prompt:Union[List, str, Dict]) -> Dict[str, Dict]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_prompt2str(self:Any, prompt:Union[List, str, Dict], role_dict:Dict[str, Dict], for_gen:bool=False) -> Tuple[str, bool]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_role2str(self:Any, role_prompt:Dict, role_dict:Dict[str, Dict], for_gen:bool=False) -> Tuple[str, bool]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_encode_speical_tokens(self:Any, prompt:List[Union[str, int]]) -> str`
    Descrizione: Gestisce tokenizzazione o manipolazione token.

- File: `vendor\llada\opencompass\opencompass\models\base_api.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `BaseAPIModel`
    Metodo: `__init__(self:Any, path:str, query_per_second:int=1, rpm_verbose:bool=False, retry:int=2, max_seq_len:int=2048, meta_template:Optional[Dict]=None, generation_kwargs:Dict=dict(), verbose:bool=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, inputs:List[PromptType], max_out_len:int) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `flush(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `acquire(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `release(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_ppl(self:Any, inputs:List[PromptType], mask_length:Optional[List[int]]=None) -> List[float]`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_token_len(self:Any, prompt:str) -> int`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `wait(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `to(self:Any, device:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `APITemplateParser`
    Metodo: `__init__(self:Any, meta_template:Optional[Dict]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `parse_template(self:Any, prompt_template:PromptType, mode:str) -> PromptType`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_update_role_dict(self:Any, prompts:Union[List, str]) -> Dict[str, Dict]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_split_rounds(self:Any, prompt_template:List[Union[str, Dict]], single_round_template:List[Union[str, Dict]]) -> List[int]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_prompt2api(self:Any, prompts:Union[List, str], role_dict:Dict[str, Dict], for_gen:bool=False) -> Tuple[List, bool]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_role2api_role(self:Any, role_prompt:Dict, role_dict:Dict[str, Dict], for_gen:bool=False) -> Tuple[Dict, bool]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `TokenBucket`
    Metodo: `__init__(self:Any, rate:Any, verbose:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_add_tokens(self:Any) -> Any`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
    Metodo: `get_token(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\models\bluelm_api.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `BlueLMAPI`
    Metodo: `__init__(self:Any, path:str, key:str, url:str, query_per_second:int=2, max_seq_len:int=2048, meta_template:Optional[Dict]=None, retry:int=5, system_prompt:str='', generation_kwargs:Optional[Dict]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, inputs:List[PromptType], max_out_len:int=512) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `get_streaming_response(self:Any, response:requests.Response) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `split_think(self:Any, text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_generate(self:Any, input:PromptType, max_out_len:int=512) -> str`
    Descrizione: Genera output testo o sequenze.

- File: `vendor\llada\opencompass\opencompass\models\bytedance_api.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `ByteDance`
    Metodo: `__init__(self:Any, path:str, accesskey:str, secretkey:str, url:str, query_per_second:int=2, max_seq_len:int=2048, meta_template:Optional[Dict]=None, retry:int=2, generation_kwargs:Dict={'temperature': 0.7, 'top_p': 0.9, 'top_k': 0}) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, inputs:List[PromptType], max_out_len:int=512) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `_generate(self:Any, input:PromptType, max_out_len:int=512) -> str`
    Descrizione: Genera output testo o sequenze.

- File: `vendor\llada\opencompass\opencompass\models\claude_allesapin.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `ClaudeAllesAPIN`
    Metodo: `__init__(self:Any, path:str, url:str, key:str, query_per_second:int=1, max_seq_len:int=2048, meta_template:Optional[Dict]=None, retry:int=2) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, inputs:List[PromptType], max_out_len:int=512, **kwargs:Any) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `_generate(self:Any, input:PromptType, max_out_len:int) -> str`
    Descrizione: Genera output testo o sequenze.

- File: `vendor\llada\opencompass\opencompass\models\claude_api\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\models\claude_api\claude_api.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `Claude`
    Metodo: `__init__(self:Any, key:str, path:str='claude-2', query_per_second:int=2, max_seq_len:int=2048, meta_template:Optional[Dict]=None, retry:int=2) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, inputs:List[PromptType], max_out_len:int=512) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `_generate(self:Any, input:PromptType, max_out_len:int=512) -> str`
    Descrizione: Genera output testo o sequenze.

- File: `vendor\llada\opencompass\opencompass\models\claude_api\postprocessors.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `gsm8k_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `humaneval_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `lcsts_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `mbpp_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `strategyqa_pred_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `flores_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `flores_postprocess_chinese(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `record_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `humaneval_claude2_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `xsum_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `yes_no_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\models\claude_sdk_api.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `ClaudeSDK`
    Metodo: `__init__(self:Any, key:str, path:str='claude-2', query_per_second:int=2, max_seq_len:int=2048, meta_template:Optional[Dict]=None, temperature:Optional[float]=0.0, thinking:Optional[Dict]=None, retry:int=2) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, inputs:List[PromptType], max_out_len:int=512) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `_generate(self:Any, input:PromptType, max_out_len:int=512) -> str`
    Descrizione: Genera output testo o sequenze.

- File: `vendor\llada\opencompass\opencompass\models\deepseek_api.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `DeepseekAPI`
    Metodo: `__init__(self:Any, path:str, key:str, url:str, query_per_second:int=2, max_seq_len:int=2048, meta_template:Optional[Dict]=None, retry:int=2, system_prompt:str='') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, inputs:List[PromptType], max_out_len:int=512) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `_generate(self:Any, input:PromptType, max_out_len:int=512) -> str`
    Descrizione: Genera output testo o sequenze.

- File: `vendor\llada\opencompass\opencompass\models\dllm.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_get_meta_template(meta_template:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `add_gumbel_noise(logits:Any, temperature:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_num_transfer_tokens(mask_index:Any, steps:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `_convert_chat_messages(inputs:Any, merge_role:Any=True, skip_empty_prompt:Any=True) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_convert_base_messages(inputs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `LLaDAModel`
    Metodo: `__init__(self:Any, path:str, hf_cache_dir:Optional[str]=None, max_seq_len:int=2048, tokenizer_path:Optional[str]=None, tokenizer_kwargs:dict=dict(), peft_path:Optional[str]=None, tokenizer_only:bool=False, model_kwargs:dict=dict(device_map='auto'), generation_kwargs:dict=dict(), meta_template:Optional[Dict]=None, extract_pred_after_decode:bool=False, batch_padding:bool=False, pad_token_id:Optional[int]=None, mode:str='none', use_fastchat_template:bool=False, end_str:Optional[str]=None, stop_words:Optional[List[str]]=[], cfg:Any=0, temperature:Any=0.0, remasking:Any='low_confidence', mask_id:Any=126336, padding_id:Any=126081, mc_num:Any=1, gen_steps:Any=512, gen_length:Any=512, gen_blocksize:Any=512, batch_size_:Any=1, diff_confidence_eos_eot_inf:Any=False, diff_logits_eos_inf:Any=False) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_load_tokenizer(self:Any, path:str, tokenizer_path:Optional[str], tokenizer_kwargs:dict) -> Any`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
    Metodo: `_set_model_kwargs_torch_dtype(self:Any, model_kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_load_model(self:Any, path:str, model_kwargs:dict, peft_path:Optional[str]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_get_loglikelihood(self:Any, inputs:str, conts:str) -> float`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_mink_percent(self:Any, inputs:List[str], k:int=20) -> List[float]`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `_get_mink_percent(self:Any, inputs:List[str], k:int=20) -> List[float]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_token_len(self:Any, prompt:str) -> int`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `generate(self:Any, inputs:List[str], max_out_len:int) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `get_ppl(self:Any, inputs:List[str], mask_length:Optional[List[int]]=None) -> List[float]`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_loglikelihood(self:Any, inputs:List[str], conts:List[str]) -> List[float]`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_logits(self:Any, inputs:Any, prompt_index:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `LLaDABaseModel`
    Metodo: `__init__(self:Any, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, inputs:List[str], max_out_len:int) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `get_token_len(self:Any, prompt:str, add_special_tokens:bool=True) -> int`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\models\doubao.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `Doubao`
    Metodo: `__init__(self:Any, path:str, endpoint_id:str, access_key:str, secret_key:str, query_per_second:int=2, max_seq_len:int=2048, meta_template:Optional[Dict]=None, retry:int=2) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, inputs:List[PromptType], max_out_len:int=512) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `_generate(self:Any, input:PromptType, max_out_len:int=512) -> str`
    Descrizione: Genera output testo o sequenze.

- File: `vendor\llada\opencompass\opencompass\models\doubao_api.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `Doubao`
    Metodo: `__init__(self:Any, path:str, accesskey:str, secretkey:str, query_per_second:int=2, max_seq_len:int=2048, meta_template:Optional[Dict]=None, retry:int=2, generation_kwargs:Dict={'temperature': 0.7, 'top_p': 0.9}) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, inputs:List[PromptType], max_out_len:int=512) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `_generate(self:Any, input:PromptType, max_out_len:int=512) -> str`
    Descrizione: Genera output testo o sequenze.

- File: `vendor\llada\opencompass\opencompass\models\gemini_api.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `Gemini`
    Metodo: `__init__(self:Any, key:str, path:str, query_per_second:int=2, max_seq_len:int=2048, meta_template:Optional[Dict]=None, retry:int=2, temperature:float=1.0, top_p:float=0.8, top_k:float=10.0) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, inputs:List[PromptType], max_out_len:int=512) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `_generate(self:Any, input:PromptType, max_out_len:int=512) -> str`
    Descrizione: Genera output testo o sequenze.

- File: `vendor\llada\opencompass\opencompass\models\glm.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `GLM130B`
    Metodo: `__init__(self:Any, pkg_root:str, ckpt_path:str, tokenizer_only:bool=False, meta_template:Optional[Dict]=None, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_load_model(self:Any, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_token_len(self:Any, prompt:str) -> int`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `choice(self:Any, inputs:Any, choices:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, inputs:List[str], max_out_len:int) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `get_logits(self:Any, inputs:List[str]) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_ppl(self:Any, inputs:List[str], mask_length:Optional[List[int]]=None) -> List[float]`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\models\huggingface.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `MultiTokenEOSCriteria`
    Metodo: `__init__(self:Any, sequence:str, tokenizer:transformers.PreTrainedTokenizer, batch_size:int) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__call__(self:Any, input_ids:Any, scores:Any, **kwargs:Any) -> bool`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `HuggingFace`
    Metodo: `__init__(self:Any, path:str, hf_cache_dir:Optional[str]=None, max_seq_len:int=2048, tokenizer_path:Optional[str]=None, tokenizer_kwargs:dict=dict(), peft_path:Optional[str]=None, tokenizer_only:bool=False, model_kwargs:dict=dict(device_map='auto'), generation_kwargs:dict=dict(), meta_template:Optional[Dict]=None, extract_pred_after_decode:bool=False, batch_padding:bool=False, pad_token_id:Optional[int]=None, mode:str='none', use_fastchat_template:bool=False, end_str:Optional[str]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_load_tokenizer(self:Any, path:str, tokenizer_path:Optional[str], tokenizer_kwargs:dict) -> Any`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
    Metodo: `_set_model_kwargs_torch_dtype(self:Any, model_kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_load_model(self:Any, path:str, model_kwargs:dict, peft_path:Optional[str]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, inputs:List[str], max_out_len:int, min_out_len:Optional[int]=None, stopping_criteria:List[str]=[], **kwargs:Any) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `_batch_generate(self:Any, inputs:List[str], max_out_len:int, min_out_len:Optional[int]=None, stopping_criteria:List[str]=[], **kwargs:Any) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `_single_generate(self:Any, inputs:List[str], max_out_len:int, min_out_len:Optional[int]=None, stopping_criteria:List[str]=[], **kwargs:Any) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `get_logits(self:Any, inputs:List[str]) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_ppl(self:Any, inputs:List[str], mask_length:Optional[List[int]]=None) -> List[float]`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `_get_ppl(self:Any, inputs:List[str], mask_length:Optional[List[int]]=None) -> List[float]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_loglikelihood(self:Any, inputs:List[str], conts:List[str], mask_length:Optional[List[int]]=None) -> List[float]`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `_get_loglikelihood(self:Any, inputs:str, conts:str) -> float`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_mink_percent(self:Any, inputs:List[str], k:int=20) -> List[float]`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `_get_mink_percent(self:Any, inputs:List[str], k:int=20) -> List[float]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_token_len(self:Any, prompt:str) -> int`
    Descrizione: Recupera valore/stato calcolato.
  - `HuggingFaceCausalLM`
    Metodo: `_load_model(self:Any, path:str, model_kwargs:dict, peft_path:Optional[str]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `HuggingFaceChatGLM3`
    Metodo: `__init__(self:Any, path:str, hf_cache_dir:Optional[str]=None, max_seq_len:int=2048, tokenizer_path:Optional[str]=None, tokenizer_kwargs:dict=dict(), peft_path:Optional[str]=None, tokenizer_only:bool=False, model_kwargs:dict=dict(device_map='auto'), generation_kwargs:dict=dict(), meta_template:Optional[Dict]=None, extract_pred_after_decode:bool=False, batch_padding:bool=False, pad_token_id:Optional[int]=None, mode:str='none', num_extra_tokens:int=50) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, inputs:List[PromptType], max_out_len:int=512, skip_overlength:Any=False, **kwargs:Any) -> str`
    Descrizione: Genera output testo o sequenze.
    Metodo: `get_token_len(self:Any, prompt:str) -> int`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\models\huggingface_above_v4_33.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_get_stopping_criteria(stop_words:Any, tokenizer:Any, batch_size:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_get_possible_max_seq_len(max_seq_len:Any, path:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_convert_chat_messages(inputs:Any, merge_role:Any=True, skip_empty_prompt:Any=True) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_format_with_fast_chat_template(inputs:List[str], name:str='vicuna') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_get_meta_template(meta_template:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_set_model_kwargs_torch_dtype(model_kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_convert_base_messages(inputs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `HuggingFacewithChatTemplate`
    Metodo: `__init__(self:Any, path:str, model_kwargs:dict=dict(), tokenizer_path:Optional[str]=None, tokenizer_kwargs:dict=dict(), peft_path:Optional[str]=None, peft_kwargs:dict=dict(), tokenizer_only:bool=False, generation_kwargs:dict=dict(), max_seq_len:Optional[int]=None, meta_template:Optional[Dict]=None, pad_token_id:Optional[int]=None, fastchat_template:Optional[str]=None, stop_words:Optional[str]=[], mode:str='none', **other_kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_load_tokenizer(self:Any, path:Optional[str], kwargs:dict, pad_token_id:Optional[int]=None) -> Any`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
    Metodo: `_load_model(self:Any, path:str, kwargs:dict, peft_path:Optional[str]=None, peft_kwargs:dict=dict()) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_ppl_tokenwise(self:Any, inputs:List[str], label:List[List[int]], mask_length:Optional[List[int]]=None) -> List[float]`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `_get_potential_stop_words(self:Any, path:Optional[str]) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, inputs:List[str], max_out_len:int, min_out_len:Optional[int]=None, stopping_criteria:List[str]=[], **kwargs:Any) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `get_token_len(self:Any, prompt:str) -> int`
    Descrizione: Recupera valore/stato calcolato.
  - `HuggingFaceBaseModel`
    Metodo: `__init__(self:Any, path:str, model_kwargs:dict=dict(), tokenizer_path:Optional[str]=None, tokenizer_kwargs:dict=dict(), peft_path:Optional[str]=None, peft_kwargs:dict=dict(), tokenizer_only:bool=False, generation_kwargs:dict=dict(), max_seq_len:Optional[int]=None, pad_token_id:Optional[int]=None, stop_words:Optional[str]=[], drop_middle:bool=False, **other_kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, inputs:List[str], max_out_len:int, min_out_len:Optional[int]=None, stopping_criteria:List[str]=[], **kwargs:Any) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `get_ppl(self:Any, inputs:List[str], mask_length:Optional[List[int]]=None) -> List[float]`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_loglikelihood(self:Any, inputs:List[str], conts:List[str]) -> List[float]`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_token_len(self:Any, prompt:str, add_special_tokens:bool=True) -> int`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\models\hunyuan_api.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `Hunyuan`
    Metodo: `__init__(self:Any, path:str, secret_id:str, secret_key:str, endpoint:str, query_per_second:int=2, max_seq_len:int=2048, meta_template:Optional[Dict]=None, retry:int=2) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, inputs:List[PromptType], max_out_len:int=512) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `_generate(self:Any, input:PromptType, max_out_len:int=512) -> str`
    Descrizione: Genera output testo o sequenze.

- File: `vendor\llada\opencompass\opencompass\models\intern_model.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `InternLM`
    Metodo: `__init__(self:Any, path:str, max_seq_len:int=2048, tokenizer_only:bool=False, tokenizer_path:Optional[str]=None, model_config:Optional[str]=None, tokenizer_type:Optional[str]='v7', meta_template:Optional[Dict]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_load_model(self:Any, path:str, max_seq_len:int, tokenizer_path:Optional[str]=None, tokenizer_type:Optional[str]=None, model_config:Optional[str]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_load_tokenizer(self:Any, tokenizer_path:str, tokenizer_type:str, max_seq_len:int) -> Any`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
    Metodo: `get_token_len(self:Any, prompt:str) -> int`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `generate(self:Any, inputs:List[str], max_out_len:int) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `get_ppl(self:Any, input_texts:List[str], mask_length:Optional[List[int]]=None) -> List[float]`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\models\interntrain.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `InternTrainManager`
    Metodo: `__init__(self:Any, module_path:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `build(module_path:Any) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
  - `CurrentInternTrainManager`
    Metodo: `load_config(self:Any, path:Any, model_config:Any=None) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
    Metodo: `initialize_model(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LegacyInternTrainManager`
    Metodo: `load_config(self:Any, path:Any, model_config:Any=None) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
    Metodo: `initialize_model(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `InternTrain`
    Metodo: `__init__(self:Any, path:str, module_path:str, max_seq_len:int=2048, tokenizer_only:bool=False, tokenizer_path:Optional[str]=None, tokenizer_type:str='INTERNLM', model_config:Optional[Union[str, Dict]]=None, parallel_config:Optional[str]=None, model_type:str='INTERNLM2', ckpt_type:Optional[str]=None, meta_template:Optional[Dict]=None, model_dtype:Optional[str]=None, generation_kwargs:Any={}, sync_rank:bool=False, mode:Any='none', end_str:Optional[str]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_load_model(self:Any, path:str, model_config:Optional[str]=None, parallel_config:Optional[str]=None, model_type:str='INTERNLM2', model_dtype:Optional[str]=None, ckpt_type:Optional[str]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_load_tokenizer(self:Any, tokenizer_path:str, tokenizer_type:str) -> Any`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
    Metodo: `_convert_dtype(self:Any, default_dtype:Any, model_dtype:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_token_len(self:Any, prompt:str, use_bos:Any=None, use_eos:Any=None) -> int`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `generate(self:Any, inputs:List[str], max_out_len:int, min_out_len:Optional[int]=None, stopping_criteria:List[str]=[]) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `get_ppl(self:Any, input_texts:List[str], mask_length:Optional[List[int]]=None) -> List[float]`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_loglikelihood(self:Any, input_texts:List[str], conts:List[str]) -> List[float]`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_mink_percent(self:Any, input_texts:List[str], k:int=20) -> List[float]`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_logits(self:Any, input_texts:Union[str, List[str]]) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `batch_encode(self:Any, input_texts:Union[str, List[str]], max_seq_len:int, left_padding:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `batch_decode(self:Any, outputs:Any, eos_token_ids:List[int], stopping_criteria:List[str]=[]) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\models\krgpt_api.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `KrGPT`
    Metodo: `__init__(self:Any, path:str='KrGPT', url:str='http://101.69.162.5:9300/v1/chat/completions', max_seq_len:int=2048, meta_template:Optional[Dict]=None, retry:int=2, generation_kwargs:Optional[Dict]=dict()) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, inputs:List[str], max_out_len:int, **kwargs:Any) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `_generate(self:Any, input:PromptType, max_out_len:int, temperature:float=0.0) -> str`
    Descrizione: Genera output testo o sequenze.

- File: `vendor\llada\opencompass\opencompass\models\lagent.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LagentAgent`
    Metodo: `__init__(self:Any, agent_type:Any, llm:Any, actions:Any=None, protocol:Any=None, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `reset(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `set_history(self:Any, history:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `gt_response(self:Any, prompt:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `template_parser(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `template_parser(self:Any, value:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `chat(self:Any, user_input:str, history:List[dict]=None) -> Tuple[str, List[dict], List[dict]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `CodeAgent`
    Metodo: `__init__(self:Any, llm:Any, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\models\langchain.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LangchainAgent`
    Metodo: `__init__(self:Any, agent_type:Any, llm:Any, tools:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `chat(self:Any, user_input:Any, ice:Any=None) -> Tuple[str, List[dict]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\models\lightllm_api.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LightllmAPI`
    Metodo: `__init__(self:Any, path:str='LightllmAPI', url:str='http://localhost:8080/generate', meta_template:Optional[Dict]=None, max_workers_per_task:int=2, rate_per_worker:int=2, retry:int=2, generation_kwargs:Optional[Dict]=dict()) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, inputs:List[str], max_out_len:int, **kwargs:Any) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `_generate(self:Any, input:str, max_out_len:int) -> str`
    Descrizione: Genera output testo o sequenze.
    Metodo: `get_ppl(self:Any, inputs:List[str], max_out_len:int, **kwargs:Any) -> List[float]`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `_get_ppl(self:Any, input:str, max_out_len:int) -> float`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `wait(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_token_len(self:Any, prompt:str) -> int`
    Descrizione: Recupera valore/stato calcolato.
  - `LightllmChatAPI`
    Metodo: `__init__(self:Any, path:str, url:str, query_per_second:int=2, max_seq_len:int=2048, meta_template:Optional[Dict]=None, retry:int=2) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, inputs:List[PromptType], max_out_len:int=512) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `_generate(self:Any, input:PromptType, max_out_len:int=512) -> str`
    Descrizione: Genera output testo o sequenze.

- File: `vendor\llada\opencompass\opencompass\models\llama2.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `Llama2`
    Metodo: `__init__(self:Any, path:str, max_seq_len:int=2048, max_batch_size:int=16, tokenizer_only:bool=False, tokenizer_path:Optional[str]=None, meta_template:Optional[Dict]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_load_model(self:Any, path:str, max_seq_len:int, max_batch_size:int, tokenizer_path:Optional[str]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_load_tokenizer(self:Any, tokenizer_path:str) -> Any`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
    Metodo: `generate(self:Any, inputs:List[str], max_out_len:int) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `get_ppl(self:Any, inputs:List[str], mask_length:Optional[List[int]]=None) -> List[float]`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_loglikelihood(self:Any, inputs:List[str], conts:List[str], mask_length:Optional[List[int]]=None) -> List[float]`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_token_len(self:Any, prompt:str) -> int`
    Descrizione: Recupera valore/stato calcolato.
  - `Llama2Chat`
    Metodo: `__init__(self:Any, path:str, max_seq_len:int=2048, max_batch_size:int=16, tokenizer_only:bool=False, tokenizer_path:Optional[str]=None, meta_template:Optional[Dict]=None, force_bf16:bool=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_load_model(self:Any, path:str, max_seq_len:int, max_batch_size:int, tokenizer_path:Optional[str]=None, force_bf16:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_load_tokenizer(self:Any, tokenizer_path:str) -> Any`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
    Metodo: `generate(self:Any, inputs:List[PromptType], max_out_len:int=512, temperature:float=0.6) -> str`
    Descrizione: Genera output testo o sequenze.
    Metodo: `get_token_len(self:Any, prompt:str) -> int`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\models\minimax_api.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `MiniMax`
    Metodo: `__init__(self:Any, path:str, key:str, group_id:str, model_type:str='chat', url:str='https://api.minimax.chat/v1/text/chatcompletion_pro?GroupId=', query_per_second:int=2, max_seq_len:int=2048, meta_template:Optional[Dict]=None, retry:int=2) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, inputs:List[PromptType], max_out_len:int=512) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `_generate(self:Any, input:PromptType, max_out_len:int=512) -> str`
    Descrizione: Genera output testo o sequenze.
  - `MiniMaxChatCompletionV2`
    Metodo: `__init__(self:Any, path:str, key:str, url:str, query_per_second:int=2, max_seq_len:int=2048, meta_template:Optional[Dict]=None, retry:int=2) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, inputs:List[PromptType], max_out_len:int=512) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `_generate(self:Any, input:PromptType, max_out_len:int=512) -> str`
    Descrizione: Genera output testo o sequenze.

- File: `vendor\llada\opencompass\opencompass\models\mistral_api.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `Mistral`
    Metodo: `__init__(self:Any, path:str, api_key:str, url:str, query_per_second:int=2, max_seq_len:int=2048, meta_template:Optional[Dict]=None, retry:int=2) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, inputs:List[PromptType], max_out_len:int=512) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `_generate(self:Any, input:PromptType, max_out_len:int=512) -> str`
    Descrizione: Genera output testo o sequenze.

- File: `vendor\llada\opencompass\opencompass\models\mixtral.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `Mixtral`
    Metodo: `__init__(self:Any, path:str, max_seq_len:int=2048, max_batch_size:int=8, tokenizer_only:bool=False, tokenizer_path:Optional[str]=None, meta_template:Optional[Dict]=None, num_gpus:int=2) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_load_model(self:Any, path:str, max_seq_len:int, max_batch_size:int, tokenizer_path:Optional[str]=None, num_gpus:int=2) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_load_tokenizer(self:Any, tokenizer_path:str) -> Any`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
    Metodo: `generate(self:Any, inputs:List[str], max_out_len:int) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `get_ppl(self:Any, inputs:List[str], mask_length:Optional[List[int]]=None) -> List[float]`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_token_len(self:Any, prompt:str) -> int`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\models\modelscope.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `ModelScope`
    Metodo: `__init__(self:Any, path:str, ms_cache_dir:Optional[str]=None, max_seq_len:int=2048, tokenizer_path:Optional[str]=None, tokenizer_kwargs:dict=dict(), peft_path:Optional[str]=None, tokenizer_only:bool=False, model_kwargs:dict=dict(device_map='auto'), meta_template:Optional[Dict]=None, extract_pred_after_decode:bool=False, batch_padding:bool=False, pad_token_id:Optional[int]=None, mode:str='none') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_load_tokenizer(self:Any, path:str, tokenizer_path:Optional[str], tokenizer_kwargs:dict) -> Any`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
    Metodo: `_set_model_kwargs_torch_dtype(self:Any, model_kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_load_model(self:Any, path:str, model_kwargs:dict, peft_path:Optional[str]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `ModelScopeCausalLM`
    Metodo: `_load_model(self:Any, path:str, model_kwargs:dict, peft_path:Optional[str]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\models\moonshot_api.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `MoonShot`
    Metodo: `__init__(self:Any, path:str, key:str, url:str, query_per_second:int=2, max_seq_len:int=2048, meta_template:Optional[Dict]=None, retry:int=2, system_prompt:str='') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, inputs:List[PromptType], max_out_len:int=512) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `_generate(self:Any, input:PromptType, max_out_len:int=512) -> str`
    Descrizione: Genera output testo o sequenze.

- File: `vendor\llada\opencompass\opencompass\models\nanbeige_api.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `Nanbeige`
    Metodo: `__init__(self:Any, path:str, key:str, url:str=None, query_per_second:int=2, max_seq_len:int=2048, meta_template:Optional[Dict]=None, retry:int=3) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, inputs:List[PromptType], max_out_len:int=512) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `_generate(self:Any, input:PromptType, max_out_len:int=512) -> str`
    Descrizione: Genera output testo o sequenze.

- File: `vendor\llada\opencompass\opencompass\models\openai_api.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `OpenAI`
    Metodo: `__init__(self:Any, path:str='gpt-3.5-turbo', max_seq_len:int=16384, query_per_second:int=1, rpm_verbose:bool=False, retry:int=2, key:Union[str, List[str]]='ENV', org:Optional[Union[str, List[str]]]=None, meta_template:Optional[Dict]=None, openai_api_base:str=OPENAI_API_BASE, openai_proxy_url:Optional[str]=None, mode:str='none', logprobs:Optional[bool]=False, top_logprobs:Optional[int]=None, temperature:Optional[float]=None, tokenizer_path:Optional[str]=None, extra_body:Optional[Dict]=None, verbose:bool=False, think_tag:str='</think>', max_workers:Optional[int]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, inputs:List[PromptType], max_out_len:int=512, temperature:float=0.7, **kwargs:Any) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `_generate(self:Any, input:PromptType, max_out_len:int, temperature:float) -> str`
    Descrizione: Genera output testo o sequenze.
    Metodo: `get_token_len(self:Any, prompt:str) -> int`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `_bin_trim(self:Any, prompt:str, num_token:int, mode:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_preprocess_messages(self:Any, input:Union[str, PromptList], max_out_len:int, max_seq_len:int, mode:str, get_token_len_func:Any) -> tuple[List[Dict], int]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `OpenAISDK`
    Metodo: `__init__(self:Any, path:str='gpt-3.5-turbo', max_seq_len:int=16384, query_per_second:int=1, rpm_verbose:bool=False, retry:int=2, key:str | List[str]='ENV', org:str | List[str] | None=None, meta_template:Dict | None=None, openai_api_base:str | List[str]=OPENAISDK_API_BASE, openai_proxy_url:Optional[str]=None, mode:str='none', logprobs:bool | None=False, top_logprobs:int | None=None, temperature:float | None=None, tokenizer_path:str | None=None, extra_body:Dict | None=None, verbose:bool=False, http_client_cfg:dict={}, status_code_mappings:dict={}, think_tag:str='</think>', max_workers:Optional[int]=None, openai_extra_kwargs:Dict | None=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_generate(self:Any, input:PromptList | str, max_out_len:int, temperature:float, timeout:int=3600) -> str`
    Descrizione: Genera output testo o sequenze.

- File: `vendor\llada\opencompass\opencompass\models\openai_streaming.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `OpenAISDKStreaming`
    Metodo: `__init__(self:Any, path:str='gpt-3.5-turbo', max_seq_len:int=16384, query_per_second:int=1, rpm_verbose:bool=False, retry:int=2, key:str | List[str]='ENV', org:str | List[str] | None=None, meta_template:Dict | None=None, openai_api_base:str | List[str]=OPENAISDK_API_BASE, openai_proxy_url:Optional[str]=None, mode:str='none', logprobs:bool | None=False, top_logprobs:int | None=None, temperature:float | None=None, tokenizer_path:str | None=None, extra_body:Dict | None=None, verbose:bool=False, http_client_cfg:dict={}, status_code_mappings:dict={}, think_tag:str='</think>', stream:bool=True, stream_chunk_size:int=1) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_create_fresh_client(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_generate(self:Any, input:PromptList | str, max_out_len:int, temperature:float, timeout:int=3600) -> str`
    Descrizione: Genera output testo o sequenze.
    Metodo: `_handle_stream_response(self:Any, response_stream:Any, thread_id:Any=None) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `estimate_token_count(self:Any, text:str) -> int`
    Descrizione: Gestisce tokenizzazione o manipolazione token.

- File: `vendor\llada\opencompass\opencompass\models\pangu_api.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `PanGu`
    Metodo: `__init__(self:Any, path:str, access_key:str, secret_key:str, url:str, token_url:str, project_name:str, query_per_second:int=2, max_seq_len:int=2048, meta_template:Optional[Dict]=None, retry:int=2) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, inputs:List[PromptType], max_out_len:int=512) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `_get_token(self:Any) -> Any`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
    Metodo: `_generate(self:Any, input:PromptType, max_out_len:int=512) -> str`
    Descrizione: Genera output testo o sequenze.

- File: `vendor\llada\opencompass\opencompass\models\qwen_api.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `Qwen`
    Metodo: `__init__(self:Any, path:str, key:str, query_per_second:int=1, max_seq_len:int=2048, meta_template:Optional[Dict]=None, retry:int=5, generation_kwargs:Dict={}) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, inputs:List[PromptType], max_out_len:int=512) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `_generate(self:Any, input:PromptType, max_out_len:int=512) -> str`
    Descrizione: Genera output testo o sequenze.

- File: `vendor\llada\opencompass\opencompass\models\rendu_api.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `Rendu`
    Metodo: `__init__(self:Any, path:str, key:str, url:str, query_per_second:int=2, max_seq_len:int=2048, meta_template:Optional[Dict]=None, retry:int=2, generation_kwargs:Dict={'temperature': 0.7, 'top_p': 0.9}) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, inputs:List[PromptType], max_out_len:int=512) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `_generate(self:Any, input:PromptType, max_out_len:int=512) -> str`
    Descrizione: Genera output testo o sequenze.

- File: `vendor\llada\opencompass\opencompass\models\sensetime_api.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `SenseTime`
    Metodo: `__init__(self:Any, path:str, url:str, key:str='ENV', query_per_second:int=2, max_seq_len:int=2048, meta_template:Optional[Dict]=None, retry:int=2, parameters:Optional[Dict]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, inputs:List[PromptType], max_out_len:int=512) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `_generate(self:Any, input:PromptType, max_out_len:int=512) -> str`
    Descrizione: Genera output testo o sequenze.

- File: `vendor\llada\opencompass\opencompass\models\stepfun_api.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `StepFun`
    Metodo: `__init__(self:Any, path:str, key:str, url:str, query_per_second:int=2, max_seq_len:int=2048, meta_template:Optional[Dict]=None, retry:int=2, system_prompt:str='') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, inputs:List[PromptType], max_out_len:int=512) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `_generate(self:Any, input:PromptType, max_out_len:int=512) -> str`
    Descrizione: Genera output testo o sequenze.

- File: `vendor\llada\opencompass\opencompass\models\turbomind.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `valid_str(string:Any, coding:Any='utf-8') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `TurboMindModel`
    Metodo: `__init__(self:Any, path:str, backend:str='turbomind', max_seq_len:int=2048, meta_template:Optional[Dict]=None, engine_config:Dict={}, gen_config:Dict={}, batch_padding:bool=False, drop_middle:bool=False, end_str:Optional[str]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, inputs:List[str], max_out_len:int=512, stopping_criteria:List[str]=[], do_sample:Optional[bool]=None, temperature:int=1, **kwargs:Any) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `get_token_len(self:Any, prompt:str) -> int`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `wait(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_ppl(self:Any, inputs:List[str], mask_length:Optional[List[int]]=None) -> np.ndarray`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_loglikelihood(self:Any, inputs:List[str], conts:List[str], mask_length:Optional[List[int]]=None) -> List[float]`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `_build_pipe(self:Any, model_path:Any, backend:Any, engine_config:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\models\turbomind_api.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `valid_str(string:Any, coding:Any='utf-8') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `TurboMindAPIModel`
    Metodo: `__init__(self:Any, model_name:str=None, api_addr:str='http://0.0.0.0:23333', api_key:str | None=None, max_seq_len:int=2048, meta_template:Optional[Dict]=None, end_str:Optional[str]=None, temperature:float=None, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, inputs:List[PromptType], max_out_len:int=512, temperature:float=1.0) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `get_token_len(self:Any, prompt:str) -> int`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `wait(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_generate(self:Any, prompt:PromptType, max_out_len:int, temperature:float, end_str:str) -> str`
    Descrizione: Genera output testo o sequenze.

- File: `vendor\llada\opencompass\opencompass\models\turbomind_with_tf_above_v4_33.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `valid_str(string:Any, coding:Any='utf-8') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `TurboMindModelwithChatTemplate`
    Metodo: `__init__(self:Any, path:str, tokenizer_only:bool=False, backend:str='turbomind', engine_config:Dict | ConfigDict={}, gen_config:Dict={}, max_seq_len:int=None, meta_template:Optional[Dict]=None, fastchat_template:Optional[str]=None, stop_words:List[str]=[], drop_middle:bool=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_get_potential_stop_words(self:Any, path:Optional[str]) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, inputs:List[str], max_out_len:int, min_out_len:Optional[int]=None, stopping_criteria:List[str]=[], do_sample:Optional[bool]=None, temperature:float=1.0, **kwargs:Any) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `get_token_len(self:Any, prompt:str) -> int`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `_build_pipe(self:Any, model_path:Any, backend:Any, engine_config:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\models\unigpt_api.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_sign(appkey:Any, udid:Any, timestamp:Any, secret:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  Classi:
  - `UniGPT`
    Metodo: `__init__(self:Any, path:str, appkey:str, secret:str, url:str, query_per_second:int=2, max_seq_len:int=2048, meta_template:Optional[Dict]=None, retry:int=2, temperature:float=0.2) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, inputs:List[PromptType], max_out_len:int=512) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `_generate(self:Any, input:PromptType, max_out_len:int=512) -> str`
    Descrizione: Genera output testo o sequenze.

- File: `vendor\llada\opencompass\opencompass\models\vllm.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `VLLM`
    Metodo: `__init__(self:Any, path:str, max_seq_len:int=2048, model_kwargs:dict=None, generation_kwargs:dict=dict(), meta_template:Optional[Dict]=None, mode:str='none', use_fastchat_template:bool=False, lora_path:str=None, stop_words:List[str]=[]) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_load_model(self:Any, path:str, add_model_kwargs:dict=None, num_retry:int=3) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, inputs:List[str], max_out_len:int, stopping_criteria:List[str]=[], **kwargs:Any) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `get_ppl(self:Any, inputs:List[str], mask_length:Optional[List[int]]=None) -> List[float]`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_loglikelihood(self:Any, inputs:List[str], conts:List[str]) -> List[float]`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_token_len(self:Any, prompt:str, add_special_tokens:bool=True) -> int`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\models\vllm_with_tf_above_v4_33.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `VLLMwithChatTemplate`
    Metodo: `__init__(self:Any, path:str, model_kwargs:dict=dict(), tokenizer_only:bool=False, generation_kwargs:dict=dict(), max_seq_len:int=None, meta_template:Optional[Dict]=None, fastchat_template:Optional[str]=None, stop_words:List[str]=[]) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_load_model(self:Any, path:str, added_model_kwargs:dict=dict()) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_get_potential_stop_words(self:Any, path:Optional[str]) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, inputs:List[str], max_out_len:int, stopping_criteria:List[str]=[], **kwargs:Any) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `get_token_len(self:Any, prompt:str) -> int`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\models\xunfei_api.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `XunFei`
    Metodo: `__init__(self:Any, path:str, appid:str, api_secret:str, api_key:str, domain:str='general', query_per_second:int=2, max_seq_len:int=2048, meta_template:Optional[Dict]=None, retry:int=2) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_url(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `generate(self:Any, inputs:List[PromptType], max_out_len:int=512) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `_generate(self:Any, input:PromptType, max_out_len:int=512) -> List[str]`
    Descrizione: Genera output testo o sequenze.
  - `XunFeiSpark`
    Metodo: `__init__(self:Any, path:str, url:str, app_id:str, api_key:str, api_secret:str, query_per_second:int=2, max_seq_len:int=2048, meta_template:Optional[Dict]=None, retry:int=2) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, inputs:List[PromptType], max_out_len:int=512) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `_generate(self:Any, input:PromptType, max_out_len:int=512) -> str`
    Descrizione: Genera output testo o sequenze.

- File: `vendor\llada\opencompass\opencompass\models\yayi_api.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `generate_random_string(length:Any=16) -> Any`
    Descrizione: Genera output testo o sequenze.
  - `get_current_time(format:Any='%Y-%m-%d %H:%M:%S') -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `get_current_timestamp() -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `encode_base64_string(s:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_current_time_gmt_format() -> Any`
    Descrizione: Recupera valore/stato calcolato.
  Classi:
  - `Yayi`
    Metodo: `__init__(self:Any, path:str, url:str, url_path:str, x_tilake_app_key:str, x_tilake_app_secret:str, x_tilake_ca_sginature_method:str, query_per_second:int=2, max_seq_len:int=2048, meta_template:Optional[Dict]=None, retry:int=2, temperature:float=0.4) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate_signature(self:Any, method:Any, accept:Any, content_type:Any, date:Any, url_path:Any) -> Any`
    Descrizione: Genera output testo o sequenze.
    Metodo: `generate_header(self:Any, content_type:Any, accept:Any, date:Any, signature:Any) -> Any`
    Descrizione: Genera output testo o sequenze.
    Metodo: `generate(self:Any, inputs:List[PromptType], max_out_len:int=512) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `_generate(self:Any, input:PromptType, max_out_len:int=512) -> str`
    Descrizione: Genera output testo o sequenze.

- File: `vendor\llada\opencompass\opencompass\models\yi_api.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `YiAPI`
    Metodo: `__init__(self:Any, path:str, key:str, url:str, query_per_second:int=2, max_seq_len:int=2048, meta_template:Optional[Dict]=None, retry:int=2, system_prompt:str='') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, inputs:List[PromptType], max_out_len:int=512) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `_generate(self:Any, input:PromptType, max_out_len:int=512) -> str`
    Descrizione: Genera output testo o sequenze.

- File: `vendor\llada\opencompass\opencompass\models\zhipuai_api.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `ZhiPuAI`
    Metodo: `__init__(self:Any, path:str, key:str, query_per_second:int=2, max_seq_len:int=2048, meta_template:Optional[Dict]=None, retry:int=2) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, inputs:List[PromptType], max_out_len:int=512) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `_generate(self:Any, input:PromptType, max_out_len:int=512) -> str`
    Descrizione: Genera output testo o sequenze.

- File: `vendor\llada\opencompass\opencompass\models\zhipuai_v2_api.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `ZhiPuV2AI`
    Metodo: `__init__(self:Any, path:str, key:str, query_per_second:int=2, max_seq_len:int=2048, meta_template:Optional[Dict]=None, retry:int=2, generation_kwargs:Dict={'tools': [{'type': 'web_search', 'enable': False}]}) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, inputs:List[PromptType], max_out_len:int=512) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `_generate(self:Any, input:PromptType, max_out_len:int=512) -> str`
    Descrizione: Genera output testo o sequenze.

- File: `vendor\llada\opencompass\opencompass\openicl\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\openicl\icl_dataset_reader.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `load_partial_dataset(dataset:Dataset, size:Optional[Union[int, float, str]]=None) -> Dataset`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  Classi:
  - `DatasetReader`
    Metodo: `__init__(self:Any, dataset:Union[Dataset, DatasetDict, str], input_columns:Union[List[str], str], output_column:Optional[str], input_template:Optional[PromptTemplate]=None, output_template:Optional[PromptTemplate]=None, train_split:str='train', train_range:Optional[Union[int, float, str]]=None, test_split:str='test', test_range:Optional[Union[int, float, str]]=None) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate_input_field_prompt(self:Any, entry:Dict) -> str`
    Descrizione: Genera output testo o sequenze.
    Metodo: `generate_input_field_corpus(self:Any, dataset:Union[Dataset, DatasetDict], split:Optional[str]=None) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `generate_output_field_prompt(self:Any, entry:Dict) -> str`
    Descrizione: Genera output testo o sequenze.
    Metodo: `generate_output_field_corpus(self:Any, dataset:Union[Dataset, DatasetDict], split:Optional[str]=None) -> List[str]`
    Descrizione: Genera output testo o sequenze.
    Metodo: `generate_input_output_field_prompt(self:Any, entry:Dict) -> str`
    Descrizione: Genera output testo o sequenze.
    Metodo: `_check_dataset_reader(obj:Any) -> 'DatasetReader'`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__len__(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__getitem__(self:Any, idx:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__repr__(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `DatasetEncoder`
    Metodo: `__init__(self:Any, datalist:List, model_name:Any=None, tokenizer:Any=None) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `init_dataset(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__len__(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__getitem__(self:Any, idx:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\openicl\icl_evaluator\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\openicl\icl_evaluator\code_evaluator.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `CodeEvaluator`
    Metodo: `__init__(self:Any, language:str='py', ip_address:str='localhost', retry:int=5) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_extract_code(self:Any, text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_code_eval_service(self:Any, input_data:Union[Dict, List, str]) -> Tuple[bool, Union[Dict, List, Any]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_process_completions(self:Any, completion:str) -> list`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_evaluate(self:Any, input_data:Union[Dict, List]) -> Tuple[bool, Optional[Union[Dict, List]], Optional[str]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_process_results(self:Any, outputs:List, prompts:List, total_count:int) -> Dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:List, references:List, test_set:Dataset) -> Dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\openicl\icl_evaluator\hf_metrics\accuracy.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `Accuracy`
    Metodo: `_info(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_compute(self:Any, predictions:Any, references:Any, normalize:Any=True, sample_weight:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\openicl\icl_evaluator\hf_metrics\rouge.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `Tokenizer`
    Metodo: `__init__(self:Any, tokenizer_func:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `tokenize(self:Any, text:Any) -> Any`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
  - `Rouge`
    Metodo: `_info(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_compute(self:Any, predictions:Any, references:Any, rouge_types:Any=None, use_aggregator:Any=True, use_stemmer:Any=False, tokenizer:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\openicl\icl_evaluator\hf_metrics\sacrebleu.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `Sacrebleu`
    Metodo: `_info(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_compute(self:Any, predictions:Any, references:Any, smooth_method:Any='exp', smooth_value:Any=None, force:Any=False, lowercase:Any=False, tokenize:Any=None, use_effective_order:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\openicl\icl_evaluator\hf_metrics\squad.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `Squad`
    Metodo: `_info(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_compute(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\openicl\icl_evaluator\icl_agent_evaluator.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `extract_answer(result:dict) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `PassRateEvaluator`
    Metodo: `__init__(self:Any, fail_words:Any=DEFAULT_FAIL_WORDS) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:List, references:List=None) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `check_real_valid(self:Any, answer:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `WinRateEvaluator`
    Metodo: `__init__(self:Any, model:Any='gpt-3.5-turbo-16k', temperature:Any=0, **kwargs:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:List, references:List, origin_prompt:List, steps:List) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `check_solve_query(self:Any, query:str, answer:str) -> bool`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `select_best_final_answer(self:Any, query:str, answers:list) -> int`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `compare_steps(self:Any, steps_list:list) -> int`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_openai_function(self:Any, msg:str, max_out_len:int, functions:dict, function_call:dict, **kwargs:Any) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\openicl\icl_evaluator\icl_aucroc_evaluator.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `AUCROCEvaluator`
    Metodo: `__init__(self:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:List, references:List) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\openicl\icl_evaluator\icl_base_evaluator.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `compute_pass_at_k(n:Any, c:Any, k:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_compute_g_pass_at_k(n:Any, c:Any, k:Any, m:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `compute_g_pass_at_k(n:Any, c:Any, k:Any, t:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `compute_mg_pass_at_k(n:Any, c:Any, k:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `BaseEvaluator`
    Metodo: `__init__(self:Any, pred_postprocessor:Any=None) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `output_dir(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `dataset_replica_idx(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `group(self:Any, n:int, details:List[Dict[str, Any]], test_set:Dataset) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `reduce(self:Any, details:List[Dict[str, Any]]) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `pred_postprocess(self:Any, predictions:List) -> Dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `evaluate(self:Any, k:Union[int, List[int]], n:int, original_dataset:Dataset, **score_kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `is_num_equal(predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\openicl\icl_evaluator\icl_bpc_evaluator.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `BPCEvaluator`
    Metodo: `score(self:Any, loss:List[float], total_chr_num:List[float]) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\openicl\icl_evaluator\icl_circular_evaluator.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `CircularEvaluator`
    Metodo: `__init__(self:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\openicl\icl_evaluator\icl_em_evaluator.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `EMEvaluator`
    Metodo: `__init__(self:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\openicl\icl_evaluator\icl_hf_evaluator.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `HuggingfaceEvaluator`
    Metodo: `__init__(self:Any, metric:str, seed:int=0, pred_postprocessor:Any=None) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_preprocess(self:Any, predictions:List, references:List) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_postprocess(self:Any, scores:dict) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:List, references:List, test_set:Any=None) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `AccEvaluator`
    Metodo: `__init__(self:Any, pred_postprocessor:Optional[ConfigDict]=None) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_preprocess(self:Any, predictions:List, references:List, test_set:Any=None) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_postprocess(self:Any, scores:dict) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `AccContaminationEvaluator`
    Metodo: `score(self:Any, predictions:List, references:List, test_set:Dataset) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `RougeEvaluator`
    Metodo: `__init__(self:Any, pred_postprocessor:Optional[ConfigDict]=None) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_postprocess(self:Any, scores:dict) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `BleuEvaluator`
    Metodo: `__init__(self:Any, pred_postprocessor:Optional[ConfigDict]=None) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `BleuFloresEvaluator`
    Metodo: `__init__(self:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_preprocess(self:Any, predictions:List, references:List) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `MccEvaluator`
    Metodo: `__init__(self:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_postprocess(self:Any, scores:dict) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `SquadEvaluator`
    Metodo: `__init__(self:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_preprocess(self:Any, predictions:List, references:List) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_postprocess(self:Any, scores:dict) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `EDAccEvaluator`
    Metodo: `__init__(self:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_preprocess(self:Any, predictions:List, references:List) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `AccwithDetailsEvaluator`
    Metodo: `score(self:Any, predictions:Any, references:Any, origin_prompt:Any) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\openicl\icl_evaluator\icl_jieba_rouge_evaluator.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `JiebaRougeEvaluator`
    Metodo: `__init__(self:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\openicl\icl_evaluator\icl_judge_evaluator.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `JudgeEvaluator`
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `RMBEvaluator`
    Metodo: `calculate_pair_accuracy(self:Any, data:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `calculate_bon_accuracy(self:Any, data:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `Judgerbenchv2Evaluator`
    Metodo: `get_rank_dict(self:Any, score_dict:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `extract_winner(self:Any, s:Any, lan:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\openicl\icl_evaluator\icl_korbench_evaluator.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `read_json_or_jsonl(data_path:Any, split:Any='', mapping_key:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `read_json_or_jsonl_with_idx(data_path:Any, split:Any='', idx:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `korbenchEvaluator`
    Metodo: `__init__(self:Any, question_type:Any, mode:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `evaluate_responses(self:Any, data:Any, question_type:Any, mode:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `extract_text_from_brackets(self:Any, text:Any, clean_level:Any='basic') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `clean_latex(self:Any, latex_expr:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `evaluate_response_vs_answer(self:Any, response:Any, answer:Any, question_type:Any, rule_id:Any, idx:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `rule5_normalize_content(self:Any, content:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\openicl\icl_evaluator\icl_misc_evaluator.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `AveragePPLEvaluator`
    Metodo: `score(self:Any, ppl:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `AverageMinKEvaluator`
    Metodo: `score(self:Any, mink:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `AverageInferencePPLEvaluator`
    Metodo: `score(self:Any, ppl:Any, token_len:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\openicl\icl_evaluator\icl_plugin_evaluator.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `TEvalEvaluator`
    Metodo: `__init__(self:Any, subset:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, references:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\openicl\icl_evaluator\icl_toxic_evaluator.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `PerspectiveAPIClient`
    Metodo: `__init__(self:Any, key:str, batch_size:int, max_length:int=20480) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_initialize(self:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `create_request_body(text:str) -> dict`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
    Metodo: `extract_toxicity_attributes(self:Any, response:dict) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_toxicity_scores(self:Any, predictions:List) -> dict`
    Descrizione: Recupera valore/stato calcolato.
  - `ToxicEvaluator`
    Metodo: `__init__(self:Any, key:str='ENV', thr:float=0.5, batch_size:int=4) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_scores(self:Any, predictions:List) -> dict`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_metrics(self:Any, scores:dict) -> dict`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `score(self:Any, predictions:List, references:List) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\openicl\icl_evaluator\lm_evaluator.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `extract_dicts(data:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `order_preds_and_record_references(predictions:List, references:List, infer_order:List, seed:int=666, keep_preds:bool=False, base_model_abbrs:List[str]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `count_chinese_characters(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `count_english_words(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `LMEvaluator`
    Metodo: `__init__(self:Any, prompt_template:ConfigDict, judge_cfg:ConfigDict, output_path:str, meta_review_prompt_template:Optional[ConfigDict]=None, pack_all_predictions:Optional[bool]=False, dataset_cfg:Optional[ConfigDict]=None, pred_postprocessor:Optional[ConfigDict]=None, dict_postprocessor:Optional[ConfigDict]=None, keep_predictions:bool=False, multi_eval:bool=False) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `score(self:Any, predictions:Any, judgements:Optional[List]=None, references:Optional[List]=None, meta:Optional[bool]=False, infer_order:Optional[str]='random') -> Dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `postprocess(self:Any, output:Dict) -> Dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\openicl\icl_inferencer\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\openicl\icl_inferencer\icl_agent_inferencer.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `model_adapter(model:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `AgentInferencerOutputHandler`
    Metodo: `__init__(self:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `write_to_json(self:Any, save_dir:str, filename:str) -> Any`
    Descrizione: Serializza e salva output su disco.
    Metodo: `save_results(self:Any, origin_prompt:list, prediction:str, steps:list, idx:int, gold:str=None) -> Any`
    Descrizione: Serializza e salva output su disco.
    Metodo: `save_multiround_results(self:Any, origin_prompt:list, prediction:str, steps:list, idx:int, gold:str=None) -> Any`
    Descrizione: Serializza e salva output su disco.
  - `AgentInferencer`
    Metodo: `__init__(self:Any, model:Any, **kwargs:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `infer_last(self:Any, chat:List[dict], index:int, output_handler:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `infer_every(self:Any, chat:List[dict], index:int, output_handler:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `infer_every_with_gt(self:Any, chat:List[dict], index:int, output_handler:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\openicl\icl_inferencer\icl_attack_inferencer.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `AttackInferencer`
    Metodo: `__init__(self:Any, model:BaseModel, max_out_len:int, adv_key:str, metric_key:str='accuracy', max_seq_len:Optional[int]=None, batch_size:Optional[int]=1, gen_field_replace_token:Optional[str]='', output_json_filepath:Optional[str]='./icl_inference_output', output_json_filename:Optional[str]='predictions', save_every:Optional[int]=1, dataset_cfg:Optional[List[int]]=None, **kwargs:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `predict(self:Any, adv_prompt:Any) -> List`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_generation_prompt_list_from_retriever_indices(self:Any, ice_idx_list:List[List[int]], extra_prompt:dict, retriever:BaseRetriever, gen_field_replace_token:str, max_seq_len:Optional[int]=None, ice_template:Optional[PromptTemplate]=None, prompt_template:Optional[PromptTemplate]=None) -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\openicl\icl_inferencer\icl_base_inferencer.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `dump_results_dict(results_dict:Any, filename:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `BaseInferencer`
    Metodo: `__init__(self:Any, model:Any, max_seq_len:Optional[int]=None, batch_size:Optional[int]=1, output_json_filepath:Optional[str]='./icl_inference_output', output_json_filename:Optional[str]='predictions', fix_id_list:Optional[List[int]]=None, **kwargs:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `inference(self:Any, retriever:BaseRetriever, ice_template:Optional[PromptTemplate]=None, prompt_template:Optional[PromptTemplate]=None, output_json_filepath:Optional[str]=None, output_json_filename:Optional[str]=None) -> List`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_dataloader(datalist:List[List], batch_size:int) -> DataLoader`
    Descrizione: Recupera valore/stato calcolato.
  - `GenInferencerOutputHandler`
    Metodo: `__init__(self:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `write_to_json(self:Any, save_dir:str, filename:str) -> Any`
    Descrizione: Serializza e salva output su disco.
    Metodo: `save_results(self:Any, origin_prompt:Any, prediction:Any, idx:Any, gold:Any=None) -> Any`
    Descrizione: Serializza e salva output su disco.
  - `PPLInferencerOutputHandler`
    Metodo: `__init__(self:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `write_to_json(self:Any, save_dir:str, filename:str) -> Any`
    Descrizione: Serializza e salva output su disco.
    Metodo: `save_ice(self:Any, ice:Any) -> Any`
    Descrizione: Serializza e salva output su disco.
    Metodo: `save_predictions(self:Any, predictions:Any) -> Any`
    Descrizione: Serializza e salva output su disco.
    Metodo: `save_prompt_and_ppl(self:Any, label:Any, input:Any, prompt:Any, ppl:Any, idx:Any) -> Any`
    Descrizione: Serializza e salva output su disco.
    Metodo: `save_golds(self:Any, golds:Any) -> Any`
    Descrizione: Serializza e salva output su disco.
  - `CLPInferencerOutputHandler`
    Metodo: `__init__(self:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `write_to_json(self:Any, save_dir:str, filename:str) -> Any`
    Descrizione: Serializza e salva output su disco.
    Metodo: `save_ice(self:Any, ice:Any) -> Any`
    Descrizione: Serializza e salva output su disco.
    Metodo: `save_prompt_and_condprob(self:Any, input:Any, prompt:Any, cond_prob:Any, idx:Any, choices:Any, gold:Any=None) -> Any`
    Descrizione: Serializza e salva output su disco.

- File: `vendor\llada\opencompass\opencompass\openicl\icl_inferencer\icl_chat_inferencer.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `promptlist_to_openai(prompt:Union[str, PromptList]) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `LMTemplateParser`
    Metodo: `__init__(self:Any, meta_template:Optional[dict]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `parse_template(self:Any, chat:List[dict], mode:Any='gen') -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `APITemplateParser`
    Metodo: `__init__(self:Any, meta_template:Optional[dict]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `parse_template(self:Any, chat:List[dict], mode:Any='gen') -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `ChatOutputHandler`
    Metodo: `__init__(self:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `write_to_json(self:Any, save_dir:str, filename:str) -> Any`
    Descrizione: Serializza e salva output su disco.
    Metodo: `save_results(self:Any, origin_prompt:list, prediction:str, idx:int, gold:str=None) -> Any`
    Descrizione: Serializza e salva output su disco.
    Metodo: `save_multiround_results(self:Any, origin_prompt:list, prediction:str, idx:int, gold:str=None) -> Any`
    Descrizione: Serializza e salva output su disco.
  - `ChatInferencer`
    Metodo: `__init__(self:Any, model:Any, output_json_filepath:Optional[str]='./icl_inference_output', output_json_filename:Optional[str]='predictions', save_every:Optional[int]=1, infer_mode:str='last', max_out_len:int=512, **kwargs:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_set_meta_template(self:Any, model:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `inference(self:Any, retriever:BaseRetriever, ice_template:Optional[PromptTemplate]=None, prompt_template:Optional[PromptTemplate]=None, output_json_filepath:Optional[str]=None, output_json_filename:Optional[str]=None) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_chat_list(self:Any, ice_idx_list:List[List[int]], retriever:BaseRetriever, prompt_template:Optional[PromptTemplate]=None) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `infer_last(self:Any, chat:List[dict], index:int, output_handler:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `infer_every(self:Any, chat:List[dict], index:int, output_handler:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `infer_every_with_gt(self:Any, chat:List[dict], index:int, output_handler:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\openicl\icl_inferencer\icl_clp_inferencer.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `CLPInferencer`
    Metodo: `__init__(self:Any, model:BaseModel, max_seq_len:Optional[int]=None, batch_size:Optional[int]=1, output_json_filepath:Optional[str]='./icl_inference_output', output_json_filename:Optional[str]='predictions', single_token:bool=True, **kwargs:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `inference(self:Any, retriever:BaseRetriever, ice_template:Optional[PromptTemplate]=None, prompt_template:Optional[PromptTemplate]=None, output_json_filepath:Optional[str]=None, output_json_filename:Optional[str]=None, normalizing_str:Optional[str]=None) -> List`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_get_cond_prob(self:Any, input_texts:List[str], target_pos:List[int], choice_ids:List[int]) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\openicl\icl_inferencer\icl_gen_inferencer.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `GenInferencer`
    Metodo: `__init__(self:Any, model:BaseModel, max_out_len:int, stopping_criteria:List[str]=[], max_seq_len:Optional[int]=None, min_out_len:Optional[int]=None, batch_size:Optional[int]=1, gen_field_replace_token:Optional[str]='', output_json_filepath:Optional[str]='./icl_inference_output', output_json_filename:Optional[str]='predictions', save_every:Optional[int]=1, **kwargs:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `inference(self:Any, retriever:BaseRetriever, ice_template:Optional[PromptTemplate]=None, prompt_template:Optional[PromptTemplate]=None, output_json_filepath:Optional[str]=None, output_json_filename:Optional[str]=None) -> List`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_generation_prompt_list_from_retriever_indices(self:Any, ice_idx_list:List[List[int]], retriever:BaseRetriever, gen_field_replace_token:str, max_seq_len:Optional[int]=None, ice_template:Optional[PromptTemplate]=None, prompt_template:Optional[PromptTemplate]=None) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `GLMChoiceInferencer`
    Metodo: `__init__(self:Any, *args:Any, choices:Any=['A', 'B', 'C', 'D'], **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `inference(self:Any, retriever:BaseRetriever, ice_template:Optional[PromptTemplate]=None, prompt_template:Optional[PromptTemplate]=None, output_json_filepath:Optional[str]=None, output_json_filename:Optional[str]=None) -> List`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\openicl\icl_inferencer\icl_inference_ppl_only_inferencer.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `InferencePPLOnlyInferencer`
    Metodo: `__init__(self:Any, model:BaseModel, max_seq_len:Optional[int]=None, batch_size:Optional[int]=1, output_json_filepath:Optional[str]='./icl_inference_output', output_json_filename:Optional[str]='predictions', save_every:Optional[int]=1, **kwargs:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `inference(self:Any, retriever:BaseRetriever, ice_template:Optional[PromptTemplate]=None, prompt_template:Optional[PromptTemplate]=None, output_json_filepath:Optional[str]=None, output_json_filename:Optional[str]=None) -> List`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_generation_prompt_list_from_retriever_indices(self:Any, ice_idx_list:List[List[int]], retriever:BaseRetriever, max_seq_len:Optional[int]=None, ice_template:Optional[PromptTemplate]=None, prompt_template:Optional[PromptTemplate]=None) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_generation_prompt_list_and_label(self:Any, ice_idx_list:List[List[int]], retriever:BaseRetriever, max_seq_len:Optional[int]=None, ice_template:Optional[PromptTemplate]=None, prompt_template:Optional[PromptTemplate]=None) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `InferencePPLOnlyInferencerOutputHandler`
    Metodo: `__init__(self:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `write_to_json(self:Any, save_dir:str, filename:str) -> Any`
    Descrizione: Serializza e salva output su disco.
    Metodo: `save_results(self:Any, origin_prompt:Any, ppl:Any, token_len:Any, idx:Any) -> Any`
    Descrizione: Serializza e salva output su disco.

- File: `vendor\llada\opencompass\opencompass\openicl\icl_inferencer\icl_ll_inferencer.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LLInferencer`
    Metodo: `__init__(self:Any, model:BaseModel, max_seq_len:Optional[int]=None, batch_size:Optional[int]=1, output_json_filepath:Optional[str]='./icl_inference_output', output_json_filename:Optional[str]='predictions', labels:Optional[List]=None, **kwargs:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `inference(self:Any, retriever:BaseRetriever, ice_template:Optional[PromptTemplate]=None, prompt_template:Optional[PromptTemplate]=None, output_json_filepath:Optional[str]=None, output_json_filename:Optional[str]=None) -> List`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LLInferencerOutputHandler`
    Metodo: `__init__(self:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `write_to_json(self:Any, save_dir:str, filename:str) -> Any`
    Descrizione: Serializza e salva output su disco.
    Metodo: `save_ice(self:Any, ice:Any) -> Any`
    Descrizione: Serializza e salva output su disco.
    Metodo: `save_predictions(self:Any, predictions:Any) -> Any`
    Descrizione: Serializza e salva output su disco.
    Metodo: `save_prompt_and_loglikelihood(self:Any, label:Any, input:Any, prompt:Any, loglikelihood:Any, idx:Any) -> Any`
    Descrizione: Serializza e salva output su disco.
    Metodo: `save_golds(self:Any, golds:Any) -> Any`
    Descrizione: Serializza e salva output su disco.

- File: `vendor\llada\opencompass\opencompass\openicl\icl_inferencer\icl_mink_percent_inferencer.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `MinKPercentInferencer`
    Metodo: `__init__(self:Any, model:BaseModel, max_seq_len:Optional[int]=None, batch_size:Optional[int]=1, output_json_filepath:Optional[str]='./icl_inference_output', output_json_filename:Optional[str]='predictions', save_every:Optional[int]=1, **kwargs:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `inference(self:Any, retriever:BaseRetriever, ice_template:Optional[PromptTemplate]=None, prompt_template:Optional[PromptTemplate]=None, output_json_filepath:Optional[str]=None, output_json_filename:Optional[str]=None) -> List`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_generation_prompt_list_from_retriever_indices(self:Any, ice_idx_list:List[List[int]], retriever:BaseRetriever, max_seq_len:Optional[int]=None, ice_template:Optional[PromptTemplate]=None, prompt_template:Optional[PromptTemplate]=None) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `PPLOnlyInferencerOutputHandler`
    Metodo: `__init__(self:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `write_to_json(self:Any, save_dir:str, filename:str) -> Any`
    Descrizione: Serializza e salva output su disco.
    Metodo: `save_results(self:Any, origin_prompt:Any, mink:Any, idx:Any) -> Any`
    Descrizione: Serializza e salva output su disco.

- File: `vendor\llada\opencompass\opencompass\openicl\icl_inferencer\icl_ppl_inferencer.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `PPLInferencer`
    Metodo: `__init__(self:Any, model:BaseModel, max_seq_len:Optional[int]=None, batch_size:Optional[int]=1, output_json_filepath:Optional[str]='./icl_inference_output', output_json_filename:Optional[str]='predictions', labels:Optional[List]=None, **kwargs:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `inference(self:Any, retriever:BaseRetriever, ice_template:Optional[PromptTemplate]=None, prompt_template:Optional[PromptTemplate]=None, output_json_filepath:Optional[str]=None, output_json_filename:Optional[str]=None, normalizing_str:Optional[str]=None) -> List`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\openicl\icl_inferencer\icl_ppl_only_inferencer.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `PPLOnlyInferencer`
    Metodo: `__init__(self:Any, model:BaseModel, max_seq_len:Optional[int]=None, batch_size:Optional[int]=1, output_json_filepath:Optional[str]='./icl_inference_output', output_json_filename:Optional[str]='predictions', save_every:Optional[int]=1, **kwargs:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `inference(self:Any, retriever:BaseRetriever, ice_template:Optional[PromptTemplate]=None, prompt_template:Optional[PromptTemplate]=None, output_json_filepath:Optional[str]=None, output_json_filename:Optional[str]=None) -> List`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_generation_prompt_list_from_retriever_indices(self:Any, ice_idx_list:List[List[int]], retriever:BaseRetriever, max_seq_len:Optional[int]=None, ice_template:Optional[PromptTemplate]=None, prompt_template:Optional[PromptTemplate]=None) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `PPLOnlyInferencerOutputHandler`
    Metodo: `__init__(self:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `write_to_json(self:Any, save_dir:str, filename:str) -> Any`
    Descrizione: Serializza e salva output su disco.
    Metodo: `save_results(self:Any, origin_prompt:Any, ppl:Any, idx:Any) -> Any`
    Descrizione: Serializza e salva output su disco.

- File: `vendor\llada\opencompass\opencompass\openicl\icl_inferencer\icl_sc_inferencer.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `SCInferencer`
    Metodo: `__init__(self:Any, model:BaseModel, max_out_len:int, max_seq_len:Optional[int]=None, batch_size:Optional[int]=1, gen_field_replace_token:Optional[str]='', output_json_filepath:Optional[str]='./icl_inference_output', output_json_filename:Optional[str]='predictions', save_every:Optional[int]=1, sc_size:Optional[int]=1, infer_type:Optional[str]='', generation_kwargs:dict={}, **kwargs:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `inference(self:Any, retriever:BaseRetriever, ice_template:Optional[PromptTemplate]=None, prompt_template:Optional[PromptTemplate]=None, output_json_filepath:Optional[str]=None, output_json_filename:Optional[str]=None) -> List`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_generation_prompt_list_from_retriever_indices(self:Any, ice_idx_list:List[List[int]], retriever:BaseRetriever, gen_field_replace_token:str, max_seq_len:Optional[int]=None, ice_template:Optional[PromptTemplate]=None, prompt_template:Optional[PromptTemplate]=None) -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\openicl\icl_inferencer\icl_sw_ce_loss_inferencer.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `SWCELossInferencer`
    Metodo: `__init__(self:Any, model:BaseModel, max_seq_len:Optional[int]=None, batch_size:Optional[int]=1, output_json_filepath:Optional[str]='./icl_inference_output', output_json_filename:Optional[str]='predictions', save_every:Optional[int]=1, block_size:Optional[int]=1900, stride:Optional[int]=512, **kwargs:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `inference(self:Any, retriever:BaseRetriever, ice_template:Optional[PromptTemplate]=None, prompt_template:Optional[PromptTemplate]=None, output_json_filepath:Optional[str]=None, output_json_filename:Optional[str]=None) -> List`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_encoding_from_retriever_indices(self:Any, ice_idx_list:List[List[int]], retriever:BaseRetriever, max_seq_len:Optional[int]=None, prompt_template:Optional[PromptTemplate]=None, dtype:str='auto') -> Tuple[List, List]`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `_get_cross_entropy(self:Any, logits:torch.Tensor, targets:torch.Tensor, attention_mask:torch.Tensor=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `SlidingWindowEvalDataset`
    Metodo: `__init__(self:Any, data:HFDataset, block_size:int=1900, stride:int=512) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_prepare(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__len__(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__getitem__(self:Any, item:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `total_chr_num(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `SWCELossInferencerOutputHandler`
    Metodo: `__init__(self:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `write_to_json(self:Any, save_dir:str, filename:str) -> Any`
    Descrizione: Serializza e salva output su disco.
    Metodo: `save_results(self:Any, loss:float, total_chr_num:int, idx:Union[str, int]) -> None`
    Descrizione: Serializza e salva output su disco.

- File: `vendor\llada\opencompass\opencompass\openicl\icl_inferencer\icl_tot_inferencer.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `ToTInferencer`
    Metodo: `__init__(self:Any, model:BaseModel, max_out_len:int, max_seq_len:Optional[int]=None, batch_size:Optional[int]=1, gen_field_replace_token:Optional[str]='', output_json_filepath:Optional[str]='./icl_inference_output', output_json_filename:Optional[str]='predictions', save_every:Optional[int]=1, naive_run:bool=False, prompt_wrapper:dict={}, prompt_sample:str='standard', method_generate:str='sample', method_evaluate:str='value', method_select:str='greedy', n_generate_sample:int=1, n_evaluate_sample:int=1, n_select_sample:int=1, generation_kwargs:dict={}, **kwargs:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_value(self:Any, x:str, y:str, n_evaluate_sample:int, cache_value:bool=True) -> str`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_values(self:Any, x:str, ys:List[str], n_evaluate_sample:int, cache_value:bool=True) -> List[str]`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_votes(self:Any, x:str, ys:List[str], n_evaluate_sample:int) -> List[str]`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_proposals(self:Any, x:str, y:str) -> List[str]`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_samples(self:Any, x:str, y:str, n_generate_sample:int, prompt_sample:str) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `tot_solve(self:Any, x:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `inference(self:Any, retriever:BaseRetriever, ice_template:Optional[PromptTemplate]=None, prompt_template:Optional[PromptTemplate]=None, output_json_filepath:Optional[str]=None, output_json_filename:Optional[str]=None) -> List`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\openicl\icl_prompt_template.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `PromptTemplate`
    Metodo: `__init__(self:Any, template:Union[Dict, str], ice_token:Optional[str]=None, sep_token:Optional[str]=None) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_check_template_legacy(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate_ice_item(self:Any, entry:Dict, label:Hashable) -> PromptType`
    Descrizione: Genera output testo o sequenze.
    Metodo: `generate_label_prompt_item(self:Any, entry:Dict, ice:PromptType, label:Hashable, remain_sep:Optional[bool]=False) -> str`
    Descrizione: Genera output testo o sequenze.
    Metodo: `generate_item(self:Any, entry:Dict, output_field:Optional[Hashable]=None, output_field_replace_token:Optional[str]='', ice_field_replace_token:Optional[str]='') -> PromptType`
    Descrizione: Genera output testo o sequenze.
    Metodo: `_check_prompt_template(obj:Any) -> 'PromptTemplate'`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__repr__(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_encode_template(self:Any, prompt_template:Union[List[Union[str, Dict]], str], ice:bool) -> PromptType`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\openicl\icl_retriever\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\openicl\icl_retriever\icl_base_retriever.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `BaseRetriever`
    Metodo: `__init__(self:Any, dataset:Any, ice_separator:Optional[str]='\n', ice_eos_token:Optional[str]='\n', ice_num:Optional[int]=1) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `retrieve(self:Any) -> List[List[int]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_labels(self:Any, ice_template:Optional[PromptTemplate]=None, prompt_template:Optional[PromptTemplate]=None) -> List[str]`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `generate_ice(self:Any, idx_list:List[int], ice_template:Optional[PromptTemplate]=None) -> str`
    Descrizione: Genera output testo o sequenze.
    Metodo: `generate_label_prompt(self:Any, idx:int, ice:str, label:Any, ice_template:Optional[PromptTemplate]=None, prompt_template:Optional[PromptTemplate]=None, remain_sep:Optional[bool]=False) -> str`
    Descrizione: Genera output testo o sequenze.
    Metodo: `generate_prompt_for_generate_task(self:Any, idx:Any, ice:Any, gen_field_replace_token:Any='', ice_template:Optional[PromptTemplate]=None, prompt_template:Optional[PromptTemplate]=None) -> Any`
    Descrizione: Genera output testo o sequenze.
    Metodo: `generate_prompt_and_label_for_generate_task(self:Any, idx:Any, ice:Any, gen_field_replace_token:Any='', ice_template:Optional[PromptTemplate]=None, prompt_template:Optional[PromptTemplate]=None) -> Any`
    Descrizione: Genera output testo o sequenze.
    Metodo: `generate_prompt_for_adv_generate_task(self:Any, idx:Any, ice:Any, extra_prompt:Any=dict(), gen_field_replace_token:Any='', ice_template:Optional[PromptTemplate]=None, prompt_template:Optional[PromptTemplate]=None) -> Any`
    Descrizione: Genera output testo o sequenze.

- File: `vendor\llada\opencompass\opencompass\openicl\icl_retriever\icl_bm25_retriever.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `BM25Retriever`
    Metodo: `__init__(self:Any, dataset:Any, ice_separator:Optional[str]='\n', ice_eos_token:Optional[str]='\n', ice_num:Optional[int]=1) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `retrieve(self:Any) -> List[List]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\openicl\icl_retriever\icl_dpp_retriever.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `fast_map_dpp(kernel_matrix:Any, max_length:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `DPPRetriever`
    Metodo: `__init__(self:Any, dataset:Any, ice_separator:Optional[str]='\n', ice_eos_token:Optional[str]='\n', ice_num:Optional[int]=1, sentence_transformers_model_name:Optional[str]='all-mpnet-base-v2', tokenizer_name:Optional[str]='gpt2-xl', batch_size:Optional[int]=1, candidate_num:Optional[int]=1, seed:Optional[int]=1, scale_factor:Optional[float]=0.1) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `dpp_search(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `retrieve(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_kernel(self:Any, embed:Any, candidates:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\openicl\icl_retriever\icl_fix_k_retriever.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `FixKRetriever`
    Metodo: `__init__(self:Any, dataset:Any, fix_id_list:List[int], ice_separator:Optional[str]='\n', ice_eos_token:Optional[str]='\n', ice_num:Optional[int]=1) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `retrieve(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\openicl\icl_retriever\icl_mdl_retriever.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `entropy(probs:np.array, label_dim:int=0, mask:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `MDLRetriever`
    Metodo: `__init__(self:Any, dataset:Any, ice_separator:Optional[str]='\n', ice_eos_token:Optional[str]='\n', ice_num:Optional[int]=1, sentence_transformers_model_name:Optional[str]='all-mpnet-base-v2', tokenizer_name:Optional[str]='gpt2-xl', batch_size:Optional[int]=1, candidate_num:Optional[int]=1, ce_model_name:Optional[str]='gpt2-xl', select_time:Optional[int]=5, ice_template:Optional[PromptTemplate]=None, prompt_template:Optional[PromptTemplate]=None, labels:Optional[List]=None, seed:Optional[int]=1) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `topk_search(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `retrieve(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `cal_ce(self:Any, input_texts:List[str], mask_length:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\openicl\icl_retriever\icl_random_retriever.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `RandomRetriever`
    Metodo: `__init__(self:Any, dataset:Any, ice_separator:Optional[str]='\n', ice_eos_token:Optional[str]='\n', ice_num:Optional[int]=1, seed:Optional[int]=43) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `retrieve(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\openicl\icl_retriever\icl_sliding_k_retriever.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `SlidingWindowRetriever`
    Metodo: `__init__(self:Any, dataset:Any, k:int, ice_separator:Optional[str]='\n', ice_eos_token:Optional[str]='\n', ice_num:Optional[int]=1) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `retrieve(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\openicl\icl_retriever\icl_topk_retriever.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `ignore_pad_dict(features:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `TopkRetriever`
    Metodo: `__init__(self:Any, dataset:Any, ice_separator:Optional[str]='\n', ice_eos_token:Optional[str]='\n', ice_num:Optional[int]=1, sentence_transformers_model_name:Optional[str]='all-mpnet-base-v2', tokenizer_name:Optional[str]='gpt2-xl', batch_size:Optional[int]=1) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `create_index(self:Any) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
    Metodo: `knn_search(self:Any, ice_num:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, dataloader:Any, process_bar:Any=False, information:Any='') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `retrieve(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `ListWrapper`
    Metodo: `__init__(self:Any, data:List[Any]) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `to(self:Any, device:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `DataCollatorWithPaddingAndCuda`
    Metodo: `__call__(self:Any, features:List[Dict[str, Union[List[int], torch.Tensor]]]) -> BatchEncoding`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\openicl\icl_retriever\icl_votek_retriever.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `VotekRetriever`
    Metodo: `__init__(self:Any, dataset:Any, ice_separator:Optional[str]='\n', ice_eos_token:Optional[str]='\n', ice_num:Optional[int]=1, sentence_transformers_model_name:Optional[str]='all-mpnet-base-v2', tokenizer_name:Optional[str]='gpt2-xl', batch_size:Optional[int]=1, votek_k:Optional[int]=3) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `votek_select(self:Any, embeddings:Any=None, select_num:Any=None, k:Any=None, overlap_threshold:Any=None, vote_file:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `vote_k_search(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `retrieve(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\openicl\icl_retriever\icl_zero_retriever.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `ZeroRetriever`
    Metodo: `__init__(self:Any, dataset:Any, ice_eos_token:Optional[str]='') -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `retrieve(self:Any, id_list:List[int]=None) -> List[List]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\openicl\utils\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\openicl\utils\logging.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_logger(name:Any, level:Any=LOG_LEVEL, log_file:Any=None, file_mode:Any='w') -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\partitioners\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\partitioners\base.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `BasePartitioner`
    Metodo: `__init__(self:Any, out_dir:str, keep_keys:Optional[List[str]]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__call__(self:Any, cfg:ConfigDict) -> List[Dict]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `parse_model_dataset_args(self:Any, cfg:ConfigDict) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `partition(self:Any, models:List[ConfigDict], datasets:List[ConfigDict], work_dir:str, out_dir:str, add_cfg:Dict={}) -> List[Dict]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\partitioners\naive.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `NaivePartitioner`
    Metodo: `__init__(self:Any, out_dir:str, n:int=1, keep_keys:Optional[List[str]]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `partition(self:Any, model_dataset_combinations:List[Dict[str, List[ConfigDict]]], work_dir:str, out_dir:str, add_cfg:Dict={}) -> List[Dict]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\partitioners\num_worker.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `NumWorkerPartitioner`
    Metodo: `__init__(self:Any, out_dir:str, num_worker:int=8, num_split:Optional[int]=None, min_task_size:int=16, strategy:str='heuristic', dataset_size_path:str='.cache/dataset_size.json', keep_keys:Optional[List[str]]=None, force_rebuild:bool=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `partition(self:Any, model_dataset_combinations:List[Dict[str, List]], work_dir:str, out_dir:str, add_cfg:Dict={}) -> List[ConfigDict]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `dataset_size(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `split_dataset(self:Any, dataset_cfg:ConfigDict) -> List[ConfigDict]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_size(self:Any, dataset:ConfigDict) -> int`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\partitioners\size.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `SizePartitioner`
    Metodo: `__init__(self:Any, out_dir:str, max_task_size:int=40000, gen_task_coef:int=20, strategy:str='heuristic', dataset_size_path:str='.cache/dataset_size.json', keep_keys:Optional[List[str]]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `partition(self:Any, model_dataset_combinations:List[Dict[str, List[ConfigDict]]], work_dir:str, out_dir:str, add_cfg:Dict={}) -> List[ConfigDict]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `dataset_size(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `split_dataset(self:Any, dataset_cfg:ConfigDict) -> List[ConfigDict]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_factor(self:Any, dataset:ConfigDict) -> int`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_cost(self:Any, dataset:ConfigDict, get_raw_factors:bool=False) -> Union[int, Tuple[int, int]]`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\partitioners\sub_naive.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `remove_duplicate_pairs(model_combinations:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `replicate_tasks_with_judge_models(tasks:Any, judge_models:Any, meta_judge_model:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `remove_already_tasks(tasks:Any, work_dir:Any, meta_judge_model:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_model_combinations(mode:Any, models:List[ConfigDict], base_models:Optional[List[ConfigDict]]=[], compare_models:Optional[List[ConfigDict]]=[]) -> List`
    Descrizione: Recupera valore/stato calcolato.
  Classi:
  - `SubjectiveNaivePartitioner`
    Metodo: `__init__(self:Any, out_dir:str, models:Optional[List[ConfigDict]]=[], base_models:Optional[List[ConfigDict]]=[], compare_models:Optional[List[ConfigDict]]=[], judge_models:Optional[List[ConfigDict]]=[], meta_judge_model:Optional[ConfigDict]=None, model_pairs:Optional[List[Tuple]]=None, keep_keys:Optional[List[str]]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `partition(self:Any, models:List[ConfigDict], datasets:List[ConfigDict], work_dir:str, out_dir:str, add_cfg:Dict={}) -> List[Dict]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\partitioners\sub_num_worker.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `SubjectiveNumWorkerPartitioner`
    Metodo: `__init__(self:Any, out_dir:str, models:Optional[List[ConfigDict]]=[], base_models:Optional[List[ConfigDict]]=[], compare_models:Optional[List[ConfigDict]]=[], judge_models:Optional[List[ConfigDict]]=[], meta_judge_model:Optional[ConfigDict]=None, model_pairs:Optional[List[Tuple]]=None, num_worker:int=8, num_worker_split:Optional[int]=None, min_task_size:int=16, strategy:str='heuristic', dataset_size_path:str='.cache/dataset_size.json', keep_keys:Optional[List[str]]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `partition(self:Any, models:List[ConfigDict], datasets:List[ConfigDict], work_dir:str, out_dir:str, add_cfg:Dict={}) -> List[ConfigDict]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `dataset_size(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `split_dataset(self:Any, dataset_cfg:ConfigDict) -> List[ConfigDict]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_size(self:Any, dataset:ConfigDict) -> int`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\partitioners\sub_size.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `SubjectiveSizePartitioner`
    Metodo: `__init__(self:Any, out_dir:str, models:Optional[List[ConfigDict]]=[], base_models:Optional[List[ConfigDict]]=[], compare_models:Optional[List[ConfigDict]]=[], judge_models:Optional[List[ConfigDict]]=[], meta_judge_model:Optional[ConfigDict]=None, model_pairs:Optional[List[Tuple]]=None, max_task_size:int=40000, gen_task_coef:int=20, strategy:str='heuristic', dataset_size_path:str='.cache/dataset_size.json', keep_keys:Optional[List[str]]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `partition(self:Any, models:List[ConfigDict], datasets:List[ConfigDict], work_dir:str, out_dir:str, add_cfg:Dict={}) -> List[ConfigDict]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `dataset_size(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `split_dataset(self:Any, dataset_cfg:ConfigDict) -> List[ConfigDict]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_factor(self:Any, dataset:ConfigDict) -> int`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_cost(self:Any, dataset:ConfigDict, get_raw_factors:bool=False) -> Union[int, Tuple[int, int]]`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\registry.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `build_from_cfg(cfg:Any) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
  Classi:
  - `Registry`
    Metodo: `register_module(self:Any, name:Optional[Union[str, List[str]]]=None, force:bool=True, module:Optional[Type]=None) -> Union[type, Callable]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\runners\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\runners\base.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `BaseRunner`
    Metodo: `__init__(self:Any, task:ConfigDict, debug:bool=False, lark_bot_url:str=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__call__(self:Any, tasks:List[Dict[str, Any]]) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `launch(self:Any, tasks:List[Dict[str, Any]]) -> List[Tuple[str, int]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `summarize(self:Any, status:List[Tuple[str, int]]) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\runners\dlc.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `DLCRunner`
    Metodo: `__init__(self:Any, task:ConfigDict, aliyun_cfg:ConfigDict, max_num_workers:int=32, eval_with_gpu:list=['plugin_eval'], retry:int=2, debug:bool=False, lark_bot_url:str=None, keep_tmp_file:bool=True, preemptible:bool=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `launch(self:Any, tasks:List[Dict[str, Any]]) -> List[Tuple[str, int]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_launch(self:Any, cfg:ConfigDict, random_sleep:Optional[bool]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_job_failed(self:Any, return_code:int, output_paths:List[str]) -> bool`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\runners\local.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_command_template(gpu_ids:List[int]) -> str`
    Descrizione: Recupera valore/stato calcolato.
  Classi:
  - `LocalRunner`
    Metodo: `__init__(self:Any, task:ConfigDict, max_num_workers:int=16, debug:bool=False, max_workers_per_gpu:int=1, lark_bot_url:str=None, keep_tmp_file:bool=False, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `launch(self:Any, tasks:List[Dict[str, Any]]) -> List[Tuple[str, int]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_launch(self:Any, task:Any, gpu_ids:Any, index:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\runners\local_api.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `monkey_run(self:Any, tokens:SyncManager.Semaphore) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `redirect_std_to_file(filename:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `reset_std() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `launch(task:BaseTask, tokens:SyncManager.Semaphore) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `submit(task:Any, type:Any, tokens:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `LocalAPIRunner`
    Metodo: `__init__(self:Any, task:ConfigDict, concurrent_users:int, max_num_workers:int=16, debug:bool=False, lark_bot_url:str=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `launch(self:Any, tasks:List[Dict[str, Any]]) -> List[Tuple[str, int]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\runners\rjob.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `RJOBRunner`
    Metodo: `__init__(self:Any, task:ConfigDict, rjob_cfg:ConfigDict, max_num_workers:int=32, retry:int=100, debug:bool=False, lark_bot_url:str=None, keep_tmp_file:bool=True, phase:str='unknown') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `launch(self:Any, tasks:List[Dict[str, Any]]) -> List[Tuple[str, int]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_run_task(self:Any, task_name:Any, log_path:Any, poll_interval:Any=60) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_launch(self:Any, cfg:ConfigDict, random_sleep:Optional[bool]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_job_failed(self:Any, return_code:int, output_paths:List[str]) -> bool`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\runners\slurm.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `SlurmRunner`
    Metodo: `__init__(self:Any, task:ConfigDict, max_num_workers:int=32, retry:int=2, partition:str=None, quotatype:str=None, qos:str=None, debug:bool=False, lark_bot_url:str=None, extra_command:Optional[List[str]]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `launch(self:Any, tasks:List[Dict[str, Any]]) -> List[Tuple[str, int]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_launch(self:Any, cfg:ConfigDict, random_sleep:bool=True) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_job_failed(self:Any, return_code:int, output_paths:List[str]) -> bool`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\runners\slurm_sequential.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `SlurmSequentialRunner`
    Metodo: `__init__(self:Any, task:ConfigDict, task_prefix:str='', max_num_workers:int=32, retry:int=2, partition:str=None, quotatype:str=None, qos:str=None, debug:bool=False, lark_bot_url:str=None, extra_command:Optional[List[str]]=None, keep_tmp_file:bool=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `launch(self:Any, tasks:List[Dict[str, Any]]) -> List[Tuple[str, int]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_launch_wo_debug(self:Any, tasks:List[Dict[str, Any]]) -> List[Tuple[str, int]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_launch(self:Any, cfg:ConfigDict, child_conn:Pipe=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_job_failed(self:Any, return_code:int, output_paths:List[str]) -> bool`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\runners\volc.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `VOLCRunner`
    Metodo: `__init__(self:Any, task:ConfigDict, volcano_cfg:ConfigDict, queue_name:str, preemptible:bool=False, priority:Optional[int]=None, max_num_workers:int=32, retry:int=2, debug:bool=False, lark_bot_url:str=None, keep_tmp_file:bool=True) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `launch(self:Any, tasks:List[Dict[str, Any]]) -> List[Tuple[str, int]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_launch(self:Any, task_cfg:ConfigDict, random_sleep:bool=True) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_run_task(self:Any, cmd:Any, log_path:Any, poll_interval:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_job_failed(self:Any, task_status:str, output_paths:List[str]) -> bool`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_choose_flavor(self:Any, num_gpus:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\summarizers\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\summarizers\circular.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `CircularSummarizer`
    Metodo: `__init__(self:Any, config:ConfigDict, dataset_abbrs:Optional[List[str]]=None, summary_groups:List=[], prompt_db:Any=None, metric_types:Any=None) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_format_table(self:Any, parsed_results:Any, dataset_metrics:Any, dataset_eval_mode:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\summarizers\default.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `model_abbr_from_cfg_used_in_summarizer(model:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `DefaultSummarizer`
    Metodo: `__init__(self:Any, config:ConfigDict, dataset_abbrs:Optional[List[str]]=None, summary_groups:List=[], prompt_db:Any=None) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_pick_up_results(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_calculate_group_metrics(self:Any, raw_results:Any, parsed_results:Any, dataset_metrics:Any, dataset_eval_mode:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_format_table(self:Any, parsed_results:Any, dataset_metrics:Any, dataset_eval_mode:Any, required_dataset_abbrs:Any=None, skip_all_slash:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_format_raw_txt(self:Any, raw_results:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_format_md_table(table:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_output_to_file(self:Any, output_path:Any, time_str:Any, table:Any, raw_txts:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `summarize(self:Any, output_path:str=None, time_str:str=datetime.now().strftime('%Y%m%d_%H%M%S')) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\summarizers\default_subjective.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `model_abbr_from_cfg_used_in_summarizer(model:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `DefaultSubjectiveSummarizer`
    Metodo: `__init__(self:Any, config:ConfigDict, dataset_abbrs:Optional[List[str]]=None, summary_groups:List=[], prompt_db:Any=None) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_pick_up_results(self:Any, judge_abbr:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_calculate_group_metrics(self:Any, raw_results:Any, parsed_results:Any, dataset_metrics:Any, dataset_eval_mode:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_format_table(self:Any, parsed_results:Any, dataset_metrics:Any, dataset_eval_mode:Any, required_dataset_abbrs:Any=None, skip_all_slash:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_format_raw_txt(self:Any, raw_results:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_output_to_file(self:Any, output_path:Any, time_str:Any, table:Any, raw_txts:Any, judge_abbr:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `summarize(self:Any, output_path:str=None, time_str:str=datetime.now().strftime('%Y%m%d_%H%M%S')) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\summarizers\llm_compression.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LLMCompressionSummarizer`
    Metodo: `__init__(self:Any, config:ConfigDict, dataset_abbrs:Optional[List[str]]=None, summary_groups:List=None, prompt_db:Any=None) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_format_table(self:Any, parsed_results:Any, dataset_metrics:Any, dataset_eval_mode:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_format_table_pivot(self:Any, table:List[List], decimals:int=4) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_output_df_to_file(self:Any, output_path:str, timestamp:str, table:pd.DataFrame) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `summarize(self:Any, output_path:str=None, time_str:str=datetime.now().strftime('%Y%m%d_%H%M%S')) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\summarizers\multi_faceted.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `MultiFacetedSummarizer`
    Metodo: `__init__(self:Any, config:ConfigDict, dataset_abbrs_list:Optional[Dict[str, List[str]]]=None, summary_groups:List=[]) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `summarize(self:Any, output_path:str=None, time_str:str=datetime.now().strftime('%Y%m%d_%H%M%S')) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\summarizers\multi_model.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `bold(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `green_bold(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `format_float(v:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `to_float(text:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `is_section_row(row:List[str]) -> bool`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `average_rows(name:Any, rows:List[List[str]]) -> List[str]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `create_section_row(row_i:int, row:List[str], table:Any) -> List[str]`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
  - `create_win_row(rows:List[List[str]]) -> List[str]`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
  - `highlight(row:List[str], meta_col_count:int=META_COL_COUNT) -> List[str]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `MultiModelSummarizer`
    Metodo: `__init__(self:Any, config:ConfigDict, dataset_abbrs:Optional[List[str]]=None, summary_groups:List=[], prompt_db:Any=None) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `load(self:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
    Metodo: `merge(self:Any, summarizer:'MultiModelSummarizer') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `summarize(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `show_group(self:Any, group:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\summarizers\needlebench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `calculate_elementwise_average(model_name:Any, merged_df:Any, mean:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `read_after_specific_line_except_last(file_name:Any, keyword:Any, offset:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `create_model_dataframe(nested_dict:Any, model_name:Any, dataset_abbr:Any, parallel:Any=False) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
  - `convert_to_k(value:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `parse_model_scores(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `remove_empty_subfolders(plot_path:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `save_results_to_plots(txt_results_save_path:Any, mean:Any=False) -> Any`
    Descrizione: Serializza e salva output su disco.
  - `visualize(df_raw:Any, save_path:str, model_name:str, dataset_type:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `ensure_directory(path:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_dict_model_names(nested_dict:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `merge_dataframes(model_name:Any, dataset_abbrs:Any, parsed_data:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `NeedleBenchSummarizer`
    Metodo: `_format_table(self:Any, parsed_results:Any, dataset_metrics:Any, dataset_eval_mode:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_format_raw_txt(self:Any, raw_results:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_output_to_file(self:Any, output_path:Any, time_str:Any, table:Any, raw_txts:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `summarize(self:Any, output_path:str=None, time_str:str=datetime.now().strftime('%Y%m%d_%H%M%S')) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `NeedleBenchSummarizerV2`
    Metodo: `summarize(self:Any, output_path:str=None, time_str:str=datetime.now().strftime('%Y%m%d_%H%M%S')) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `NeedleBenchATCSummarizer`
    Metodo: `_format_table(self:Any, parsed_results:Any, dataset_metrics:Any, dataset_eval_mode:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_read_and_sort_dataframe(self:Any, file_path:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_output_to_file(self:Any, output_path:Any, time_str:Any, table:Any, raw_txts:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `summarize(self:Any, output_path:str=None, time_str:str=datetime.now().strftime('%Y%m%d_%H%M%S')) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\summarizers\subjective\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\summarizers\subjective\alignmentbench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `detect_mapping(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `extract_missing_rating(text:Any, search_type:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `extract_rating_plus(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `extract_rating(text:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `check_rating(rating:Any, all_dimensions:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `post_process_alignbench_plus(judgement:str, all_dimensions:Any=All_Dimensions, possible_keys:Any=['综合得分']) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `post_process_alignbench(judgement:str, all_dimensions:Any=All_Dimensions, possible_keys:Any=['综合得分']) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_dimension_results(judged_answers:Any, references:Any, fout:Any, fout_flag:Any, model:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `get_capability_results(judged_answers:Any, references:Any, fout:Any, fout_flag:Any, model:Any, categories:Any=CATEGORIES) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  Classi:
  - `AlignmentBenchSummarizer`
    Metodo: `__init__(self:Any, config:ConfigDict, judge_type:Any='general') -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `summarize(self:Any, time_str:str=datetime.now().strftime('%Y%m%d_%H%M%S')) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\summarizers\subjective\all_obj.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `post_process_allobj(judgement:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_capability_results(judged_answers:Any, references:Any, fout:Any, fout_flag:Any, model:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  Classi:
  - `AllObjSummarizer`
    Metodo: `__init__(self:Any, config:ConfigDict, judge_type:Any='single') -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `summarize(self:Any, time_str:str=datetime.now().strftime('%Y%m%d_%H%M%S')) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\summarizers\subjective\alpacaeval.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `post_process_alpacav1(completion:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `post_process_alpacav2(completion:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `AlpacaSummarizer`
    Metodo: `__init__(self:Any, config:ConfigDict, judge_type:Any='v2') -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `summarize(self:Any, time_str:str=datetime.now().strftime('%Y%m%d_%H%M%S')) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\summarizers\subjective\arenahard.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `compute_mle_elo(df:Any, SCALE:Any=400, BASE:Any=10, INIT_RATING:Any=1000) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_bootstrap_result(battles:Any, func_compute_elo:Any, num_round:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `preety_print_two_ratings(ratings_1:Any, ratings_2:Any, column_names:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `visualize_bootstrap_scores(df:Any, title:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `predict_win_rate(elo_ratings:Any, SCALE:Any=400, BASE:Any=10, INIT_RATING:Any=1000) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `model_abbr_from_cfg_used_in_summarizer(model:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `post_process_compass_arena(s:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_win_rate_column(df:Any, column:Any, baseline:Any='gpt4-0314') -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `load_model_preds(filename:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `get_battles_from_judgment(dataset:Any, subdir_path:Any, post_process:Any, WEIGHT:Any=3) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  Classi:
  - `ArenaHardSummarizer`
    Metodo: `__init__(self:Any, config:ConfigDict, judge_type:Any='general', check_pos_bias:Any=True, summary_type:Any='single') -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_score(self:Any, time_str:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `summarize(self:Any, time_str:str=datetime.now().strftime('%Y%m%d_%H%M%S')) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\summarizers\subjective\charm.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `post_process_charm_mem(judgement:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_judgeanswer_and_reference_charm_mem(dataset:Any, subdir_path:Any, post_process:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `get_accuracy(judged_answers:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  Classi:
  - `CharmMemSummarizer`
    Metodo: `__init__(self:Any, config:ConfigDict, judge_type:Any='single') -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `summarize(self:Any, time_str:str=datetime.now().strftime('%Y%m%d_%H%M%S')) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\summarizers\subjective\common_summarizer.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `model_abbr_from_cfg_used_in_summarizer(model:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `post_process_single_rate(judgement:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_capability_results(judged_answers:Any, references:Any, fout:Any, fout_flag:Any, model_abbr:Any, judge_model_abbr:Any, dataset_abbr:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  Classi:
  - `CommonSummarizer`
    Metodo: `__init__(self:Any, config:ConfigDict, judge_type:Any='single_rate') -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `summarize(self:Any, time_str:str=datetime.now().strftime('%Y%m%d_%H%M%S')) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\summarizers\subjective\compass_arena.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `model_abbr_from_cfg_used_in_summarizer(model:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `post_process_compass_arena(s:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `check_position_bias(judged_answers:Any, references:Any, banned_choice:Any=['C']) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `CompassArenaSummarizer`
    Metodo: `__init__(self:Any, config:ConfigDict, judge_type:Any='general', check_pos_bias:Any=True, summary_type:Any='single') -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_score(self:Any, time_str:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `summarize(self:Any, time_str:str=datetime.now().strftime('%Y%m%d_%H%M%S')) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\summarizers\subjective\compass_arena_bradley_terry.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_matchups_models(df:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `preprocess_for_elo(df:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `preprocess_for_bt(df:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `preprocess_for_style(df:Any, apply_ratio:List[int]=None, style_variables:List[str]=STYLE_CONTROL_VARIABLES_V1, control_variables:List[str]=EXTRA_CONTROL_VARIABLES, style_var_suffixes:List[str]=None, add_one:bool=True, normalize_style_features:bool=True) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `fit_vectorized_elo(matchups:Any, outcomes:Any, sample_indices:Any, num_models:int, k:float=4.0, base:float=10.0, init_rating:float=1000.0, scale:float=400.0) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `compute_elo(df:Any, k:float=4.0, base:float=10.0, init_rating:float=1000.0, scale:float=400.0) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `compute_bootstrap_elo(df:Any, num_round:int=100, k:float=4.0, base:float=10.0, init_rating:float=1000.0, scale:float=400.0) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `bt_loss_and_grad(ratings:Any, matchups:Any, outcomes:Any, weights:Any, alpha:Any=1.0) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `fit_bt(matchups:Any, outcomes:Any, weights:Any, n_models:Any, alpha:Any, tol:Any=1e-06) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `scale_and_offset(ratings:Any, models:Any, scale:float=400.0, init_rating:float=1000.0, baseline_model:str=None, baseline_rating:float=1000.0) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `compute_bt(df:Any, base:float=10.0, scale:float=400.0, init_rating:float=1000.0, baseline_model:str=None, baseline_rating:float=1000.0, tol:float=1e-06) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `compute_bootstrap_bt(battles:Any, num_round:int, base:float=10.0, scale:float=400.0, init_rating:float=1000.0, baseline_model:str=None, baseline_rating:float=1000.0, tol:float=1e-06, num_cpu:int=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `contextual_bt_loss_and_grad(params:Any, n_competitors:Any, matchups:Any, features:Any, outcomes:Any, alpha:Any=1.0, reg:Any=1.0, half_reg:Any=0.5) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `fit_contextual_bt(matchups:Any, features:Any, outcomes:Any, models:Any, idxs:Any=None, alpha:Any=math.log(10.0), reg:Any=0.5, tol:Any=1e-06) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `compute_style_control(df:pd.DataFrame, alpha:float=math.log(10.0), reg:float=0.5, scale:float=400.0, init_rating:float=1000.0, baseline_model:str=None, baseline_rating:float=1000.0, normalize_style_features:bool=True, control_variables:List[str]=None, odds_ratio:bool=True, tol:float=1e-06) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `compute_bootstrap_style_control(df:Any, num_round:int, alpha:float=math.log(10.0), reg:float=0.5, scale:float=400.0, init_rating:float=1000.0, baseline_model:str=None, baseline_rating:float=1000.0, normalize_style_features:bool=True, control_variables:List[str]=None, odds_ratio:bool=True, tol:float=1e-06, num_cpu:int=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `CompassArenaBradleyTerrySummarizer`
    Metodo: `__init__(self:Any, config:ConfigDict, dataset_abbrs:Optional[List[str]]=None, summary_groups:List=None, prompt_db:Any=None, rating_system:str='bradleyterry', report_pred_win_rates:bool=True, num_bootstrap:int=300, num_cpu:int=None, with_control_vars:bool=True, normalize_style_features:bool=True, odds_ratio:bool=True, groups:List[str]=None) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_pick_up_results(self:Any, judge_abbr:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_calculate_ratings(self:Any, matches:Dict, base_model:str=None, groups:List[str]=None) -> Tuple[pd.DataFrame, Dict]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_output_to_file(self:Any, output_path:Any, time_str:str, tables:Dict, metadata:Dict, judge_abbr:str, dataset_eval_mode:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `flip_dict_levels(self:Any, original_dict:Dict) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `predict_win_rate(self:Any, ratings_df:pd.DataFrame, baseline_model:str, base:float=10.0, scaling_factor:float=400.0, round_win_rate:int=None) -> pd.DataFrame`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `summarize(self:Any, output_path:str=None, time_str:str=datetime.now().strftime('%Y%m%d_%H%M%S')) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\summarizers\subjective\compassbench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `post_process_wildbench_pair(judgement:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `CompassBenchSummarizer`
    Metodo: `__init__(self:Any, config:ConfigDict, check_pos_bias:Any=False) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_score(self:Any, time_str:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `summarize(self:Any, time_str:str=datetime.now().strftime('%Y%m%d_%H%M%S')) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\summarizers\subjective\compassbench_v13.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `post_process_wildbench_pair(judgement:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `CompassBenchSummarizer`
    Metodo: `__init__(self:Any, config:ConfigDict, check_pos_bias:Any=False) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_score(self:Any, time_str:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `summarize(self:Any, time_str:str=datetime.now().strftime('%Y%m%d_%H%M%S')) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\summarizers\subjective\corev2.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `match_general_answer(s:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `match_GPT4_answer(s:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `call_function(name:Any, arg:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `Corev2Summarizer`
    Metodo: `__init__(self:Any, config:ConfigDict, match_method:Any='smart') -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `summarize(self:Any, time_str:str=datetime.now().strftime('%Y%m%d_%H%M%S')) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\summarizers\subjective\creationbench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `post_process_creationbench(judgement:str, all_dimensions:Any=All_Dimensions, possible_keys:Any=['综合得分', 'Overall Score']) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `CreationBenchSummarizer`
    Metodo: `__init__(self:Any, config:ConfigDict, judge_type:str) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `summarize(self:Any, time_str:str=datetime.now().strftime('%Y%m%d_%H%M%S')) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\summarizers\subjective\flames.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `post_process_flames(judgement:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `FlamesSummarizer`
    Metodo: `__init__(self:Any, config:ConfigDict, judge_type:Any='general') -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `summarize(self:Any, time_str:str=datetime.now().strftime('%Y%m%d_%H%M%S')) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\summarizers\subjective\fofo.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `post_process_fofo(judgement:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `FofoSummarizer`
    Metodo: `__init__(self:Any, config:ConfigDict, judge_type:Any='single') -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_score(self:Any, time_str:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `summarize(self:Any, time_str:str=datetime.now().strftime('%Y%m%d_%H%M%S')) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\summarizers\subjective\followbench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `post_process_followbench(item:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_scores(judged_answers:Any, references:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  Classi:
  - `FollowBenchSummarizer`
    Metodo: `__init__(self:Any, config:ConfigDict) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `summarize(self:Any, time_str:str=datetime.now().strftime('%Y%m%d_%H%M%S')) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\summarizers\subjective\mtbench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `model_abbr_from_cfg_used_in_summarizer(model:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `post_process_mtbench_pair(judgement:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `post_process_mtbench_single(judgement:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_capability_results(judged_answers:Any, references:Any, fout:Any, fout_flag:Any, model_abbr:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  Classi:
  - `MTBenchSummarizer`
    Metodo: `__init__(self:Any, config:ConfigDict, judge_type:Any='single') -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `summarize(self:Any, time_str:str=datetime.now().strftime('%Y%m%d_%H%M%S')) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\summarizers\subjective\mtbench101.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `post_process_mtbench_pair(judgement:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `post_process_mtbench101(judgement:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_final_results(judged_answers:Any, references:Any, output_dir:Any, fout_flag:Any, model:Any, judgemodel:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  Classi:
  - `MTBench101Summarizer`
    Metodo: `__init__(self:Any, config:ConfigDict, judge_type:Any='single') -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `summarize(self:Any, time_str:str=datetime.now().strftime('%Y%m%d_%H%M%S')) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\summarizers\subjective\multiround.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `post_process_multiround(judgement:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_capability_results(judged_answers:Any, references:Any, fout:Any, fout_flag:Any, model:Any, categories:Any=CATEGORIES) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  Classi:
  - `MultiroundSummarizer`
    Metodo: `__init__(self:Any, config:ConfigDict) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `summarize(self:Any, time_str:str=datetime.now().strftime('%Y%m%d_%H%M%S')) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\summarizers\subjective\qacompassbench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `post_process_wildbench_pair(judgement:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `QaCompassBenchSummarizer`
    Metodo: `__init__(self:Any, config:ConfigDict, check_pos_bias:Any=False) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_score(self:Any, time_str:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `summarize(self:Any, time_str:str=datetime.now().strftime('%Y%m%d_%H%M%S')) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\summarizers\subjective\subjective.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `flatten_data(data:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `SubjectiveSummarizer`
    Metodo: `__init__(self:Any, config:ConfigDict, function:str) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `summarize(self:Any, subjective_scores:list, time_str:str=datetime.now().strftime('%Y%m%d_%H%M%S')) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\summarizers\subjective\subjective_post_process.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `post_process_autoj(judgement:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `post_process_judgelm(judgement:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\summarizers\subjective\utils.py`
  Logica d'uso: Runtime helpers, device/dtype control, tiny fallback model/tokenizer.
  Funzioni:
  - `get_outdir(cfg:Any, time_str:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `get_judgeanswer_and_reference(dataset:Any, subdir_path:Any, post_process:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `get_judgeanswer_and_reference_update(dataset:Any, subdir_path:Any, post_process:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\summarizers\subjective\wildbench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `post_process_wildbench_pair(judgement:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `post_process_wildbench_single(judgement:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_capability_results(judged_answers:Any, references:Any, fout:Any, fout_flag:Any, model_abbr:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  Classi:
  - `WildBenchSingleSummarizer`
    Metodo: `__init__(self:Any, config:ConfigDict) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `summarize(self:Any, time_str:str=datetime.now().strftime('%Y%m%d_%H%M%S')) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `WildBenchPairSummarizer`
    Metodo: `__init__(self:Any, config:ConfigDict, check_pos_bias:Any=False) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_score(self:Any, time_str:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `summarize(self:Any, time_str:str=datetime.now().strftime('%Y%m%d_%H%M%S')) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\summarizers\summarizer_pretrain.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `PretrainSummarizer`
    Metodo: `__init__(self:Any, config:ConfigDict, dataset_abbrs:Optional[List[str]]=None, summary_groups:List=[], prompt_db:Any=None) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `summarize(self:Any, output_path:str=None, time_str:str=datetime.now().strftime('%Y%m%d_%H%M%S')) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\tasks\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\tasks\base.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `extract_role_pred(s:str, begin_str:Optional[str], end_str:Optional[str]) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `BaseTask`
    Metodo: `__init__(self:Any, cfg:ConfigDict) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `run(self:Any) -> Any`
    Descrizione: Esegue pipeline o job completo.
    Metodo: `get_command(self:Any, cfg_path:Any, template:Any) -> str`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `name(self:Any) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__repr__(self:Any) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_log_path(self:Any, file_extension:str='json') -> str`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_output_paths(self:Any, file_extension:str='json') -> List[str]`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\tasks\llm_eval.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `ModelEvaluator`
    Metodo: `__init__(self:Any, config:ConfigDict) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `parse_cfg(self:Any, cfg:ConfigDict) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `evaluate(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_load_dataset(self:Any, dataset_abbr:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_evaluate_dataset(self:Any, dataset_abbr:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_make_prompt(self:Any, question:str, responses:List[str]) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_rank_models(self:Any, output:str, model_scores:defaultdict) -> Dict[str, int]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\tasks\openicl_attack.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `parse_args() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `OpenICLAttackTask`
    Metodo: `__init__(self:Any, cfg:ConfigDict) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_command(self:Any, cfg_path:Any, template:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `prompt_selection(self:Any, inferencer:Any, prompts:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `run(self:Any) -> Any`
    Descrizione: Esegue pipeline o job completo.
    Metodo: `_inference(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_set_default_value(self:Any, cfg:ConfigDict, key:str, value:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\tasks\openicl_eval.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `parse_args() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `OpenICLEvalTask`
    Metodo: `__init__(self:Any, cfg:ConfigDict) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_command(self:Any, cfg_path:Any, template:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `run(self:Any) -> Any`
    Descrizione: Esegue pipeline o job completo.
    Metodo: `_score(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_load_and_preprocess_test_data(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_load_predictions(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_process_predictions(self:Any, pred_strs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_evaluate_predictions(self:Any, pred_strs:Any, test_set:Any, pred_dicts:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_save_results(self:Any, result:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `extract_rate(self:Any, results:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `format_details(self:Any, predictions:Any, references:Any, details:Any, pred_dicts:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `calculate_bpb(self:Any, pred_dicts:List) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\tasks\openicl_infer.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `parse_args() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `OpenICLInferTask`
    Metodo: `__init__(self:Any, cfg:ConfigDict) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_command(self:Any, cfg_path:Any, template:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `run(self:Any, cur_model:Any=None, cur_model_abbr:Any=None) -> Any`
    Descrizione: Esegue pipeline o job completo.
    Metodo: `_inference(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_set_default_value(self:Any, cfg:ConfigDict, key:str, value:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\tasks\outer_eval\alpacaeval.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `PredictionMerger`
    Metodo: `__init__(self:Any, cfg:ConfigDict) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `run(self:Any) -> Any`
    Descrizione: Esegue pipeline o job completo.
  - `AlpacaEvalTask`
    Metodo: `__init__(self:Any, cfg:ConfigDict) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_command(self:Any, cfg_path:Any, template:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `run(self:Any) -> Any`
    Descrizione: Esegue pipeline o job completo.

- File: `vendor\llada\opencompass\opencompass\tasks\subjective_eval.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `parse_args() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `SubjectiveEvalTask`
    Metodo: `__init__(self:Any, cfg:ConfigDict) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_command(self:Any, cfg_path:Any, template:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `run(self:Any) -> Any`
    Descrizione: Esegue pipeline o job completo.
    Metodo: `name(self:Any) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_load_model_pred(self:Any, model_cfg:Union[ConfigDict, List[ConfigDict]], dataset_cfg:ConfigDict, eval_cfg:ConfigDict, given_preds:List[dict]) -> Union[None, List[str]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_load_model_judgements(self:Any, model_cfg:Union[ConfigDict, List[ConfigDict]], dataset_cfg:ConfigDict, eval_cfg:ConfigDict, judge_cfg:Union[ConfigDict, List[ConfigDict]]) -> Union[None, List[str]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_score(self:Any, model_cfg:Any, dataset_cfg:Any, eval_cfg:Any, output_column:Any, meta:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_output_paths(self:Any, file_extension:str='json') -> List[str]`
    Descrizione: Recupera valore/stato calcolato.

- File: `vendor\llada\opencompass\opencompass\utils\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\utils\abbr.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `model_abbr_from_cfg(cfg:Union[ConfigDict, List[ConfigDict]]) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `dataset_abbr_from_cfg(cfg:ConfigDict) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `task_abbr_from_cfg(task:Dict) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_infer_output_path(model_cfg:ConfigDict, dataset_cfg:ConfigDict, root_path:str=None, file_extension:str='json') -> str`
    Descrizione: Recupera valore/stato calcolato.
  - `deal_with_judge_model_abbr(model_cfg:Any, judge_model_cfg:Any, meta:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\utils\auxiliary.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\utils\build.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `build_dataset_from_cfg(dataset_cfg:ConfigDict) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
  - `build_model_from_cfg(model_cfg:ConfigDict) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.

- File: `vendor\llada\opencompass\opencompass\utils\collect_env.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `collect_env() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\utils\datasets.py`
  Logica d'uso: Builds CPT/SFT datasets from local/raw sources with fallback-safe behavior.
  Funzioni:
  - `get_data_path(dataset_id:str, local_mode:bool=False) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `download_dataset(data_path:Any, cache_dir:Any, remove_finished:Any=True) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\utils\datasets_info.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\opencompass\utils\dependency.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `satisfy_requirement(dep:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\utils\dict_postprocessors.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `base_postprocess(output:dict) -> dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\utils\file.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `match_files(path:str, pattern:Union[str, List], fuzzy:bool=False) -> List[Tuple[str, str]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\utils\fileio.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `patch_func(module:Any, fn_name_to_wrap:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `patch_fileio(global_vars:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `patch_hf_auto_model(cache_dir:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `calculate_md5(fpath:str, chunk_size:int=1024 * 1024) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `check_md5(fpath:Any, md5:Any, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `check_integrity(fpath:Any, md5:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `download_url_to_file(url:Any, dst:Any, hash_prefix:Any=None, progress:Any=True) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `download_url(url:Any, root:Any, filename:Any=None, md5:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_is_tarxz(filename:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_is_tar(filename:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_is_targz(filename:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_is_tgz(filename:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_is_gzip(filename:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_is_zip(filename:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `extract_archive(from_path:Any, to_path:Any=None, remove_finished:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `download_and_extract_archive(url:Any, download_root:Any, extract_root:Any=None, filename:Any=None, md5:Any=None, remove_finished:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `JSONToolkit`
    Metodo: `read_json(file_path:Union[str, Path]) -> Dict[str, Any]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `read_jsonl(file_path:Union[str, Path]) -> List[Dict[str, Any]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `save_json(data:Dict[str, Any], file_path:Union[str, Path], indent:Optional[int]=2) -> None`
    Descrizione: Serializza e salva output su disco.
    Metodo: `save_jsonl(data:List[Dict[str, Any]], file_path:Union[str, Path]) -> None`
    Descrizione: Serializza e salva output su disco.
    Metodo: `jsonl_writer(file_path:Union[str, Path]) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\utils\lark.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `parse_args() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `LarkReporter`
    Metodo: `__init__(self:Any, url:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `post(self:Any, content:Union[str, List[List[Dict]]], title:Optional[str]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\utils\logging.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_logger(log_level:Any='INFO', filter_duplicate_level:Any=None) -> MMLogger`
    Descrizione: Recupera valore/stato calcolato.
  - `pretty_print_config(cfg:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `FilterDuplicateMessage`
    Metodo: `__init__(self:Any, name:Any, filter_duplicate_level:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `filter(self:Any, record:logging.LogRecord) -> bool`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\utils\menu.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `Menu`
    Metodo: `__init__(self:Any, lists:Any, prompts:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `draw_menu(self:Any, stdscr:Any, selected_row_idx:Any, offset:Any, max_rows:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `run(self:Any) -> Any`
    Descrizione: Esegue pipeline o job completo.
    Metodo: `main_loop(self:Any, stdscr:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\utils\network.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `setup_proxies(proxy_env_name:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `check_network_connectivity(host:str='8.8.8.8', port:int=53, timeout:float=3, proxies:Optional[Dict[str, str]]=None) -> Tuple[bool, str]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `check_url_accessibility(url:str, timeout:float=3, proxies:Optional[Dict[str, str]]=None) -> Tuple[bool, Optional[int]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\utils\prompt.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `safe_format(input_str:str, **kwargs:Any) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_prompt_hash(dataset_cfg:Union[ConfigDict, List[ConfigDict]]) -> str`
    Descrizione: Recupera valore/stato calcolato.
  Classi:
  - `PromptList`
    Metodo: `format(self:Any, **kwargs:Any) -> PromptList`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `replace(self:Any, src:str, dst:Union[str, PromptList]) -> PromptList`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__add__(self:Any, other:Union[str, PromptList]) -> PromptList`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__radd__(self:Any, other:Union[str, PromptList]) -> PromptList`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__iadd__(self:Any, other:Union[str, PromptList]) -> PromptList`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__str__(self:Any) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\utils\result_station.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `save_to_station(cfg:Any, args:Any) -> Any`
    Descrizione: Serializza e salva output su disco.
  - `read_from_station(cfg:Any, args:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `find_files_by_regex(directory:Any, pattern:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `check_filenames(x:Any, filenames:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\utils\run.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `match_cfg_file(workdir:Union[str, List[str]], pattern:Union[str, List[str]]) -> List[Tuple[str, str]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `try_fill_in_custom_cfgs(config:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_config_from_arg(args:Any) -> Config`
    Descrizione: Recupera valore/stato calcolato.
  - `change_accelerator(models:Any, accelerator:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_config_type(obj:Any) -> str`
    Descrizione: Recupera valore/stato calcolato.
  - `fill_infer_cfg(cfg:Any, args:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `fill_eval_cfg(cfg:Any, args:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\utils\text_postprocessors.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `general_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `general_cn_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `first_capital_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `last_capital_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `think_pred_postprocess(prediction:str, re_pattern:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `first_option_postprocess(text:str, options:str, cushion:Any=True) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `first_capital_postprocess_multi(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `last_option_postprocess(text:str, options:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `first_number_postprocess(text:str) -> float`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `multiple_select_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `xml_tag_postprocessor(text:Any, tag:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `general_eval_wrapper_postprocess(text:str, postprocess:Optional[Union[str, Callable]]=None, **kwargs:Any) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `match_answer_pattern(response_text:str, answer_pattern:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `extract_non_reasoning_content(text:str, think_start_token:str='<think>', think_end_token:str='</think>') -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\opencompass\utils\types.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_type_from_cfg(cfg:Union[Config, Dict]) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `_check_type_list(obj:Any, typelist:List) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_check_dataset(obj:Any) -> Union[Dataset, DatasetDict]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_check_list(obj:Any) -> List`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_check_str(obj:Any) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_check_dict(obj:Any) -> Dict`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\run.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\opencompass\setup.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `readme() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `parse_requirements(fname:Any='requirements.txt', with_version:Any=True) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_version() -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `do_setup() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `DownloadNLTK`
    Metodo: `run(self:Any) -> Any`
    Descrizione: Esegue pipeline o job completo.

- File: `vendor\llada\opencompass\tests\dataset\test_humaneval.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `run_humaneval_check(completion:Any) -> Any`
    Descrizione: Esegue pipeline o job completo.
  Classi:
  - `TestHumaneval`
    Metodo: `test_vanilla(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_python_quote(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_bare_quote(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_error_space_quote(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_import_1(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_import_2(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_import_3(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_comment(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_additional(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `vendor\llada\opencompass\tests\dataset\test_local_datasets.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `reload_datasets() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `load_datasets_conf(source:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `load_datasets(source:Any, conf:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `clean_string(value:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_check_data(ms_dataset:Dataset | DatasetDict, oc_dataset:Dataset | DatasetDict, sample_size:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `TestingLocalDatasets`
    Metodo: `test_datasets(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `vendor\llada\opencompass\tests\dataset\test_ms_datasets.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `reload_datasets() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `load_datasets_conf(source:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `load_datasets(source:Any, conf:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `clean_string(value:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_check_data(ms_dataset:Dataset | DatasetDict, oc_dataset:Dataset | DatasetDict, sample_size:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `TestingMsDatasets`
    Metodo: `test_datasets(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `vendor\llada\opencompass\tests\openicl\test_prompt_template.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `TestPromptTemplate`
    Metodo: `setUp(self:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `test_init(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_generate_ice_item(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_generate_label_prompt_item(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_generate_item(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `vendor\llada\opencompass\tests\prompt\test_api_template_parser.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `TestAPITemplateParser`
    Metodo: `setUp(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `test_parse_template_str_input(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_parse_template_list_input(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_parse_template_PromptList_input_no_meta_template(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_parse_template_PromptList_input_with_meta_template(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `vendor\llada\opencompass\tests\prompt\test_lm_template_parser.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `TestLMTemplateParser`
    Metodo: `setUp(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `test_parse_template_str_input(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_parse_template_list_input(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_parse_template_PromptList_input_no_meta_template(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_parse_template_PromptList_input_with_meta_template(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `vendor\llada\opencompass\tests\prompt\test_prompt_list.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `TestPromptList`
    Metodo: `test_initialization(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_format(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_replace(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_add(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
    Metodo: `test_str(self:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `vendor\llada\opencompass\tools\case_analyzer.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `parse_args() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `dispatch_tasks(cfg:Any, force:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `main() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `BadcaseShower`
    Metodo: `__init__(self:Any, cfg:ConfigDict) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `run(self:Any) -> Any`
    Descrizione: Esegue pipeline o job completo.

- File: `vendor\llada\opencompass\tools\collect_code_preds.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `parse_args() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `gpt_python_postprocess(ori_prompt:str, text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `wizardcoder_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `collect_preds(filename:str) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `main() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\tools\compare_configs.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_files(folder:Any, extensions:Any, ignore_folder:Any=[]) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `compare_folders(folder1:Any, folder2:Any, extensions:Any, ignore_folder:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `main() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\tools\convert_alignmentbench.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `extract_predictions_from_json(input_folder:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `process_jsonl(file_path:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `save_as_json(data:Any, output_file:Any='./alignment_bench.json') -> Any`
    Descrizione: Serializza e salva output su disco.
  - `parse_args() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\tools\list_configs.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `parse_args() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `main() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\tools\prediction_merger.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `parse_args() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `dispatch_tasks(cfg:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `main() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `PredictionMerger`
    Metodo: `__init__(self:Any, cfg:ConfigDict) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `run(self:Any) -> Any`
    Descrizione: Esegue pipeline o job completo.

- File: `vendor\llada\opencompass\tools\prompt_viewer.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `parse_args() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `parse_model_cfg(model_cfg:ConfigDict) -> Dict[str, ConfigDict]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `parse_dataset_cfg(dataset_cfg:ConfigDict) -> Dict[str, ConfigDict]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `print_prompts(model_cfg:Any, dataset_cfg:Any, count:Any=1) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `main() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\tools\test_api_model.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `test_model(model_cfg:ConfigDict) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `parse_args() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `parse_model_cfg(model_cfg:ConfigDict) -> Dict[str, ConfigDict]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `main() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\tools\update_dataset_suffix.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_prompt_hash(dataset_cfg:Union[ConfigDict, List[ConfigDict]]) -> str`
    Descrizione: Recupera valore/stato calcolato.
  - `get_hash(path:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `check_and_rename(filepath:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `main() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\opencompass\tools\viz_multi_model.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `main(cfg_paths:List[Path]=Option(..., help='The path to the config file of the task', exists=True), work_dirs:List[Path]=Option(..., help='The work dirs for the task(named by timestamp), need to ensure the order is the same as cfg_paths.', exists=True), group:str=Option(None, help='If not None, show the accuracy in the group.')) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\visualization\generate.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `add_gumbel_noise(logits:Any, temperature:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_num_transfer_tokens(mask_index:Any, steps:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `generate(model:Any, prompt:Any, tokenizer:Any, steps:Any=128, gen_length:Any=128, block_length:Any=128, temperature:Any=0.0, cfg_scale:Any=0.0, remasking:Any='low_confidence', mask_id:Any=126336) -> Any`
    Descrizione: Genera output testo o sequenze.
  - `main() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\visualization\html_to_png.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\llada\visualization\visualization_paper.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `parse_generation_history(file_path:str) -> Dict[int, List[str]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `track_token_positions(history:Dict[int, List[str]]) -> List[int]`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
  - `generate_background_color(step:int, max_step:int) -> str`
    Descrizione: Genera output testo o sequenze.
  - `generate_step_visualization(current_step:int, current_tokens:List[str], token_steps:List[int], max_step:int) -> str`
    Descrizione: Genera output testo o sequenze.
  - `main(target_step:int=64) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\llada\visualization\visualization_zhihu.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `parse_generation_history(file_path:str) -> Dict[int, List[str]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `track_token_positions(history:Dict[int, List[str]]) -> List[int]`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
  - `generate_background_color(step:int, max_step:int) -> str`
    Descrizione: Genera output testo o sequenze.
  - `generate_step_visualization(current_step:int, current_tokens:List[str], token_steps:List[int], max_step:int) -> str`
    Descrizione: Genera output testo o sequenze.
  - `main(target_step:int=64) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

### `vendor\dinfer`
- File: `vendor\dinfer\benchmarks\benchmark.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `setup_distributed(rank:Any, world_size:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `load_inputs(dataset:Any, tokenizer:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `main(world_size:Any, rank:Any, gpu_id:Any, args:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `process_args(args:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\dinfer\benchmarks\benchmark_dataset.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `setup_distributed(rank:Any, world_size:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_bucket_length(length:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `load_inputs(dataset:Any, tokenizer:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `cal_bucket_len(args:Any, all_input_ids:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `warmup_cudagraph(rank:Any, device:Any, dllm:Any, args:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `cut_eos(data:Any, eos_id:Any=156892) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `main(world_size:Any, rank:Any, gpu_id:Any, args:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `process_args(args:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\dinfer\benchmarks\benchmark_dataset_fastdllm.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `setup_distributed(rank:Any, world_size:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_bucket_length(length:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `load_inputs(dataset:Any, tokenizer:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `cal_bucket_len(args:Any, all_input_ids:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `warmup_cudagraph(rank:Any, device:Any, dllm:Any, args:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `main(world_size:Any, rank:Any, gpu_id:Any, args:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\dinfer\benchmarks\benchmark_dataset_sglang.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_bucket_length(length:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `load_inputs(dataset:Any, tokenizer:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `cal_bucket_len(args:Any, all_input_ids:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `cut_eos(data:Any, eos_id:Any=156892) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `load_quant_config(config:Any, path:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `main(world_size:Any, rank:Any, gpu_id:Any, args:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\dinfer\benchmarks\benchmark_dataset_sorted.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_bucket_length(length:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `load_inputs(dataset:Any, tokenizer:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `cal_bucket_len(args:Any, all_input_ids:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `warmup_cudagraph(rank:Any, device:Any, dllm:Any, args:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `cut_eos(data:Any, eos_id:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `main(world_size:Any, rank:Any, gpu_id:Any, args:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `process_args(args:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\dinfer\evaluations\eval_dinfer.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `warmup_cudagraph(rank:Any, device:Any, dllm:Any, gen_len:Any, block_length:Any, batch_size:Any, vocab_size:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `cut_eos(data:Any, eos_id:Any=156892) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `run_benchmark(world_size:Any, rank:Any, gpu_id:Any, tokenizer:Any, args:Any) -> Any`
    Descrizione: Esegue pipeline o job completo.
  - `set_seed(seed:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `EvalConfig`
  - `DInferEvalHarness`
    Metodo: `__init__(self:Any, model_path:Any='', device:Any='cuda', mask_id:Any=126336, eos_id:Any=126081, max_length:Any=4096, batch_size:Any=2, mc_num:Any=128, is_check_greedy:Any=True, gen_length:Any=1024, block_length:Any=1024, save_dir:Any=None, show_speed:Any=False, parallel_decoding:Any='threshold', threshold:float=0.9, cache:str='', warmup_times:int=0, low_threshold:float=0.3, cont_weight:float=0, use_credit:bool=False, tp_size:int=1, parallel:Any='dp', use_compile:Any=True, master_port:Any=23456, use_cudagraph:Any=True, gpus:Any='0,1,2,3', use_bd:Any=False, prefix_look:Any=0, after_look:Any=0, use_shift:Any=False, model_type:Any='llada', save_samples:Any=False, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `rank(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `world_size(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `tokenizer_name(self:Any) -> str`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
    Metodo: `apply_chat_template(self:Any, chat_history:Any, **kwargs:Any) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_forward_process(self:Any, batch:Any, prompt_index:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_logits(self:Any, batch:Any, prompt_index:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_loglikelihood(self:Any, prefix:Any, target:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `suffix_greedy_prediction(self:Any, prefix:Any, target:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_encode_pair(self:Any, context:Any, continuation:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `loglikelihood(self:Any, requests:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `loglikelihood_rolling(self:Any, requests:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate_until(self:Any, requests:Any) -> Any`
    Descrizione: Genera output testo o sequenze.

- File: `vendor\dinfer\evaluations\eval_dinfer_sglang.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `cut_eos(data:Any, eos_id:Any=156892) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `run_benchmark(world_size:Any, rank:Any, gpu_id:Any, tokenizer:Any, args:Any) -> Any`
    Descrizione: Esegue pipeline o job completo.
  - `set_seed(seed:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `EvalConfig`
  - `DInferEvalHarness`
    Metodo: `__init__(self:Any, model_path:Any='', device:Any='cuda', mask_id:Any=126336, eos_id:Any=126081, max_length:Any=4096, batch_size:Any=2, mc_num:Any=128, is_check_greedy:Any=True, gen_length:Any=1024, block_length:Any=1024, save_dir:Any=None, show_speed:Any=False, parallel_decoding:Any='threshold', threshold:float=0.9, cache:str='', warmup_times:int=0, low_threshold:float=0.3, cont_weight:float=0, use_credit:bool=False, tp_size:int=1, parallel:Any='dp', use_compile:Any=True, master_port:Any=23456, use_cudagraph:Any=True, gpus:Any='0;1;2;3', use_bd:Any=False, prefix_look:Any=0, after_look:Any=0, use_shift:Any=False, model_type:Any='llada2', save_samples:Any=False, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `rank(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `world_size(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `tokenizer_name(self:Any) -> str`
    Descrizione: Gestisce tokenizzazione o manipolazione token.
    Metodo: `apply_chat_template(self:Any, chat_history:Any, **kwargs:Any) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_forward_process(self:Any, batch:Any, prompt_index:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_logits(self:Any, batch:Any, prompt_index:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_loglikelihood(self:Any, prefix:Any, target:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `suffix_greedy_prediction(self:Any, prefix:Any, target:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_encode_pair(self:Any, context:Any, continuation:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `loglikelihood(self:Any, requests:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `loglikelihood_rolling(self:Any, requests:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate_until(self:Any, requests:Any) -> Any`
    Descrizione: Genera output testo o sequenze.

- File: `vendor\dinfer\evaluations\tasks\mbpp_sanitized\utils.py`
  Logica d'uso: Runtime helpers, device/dtype control, tiny fallback model/tokenizer.
  Funzioni:
  - `process_docs(dataset:Dataset) -> Dataset`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `pass_at_1(references:Any, predictions:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `humaneval_postprocess(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_process_answer(text:str) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\dinfer\python\dinfer\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\dinfer\python\dinfer\decoding\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\dinfer\python\dinfer\decoding\diffusion_runner.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `freeze_gc(enable_cudagraph_gc:bool) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_to_torch(model:torch.nn.Module, reverse:bool, num_tokens:int) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `set_torch_compile_config() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `patch_model(model:torch.nn.Module, enable_compile:bool, num_tokens:int, tp_group:GroupCoordinator) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_global_graph_memory_pool() -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `set_global_graph_memory_pool(val:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `ModelRunner`
    Metodo: `__init__(self:Any, model:torch.nn.Module, device:str='cuda', enable_cuda_graph:bool=True, supported_batch_sizes:Optional[list]=None, server_args:Any=None, max_length:Any=2048, block_length:Any=32, prefill_lengths:Any=[64, 96, 128], decoding_lengths:Any=[32], enable_compile:Any=True, cache_lengths:Any=None, use_cross_block:Any=True) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `init_device_graphs(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward_normal(self:Any, input_ids:torch.Tensor=None, position_ids:torch.Tensor=None, inputs_embeds:torch.Tensor=None, pp_proxy_tensors:Optional[torch.Tensor]=None, past_key_values:Any=None, replace_position:Optional[torch.LongTensor]=None, use_cache:Optional[bool]=None, attention_mask:Optional[torch.Tensor]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, input_ids:torch.Tensor=None, position_ids:torch.Tensor=None, inputs_embeds:torch.Tensor=None, pp_proxy_tensors:Optional[torch.Tensor]=None, past_key_values:Any=None, replace_position:Optional[torch.LongTensor]=None, use_cache:Optional[bool]=False, attention_mask:Optional[torch.Tensor]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__call__(self:Any, *args:Any, **kwds:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `CudaGraphRunner`
    Metodo: `__init__(self:Any, model_runner:ModelRunner) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_capture_graph(self:Any, graph:Any, pool:Any, stream:Any, run_once_fn:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_create_device_graph(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `capture(self:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `capture_one_batch_size(self:Any, bs:int, forward:Callable, is_decode_phase:bool=True, length:int=0, cache_length:int=0, use_mask:bool=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `can_run(self:Any, input_ids:Any, position_ids:Any, past_key_values:Any, is_decode_phase:Any=True, length:Any=0, cache_length:Any=0) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `replay_prepare(self:Any, input_ids:Any, position_ids:Any, past_key_values:Any, is_decode_phase:Any, length:Any, attention_mask:Any, cache_length:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `replay(self:Any, input_ids:Any, position_ids:Any, past_key_values:Any, is_decode_phase:Any, length:Any, attention_mask:Any, cache_length:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\dinfer\python\dinfer\decoding\generate_cache.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `cache_update_tag(strategy:Any, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `block_cache_api(model:Any, x:Any, past_key_values:Any, position:Any, block_mask_index:Any, update_cache:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `generate_with_cache(model:Any, prompt:Any, steps:Any=128, gen_length:Any=128, block_length:Any=128, temperature:Any=0.0, mask_id:Any=126336, log_flops:Any=False, threshold:Any=None, cache_update_iter:Any=None, eos_early_stop:Any=False, minimal_topk:Any=1, **kwargs:Any) -> Any`
    Descrizione: Genera output testo o sequenze.
  - `generate_with_prefixcache_update(model:Any, prompt:Any, steps:Any=128, gen_length:Any=128, block_length:Any=128, temperature:Any=0.0, mask_id:Any=126336, log_flops:Any=False, threshold:Any=None, cache_update_iter:Any=None, eos_early_stop:Any=False, minimal_topk:Any=1, **kwargs:Any) -> Any`
    Descrizione: Genera output testo o sequenze.

- File: `vendor\dinfer\python\dinfer\decoding\generate_dist.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `generate_dist(model:Any, prompt:Any, steps:Any=128, gen_length:Any=128, block_length:Any=128, temperature:Any=0.0, mask_id:Any=126336, decoding:Any='distributed', **kwargs:Any) -> Any`
    Descrizione: Genera output testo o sequenze.
  - `generate(model:Any, prompt:Any, rank:Any=0, world_size:Any=1, steps:Any=128, gen_length:Any=128, block_length:Any=128, temperature:Any=0.0, remasking:Any='low_confidence', mask_id:Any=126336, threshold:Any=None, factor:Any=None) -> Any`
    Descrizione: Genera output testo o sequenze.
  - `generate_with_cache(model:Any, prompt:Any, rank:Any=0, world_size:Any=1, steps:Any=128, gen_length:Any=128, block_length:Any=128, temperature:Any=0.0, remasking:Any='low_confidence', mask_id:Any=126336, threshold:Any=None, factor:Any=None) -> Any`
    Descrizione: Genera output testo o sequenze.
  - `generate_block_cache(model:Any, prompt:Any, rank:Any=0, world_size:Any=1, steps:Any=128, gen_length:Any=128, block_length:Any=128, temperature:Any=0.0, remasking:Any='low_confidence', mask_id:Any=126336, threshold:Any=None, factor:Any=None) -> Any`
    Descrizione: Genera output testo o sequenze.
  - `generate_with_dual_cache(model:Any, prompt:Any, steps:Any=128, gen_length:Any=128, block_length:Any=128, temperature:Any=0.0, remasking:Any='low_confidence', mask_id:Any=126336, threshold:Any=None, factor:Any=None) -> Any`
    Descrizione: Genera output testo o sequenze.

- File: `vendor\dinfer\python\dinfer\decoding\generate_fastdllm.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `generate_fastdllm(model:Any, prompt:Any, steps:Any=128, gen_length:Any=128, block_length:Any=128, temperature:Any=0.0, mask_id:Any=126336, eos_id:Any=126081, decoding:Any='fastdllm', **kwargs:Any) -> Any`
    Descrizione: Genera output testo o sequenze.
  - `generate(model:Any, prompt:Any, steps:Any=128, gen_length:Any=128, block_length:Any=128, temperature:Any=0.0, remasking:Any='low_confidence', mask_id:Any=126336, eos_id:Any=126081, threshold:Any=None, factor:Any=None, early_stop:Any=False) -> Any`
    Descrizione: Genera output testo o sequenze.
  - `generate_with_prefix_cache(model:Any, prompt:Any, steps:Any=128, gen_length:Any=128, block_length:Any=128, temperature:Any=0.0, remasking:Any='low_confidence', mask_id:Any=126336, eos_id:Any=126081, threshold:Any=None, factor:Any=None, early_stop:Any=False) -> Any`
    Descrizione: Genera output testo o sequenze.
  - `generate_with_dual_cache(model:Any, prompt:Any, steps:Any=128, gen_length:Any=128, block_length:Any=128, temperature:Any=0.0, remasking:Any='low_confidence', mask_id:Any=126336, eos_id:Any=126081, threshold:Any=None, factor:Any=None, early_stop:Any=False) -> Any`
    Descrizione: Genera output testo o sequenze.
  - `main() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\dinfer\python\dinfer\decoding\generate_hierarchy.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `generate_hierarchy(model:Any, prompt:Any, steps:Any=128, gen_length:Any=128, block_length:Any=128, temperature:Any=0.0, mask_id:Any=126336, decoding:Any='origin', **kwargs:Any) -> Any`
    Descrizione: Genera output testo o sequenze.

- File: `vendor\dinfer\python\dinfer\decoding\generate_merge.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `generate_merge(model:Any, prompt:Any, kvcache:Any, steps:Any=128, gen_length:Any=128, block_length:Any=128, temperature:Any=0.0, mask_id:Any=126336, eos_id:Any=126081, decoding:Any='merge', **kwargs:Any) -> Any`
    Descrizione: Genera output testo o sequenze.

- File: `vendor\dinfer\python\dinfer\decoding\generate_uniform.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `align_exp2(x:torch.Tensor) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `select_undecoded(seq_idx:Any, orig_x:Any, x:Any, block:Any, block_loc:Any, mask_id:Any, writeback:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `gather_blocks(x:torch.Tensor, idx:torch.Tensor, block_length:int) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `select_batch_sequences_by_mask_number(x:Any, valid_flag:Any, mask_id:Any, batch_size:Any) -> Any`
    Descrizione: Gestisce logica di mascheratura/filtri di posizioni.
  - `select_batch_sequences_by_order(x:Any, valid_flag:Any, mask_id:Any, batch_size:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `DiffusionLLM`
    Metodo: `generate(self:Any, prompt:Any, gen_length:Any=128, block_length:Any=128) -> Any`
    Descrizione: Genera output testo o sequenze.
  - `BlockRunner`
    Metodo: `__init__(self:Any, diff_iteration:Any, early_stop:Any, maximum_unroll:Any, expected_tpf:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `decode(self:Any, model:Any, decoder:Any, x:Any, kv_cache:Any, block:Any, block_loc:Any, block_id:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `BlockDiffusionRunner`
    Metodo: `__init__(self:Any, diff_iteration:Any, early_stop:Any, maximum_unroll:Any, expected_tpf:Any, backend:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `prefill(self:Any, model:Any, prefilling_x:Any, kv_cache:Any, pos_ids:Any, attn_mask:Any, prefilling_limit:Any, block_length:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `decode(self:Any, model:Any, decoder:Any, x:Any, kv_cache:Any, block:Any, block_loc:Any, block_id:Any, pos_ids:Any, attn_mask:Any, block_length:Any=32, cross_block_attn_mask:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `DiffusionIteration`
    Metodo: `__init__(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, model:Any, x:Any, kv_cache:Any, block:Any, block_loc:Any, block_id:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `BaseDiffusionIteration`
    Metodo: `__init__(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, model:Any, decoder:Any, x:Any, kv_cache:Any, block:Any, block_loc:Any, block_id:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `BlockDiffusionIteration`
    Metodo: `__init__(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, model:Any, decoder:Any, x:Any, kv_cache:Any, block:Any, block_loc:Any, block_id:Any, pos_ids:Any, attn_mask:Any, past_key_values:Any, replace_position:Any, backend:Any, is_cross_block:Any=False, block_length:Any=32) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `ShiftDiffusionIteration`
    Metodo: `__init__(self:Any, use_shift:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, model:Any, decoder:Any, x:Any, kv_cache:Any, block:Any, block_loc:Any, block_id:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `BlockWiseDiffusionLLM`
    Metodo: `__init__(self:Any, model:Any, decoder:Any, iterator_factory:Any, early_stop:Any=True, cache_factory:Any=None, maximum_unroll:Any=4, expected_tpf:Any=8, use_shift:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `num_forwards(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `cache_updates(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, prompt:Any, gen_length:Any=128, block_length:Any=128) -> Any`
    Descrizione: Genera output testo o sequenze.
  - `IterationSmooth`
    Metodo: `__init__(self:Any, model:Any, cont_weight:Any=0.3, cont_weight_init:Any=0.15, cont_weight_growth:Any=0.02, threshold_decay:Any=0.02) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `reset_input_embeds(self:Any, x:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, model:Any, decoder:Any, x:Any, kv_cache:Any, block:Any, block_loc:Any, block_id:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `IterSmoothDiffusionLLM`
    Metodo: `__init__(self:Any, model:Any, decoder:Any, iterator_factory:Any, early_stop:Any=True, cache_factory:Any=None, maximum_unroll:Any=4, expected_tpf:Any=8, cont_weight:Any=0.3, cont_weight_init:Any=0.15, cont_weight_growth:Any=0.02, threshold_decay:Any=0.02) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `num_forwards(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `cache_updates(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, prompt:Any, gen_length:Any=128, block_length:Any=128) -> Any`
    Descrizione: Genera output testo o sequenze.
  - `VicinityCacheIteration`
    Metodo: `__init__(self:Any, prefix_look:Any, after_look:Any, warmup_steps:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, model:Any, decoder:Any, x:Any, kv_cache:Any, block:Any, block_loc:Any, block_id:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `VicinityCacheDiffusionLLM`
    Metodo: `__init__(self:Any, model:Any, decoder:Any, iterator_factory:Any, cache_factory:Any, maximum_unroll:Any=4, expected_tpf:Any=8, prefix_look:Any=0, after_look:Any=0, warmup_steps:Any=0, early_stop:Any=True) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `num_forwards(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `cache_updates(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `IterSmoothWithVicinityCache`
    Metodo: `__init__(self:Any, model:Any, prefix_look:Any, after_look:Any, warmup_steps:Any, cont_weight:Any=0.3, cont_weight_init:Any=0.15, cont_weight_growth:Any=0.02, threshold_decay:Any=0.02) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `reset_input_embeds(self:Any, x:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, model:Any, decoder:Any, x:Any, kv_cache:Any, block:Any, block_loc:Any, block_id:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `IterSmoothWithVicinityCacheDiffusionLLM`
    Metodo: `__init__(self:Any, model:Any, decoder:Any, iterator_factory:Any, cache_factory:Any, maximum_unroll:Any=4, expected_tpf:Any=8, prefix_look:Any=0, after_look:Any=0, warmup_steps:Any=0, early_stop:Any=True, cont_weight:Any=0.3, cont_weight_init:Any=0.15, cont_weight_growth:Any=0.02, threshold_decay:Any=0.02) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `num_forwards(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `cache_updates(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `BlockWiseDiffusionLLMWithSP`
    Metodo: `__init__(self:Any, rank:Any, world_size:Any, model:Any, decoder:Any, iterator_factory:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, prompt:Any, gen_length:Any=128, block_length:Any=128) -> Any`
    Descrizione: Genera output testo o sequenze.
  - `BlockDiffusionLLMAttnmask`
    Metodo: `__init__(self:Any, model:Any, decoder:Any, iterator_factory:Any, early_stop:Any=True, maximum_unroll:Any=4, expected_tpf:Any=8, backend:Any='vllm') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `num_forwards(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `cache_updates(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, prompt:Any, gen_length:Any=128, block_length:Any=128) -> Any`
    Descrizione: Genera output testo o sequenze.
  - `BlockDiffusionLLM`
    Metodo: `__init__(self:Any, model:Any, decoder:Any, iterator_factory:Any, cache_factory:Any, early_stop:Any=True, maximum_unroll:Any=1, expected_tpf:Any=15, backend:Any='vllm', mini_batch_size:Any=4, prefilling_limit:Any=128, use_naive_batching:Any=True) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `num_forwards(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `cache_updates(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `naive_batching_generate(self:Any, prompt:Any, gen_length:Any=128, block_length:Any=128) -> Any`
    Descrizione: Genera output testo o sequenze.
    Metodo: `dynamic_batching_generate(self:Any, prompt:Any, gen_length:Any=128, block_length:Any=128) -> Any`
    Descrizione: Genera output testo o sequenze.

- File: `vendor\dinfer\python\dinfer\decoding\parallel_strategy.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `broadcast_if_needed(x:Any, src:Any=0, group:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_transfer_index_hierarchy_fast_v2(logits:Any, temperature:Any, remasking:Any, mask_index:Any, x:Any, num_transfer_tokens:Any, mask_id:Any, threshold:Any=None, low_threshold:Any=None) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `get_transfer_index_hierarchy_remask(logits:Any, temperature:Any, mask_index:Any, x:Any, num_transfer_tokens:Any, mask_id:Any, threshold:Any=None, low_threshold:Any=None, remask_threshold:Any=0.4) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `get_transfer_index_cache(logits:Any, mask_index:Any, x:Any, block_end:Any, num_transfer_tokens:Any, temperature:Any, remasking:Any, threshold:Any=None, minimal_topk:Any=1) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `get_transfer_index(logits:Any, temperature:Any, remasking:Any, mask_index:Any, x:Any, num_transfer_tokens:Any, threshold:Any=None) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `get_transfer_index_dynamic(logits:Any, temperature:Any, remasking:Any, mask_index:Any, x:Any, num_transfer_tokens:Any, factor:Any=1) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `get_transfer_index_threshold(logits:Any, temperature:Any, mask_index:Any, x:Any, mask_id:Any, threshold:Any, rm_mask:Any=True, use_float64:Any=False, **kwargs:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  Classi:
  - `ParallelDecoder`
    Metodo: `__init__(self:Any, temperature:Any, remasking:Any='low_confidence', mask_id:Any=126336) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `block_init(self:Any, block_x:Any, block_id:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `decode(self:Any, logits:Any, block_start:Any, block_end:Any, x:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `ThresholdParallelDecoder`
    Metodo: `__init__(self:Any, temperature:Any, threshold:Any, remasking:Any='low_confidence', mask_id:Any=126336, eos_id:Any=126081, use_float64:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `decode(self:Any, logits:Any, block_start:Any, block_end:Any, x:Any, iter_threshold:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `batch_decode(self:Any, logits:Any, block_start:Any, x:Any, block_length:Any, iter_threshold:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `CreditThresholdParallelDecoder`
    Metodo: `__init__(self:Any, credit_alpha:Any=0.7, boost_gamma:Any=0.2, decay_beta:Any=0.8, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_apply_credit_fusion(self:Any, logits:Any, mask_index:Any, key:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `decode(self:Any, logits:Any, block_start:Any, block_end:Any, x:Any, iter_threshold:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `FixedParallelDecoder`
    Metodo: `__init__(self:Any, temperature:Any, steps:Any, remasking:Any='low_confidence', mask_id:Any=126336) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `block_init(self:Any, block_x:Any, block_id:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `decode(self:Any, logits:Any, block_start:Any, block_end:Any, x:Any, iter_threshold:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `HierarchyDecoder`
    Metodo: `__init__(self:Any, temperature:Any, remasking:Any='low_confidence', mask_id:Any=126336, eos_id:Any=126081, threshold:Any=None, low_threshold:Any=0.4) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_transfer_index(self:Any, logits:Any, mask_index:Any, iter_threshold:Any, **kwargs:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `block_init(self:Any, block_x:Any, block_id:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `decode(self:Any, logits:Any, block_start:Any, block_end:Any, x:Any, iter_threshold:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\dinfer\python\dinfer\decoding\serving.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `is_port_available(port:int) -> bool`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `find_continuous_ports(num_ports:int, start_port:int=30000, end_port:int=65535, protocol:str='tcp', max_attempts:int=300, retry_delay:float=0.1) -> Optional[Tuple[int, List[int]]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `load_local_config(model_dir:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `_patched(cls:Any, name_or_path:Any, *args:Any, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `init_generator(model:Any, sample_params:Any, backend:Any='vllm', max_length:Any=4096) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `generate(dllm:Any, device:Any, req_q:Any, res_q:Any) -> Any`
    Descrizione: Genera output testo o sequenze.
  - `sglang_llada2_server_process(model_path:Any, sample_params:Any, world_size:Any, rank:Any, gpu_id:Any, q:Any, res_q:Any, master_port:Any, error_q:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_sglang_llada2_server_process(model_path:Any, sample_params:Any, world_size:Any, rank:Any, gpu_id:Any, q:Any, res_q:Any, master_port:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `moe_server_process(model_path:Any, sample_params:Any, world_size:Any, rank:Any, gpu_id:Any, q:Any, res_q:Any, master_port:Any, error_q:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_moe_server_process(model_path:Any, sample_params:Any, world_size:Any, rank:Any, gpu_id:Any, q:Any, res_q:Any, master_port:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `server_process(model_path:Any, sample_params:Any, world_size:Any, rank:Any, gpu_id:Any, q:Any, res_q:Any, master_port:Any, error_q:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_server_process(model_path:Any, sample_params:Any, world_size:Any, rank:Any, gpu_id:Any, q:Any, res_q:Any, master_port:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `SamplingParams`
    Metodo: `__init__(self:Any, threshold:Any=0.9, low_threshold:Any=0.6, cache:Any='dual', temperature:Any=0.0, early_stop:Any=True, cont_weight:Any=0.3, prefix_look:Any=16, after_look:Any=16, warmup_steps:Any=4, enable_torch_compile:Any=True, mask_id:Any=156895, eos_id:Any=156892, parallel_decoding:Any='threshold', use_credit:Any=False, use_bd:Any=True, max_length:Any=4096, ep_size:Any=1, prefilling_limit:Any=256, mini_batch_size:Any=1, batch_size:Any=1, use_naive_batching:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `ServerGroup`
    Metodo: `__init__(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `add_request(self:Any, req:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_response(self:Any, timeout:Any=None) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `start_server(self:Any, model_path:Any, model_type:Any, sample_params:Any, server_port:Any, gpus:Any, backend:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `is_running(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `stop_running(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `ServerHandle`
    Metodo: `__init__(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `add_requests(self:Any, reqs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_responses(self:Any, timeout:Any=None) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `start_server(self:Any, model_path:Any, model_type:Any, sample_params:Any, server_port:Any, num_gpus:Any, dp_size:Any, tpep_size:Any, backend:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `is_running(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `stop_running(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `DiffusionLLMServing`
    Metodo: `__init__(self:Any, model:Any, model_type:Any='llada2', sample_params:Any=None, server_port:Any=None, num_gpus:Any=None, dp_size:Any=None, tpep_size:Any=None, backend:Any='sglang', timeout:Any=None, max_retries:Any=3, start_port:Any=30000, end_port:Any=60000) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, prompts:Any, gen_length:Any=128, block_length:Any=128) -> Any`
    Descrizione: Genera output testo o sequenze.
    Metodo: `stop_serving(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `port_selection(self:Any, dp_size:Any, start_port:Any, end_port:Any, max_retries:Any=3) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\dinfer\python\dinfer\decoding\utils.py`
  Logica d'uso: Runtime helpers, device/dtype control, tiny fallback model/tokenizer.
  Funzioni:
  - `add_gumbel_noise(logits:Any, temperature:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_num_transfer_tokens(mask_index:Any, steps:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `calculate_op_num(x:Any, hidden_size:Any=4096, mlp_hidden_size:Any=12288, vocab_size:Any=126464, num_hidden_layers:Any=32, cache_length:Any=0) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `gather_sequence_block(partial_data:Any, partial_start:Any, partial_end:Any, block_start:Any, block_end:Any, rank:Any, world_size:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `TokenArray`
    Metodo: `__init__(self:Any, prompt:Any, gen_length:Any, mask_id:Any, eos_id:Any, device:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `total_length(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `batch_size(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `device(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `expand(self:Any, new_len:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_generated_tokens(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `select_seqs(self:Any, idx:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__getitem__(self:Any, idx:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__setitem__(self:Any, idx:Any, vals:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `DistAlignedTokenArray`
    Metodo: `__init__(self:Any, prompt:Any, gen_length:Any, mask_id:Any, eos_id:Any, device:Any, rank:Any, world_size:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `total_length(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `device(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_generated_tokens(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `expand(self:Any, new_len:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__getitem__(self:Any, idx:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__setitem__(self:Any, idx:Any, vals:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `BlockLoc`
    Metodo: `__init__(self:Any, start:Any, end:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `BlockIterator`
    Metodo: `__init__(self:Any, x:Any, block_length:Any, start_block_align:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_get_first_block_start(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__iter__(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__next__(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `BlockDiffusionIterator`
    Metodo: `__init__(self:Any, x:Any, block_length:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_get_first_block_start(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__iter__(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__next__(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `BlockIteratorFactory`
    Metodo: `__init__(self:Any, start_block_align:Any=False, use_block_diffusion:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `create(self:Any, x:Any, block_length:Any) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
  - `KVCache`
    Metodo: `__init__(self:Any, past_key_values:Any, backend:Any='vllm', length:Any=2048, cache_align_size:Any=128) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `consolidate(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_consolidate_raw(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `num_layers(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `seq_len(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_keys(self:Any, layer_idx:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `get_values(self:Any, layer_idx:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `update(self:Any, key_states:Any, val_states:Any, layer_idx:Any, replace_position:Any=None, backend:Any='vllm') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `DiffusionKVCacheManager`
    Metodo: `__init__(self:Any, cache_update_freq:Any=None, cache_type:Any='prefix', backend:Any='vllm', max_length:Any=2048) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `require_update(self:Any, iter_no:Any, block_start:Any, block_end:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `update(self:Any, past_key_values:Any, range_start:Any=None, range_end:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `range_update(self:Any, past_key_values:Any, range_start:Any=0, range_end:Any=0, block_length:Any=32) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_key_values(self:Any, block_start:Any, block_end:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `BlockDiffusionPrefixCacheManager`
    Metodo: `get_key_values(self:Any, block_start:Any, block_end:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `extend_cache(self:Any, end:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `KVCacheFactory`
    Metodo: `__init__(self:Any, cache_type:Any, cache_update_freq:Any=None, is_bd_model:Any=False, backend:Any='vllm', max_length:Any=2048) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `create(self:Any) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.

- File: `vendor\dinfer\python\dinfer\into_sglang\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\dinfer\python\dinfer\into_sglang\algorithm.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `dInferDiffusionAlgorithm`
    Metodo: `__init__(self:Any, config:DiffusionConfig) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `run(self:Any, model_runner:ModelRunner, forward_batch:ForwardBatch) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]`
    Descrizione: Esegue pipeline o job completo.

- File: `vendor\dinfer\python\dinfer\model\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\dinfer\python\dinfer\model\configuration_bailing_moe_v2.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `BailingMoeV2Config`
    Metodo: `__init__(self:Any, vocab_size:Any=30592, hidden_size:Any=1024, intermediate_size:Any=None, num_hidden_layers:Any=24, num_attention_heads:Any=16, num_key_value_heads:Any=0, hidden_act:Any='silu', use_qkv_bias:Any=False, use_qk_norm:Any=False, use_bias:Any=True, rms_norm_eps:Any=1e-05, norm_head:Any=False, tie_word_embeddings:Any=False, embedding_dropout:Any=0.1, attention_dropout:Any=0.1, output_dropout:Any=0.1, initializer_range:Any=0.02, max_position_embeddings:Any=16384, rope_theta:Any=10000.0, use_cache:Any=True, use_sliding_window:Any=False, sliding_window:Any=4096, max_window_layers:Any=28, rope_scaling:Any=None, pad_token_id:Any=126081, num_experts:Any=16, num_shared_experts:Any=0, num_experts_per_tok:Any=2, n_group:Any=8, topk_group:Any=4, routed_scaling_factor:Any=2.5, moe_intermediate_size:Any=None, first_k_dense_replace:Any=0, head_dim:Any=None, output_router_logits:Any=False, partial_rotary_factor:Any=0.5, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\dinfer\python\dinfer\model\configuration_llada.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `StrEnum`
    Metodo: `__str__(self:Any) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__repr__(self:Any) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LayerNormType`
  - `ActivationType`
  - `BlockType`
  - `InitFnType`
  - `ModelConfig`
    Metodo: `effective_n_kv_heads(self:Any) -> int`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `ActivationCheckpointingStrategy`
  - `LLaDAConfig`
    Metodo: `__init__(self:Any, use_cache:bool=False, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `num_attention_heads(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `num_hidden_layers(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `hidden_size(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\dinfer\python\dinfer\model\configuration_llada2_moe.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LLaDA2MoeConfig`
    Metodo: `__init__(self:Any, vocab_size:Any=30592, hidden_size:Any=1024, intermediate_size:Any=None, num_hidden_layers:Any=24, num_attention_heads:Any=16, num_key_value_heads:Any=0, hidden_act:Any='silu', use_qkv_bias:Any=False, use_qk_norm:Any=False, use_bias:Any=True, rms_norm_eps:Any=1e-05, norm_head:Any=False, tie_word_embeddings:Any=False, embedding_dropout:Any=0.1, attention_dropout:Any=0.1, output_dropout:Any=0.1, initializer_range:Any=0.02, max_position_embeddings:Any=16384, rope_theta:Any=10000.0, use_cache:Any=True, use_sliding_window:Any=False, sliding_window:Any=4096, max_window_layers:Any=28, rope_scaling:Any=None, pad_token_id:Any=126081, num_experts:Any=16, num_shared_experts:Any=0, num_experts_per_tok:Any=2, n_group:Any=8, topk_group:Any=4, routed_scaling_factor:Any=2.5, moe_intermediate_size:Any=None, first_k_dense_replace:Any=0, head_dim:Any=None, output_router_logits:Any=False, partial_rotary_factor:Any=0.5, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\dinfer\python\dinfer\model\configuration_olmoe.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `OlmoeConfig`
    Metodo: `__init__(self:Any, vocab_size:Any=-1, hidden_size:Any=-1, dense_intermediate_size:Any=-1, expert_intermediate_size:Any=-1, shared_expert_intermediate_size:Any=-1, num_hidden_layers:Any=-1, num_attention_heads:Any=-1, num_key_value_heads:Any=None, hidden_act:Any='silu', max_position_embeddings:Any=4096, initializer_range:Any=0.02, rms_norm_eps:Any=1e-05, use_cache:Any=False, pad_token_id:Any=1, bos_token_id:Any=None, eos_token_id:Any=50279, tie_word_embeddings:Any=False, rope_theta:Any=-1, partial_rotary_factor:Any=-1, rope_scaling:Any=None, attention_bias:Any=False, attention_dropout:Any=0.0, clip_qkv:Any=None, num_experts_per_tok:Any=-1, num_experts:Any=-1, output_router_logits:Any=False, router_aux_loss_coef:Any=0.01, norm_topk_prob:Any=None, qk_layernorm:Any=None, moe_layer_freq:Any=[], moe_router_enable_expert_bias:Any=None, moe_router_score_function:Any=None, routed_scaling_factor:Any=1, router_num_group:Any=-2, router_topk_group:Any=-2, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\dinfer\python\dinfer\model\modeling_fused_olmoe.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `torch_all_reduce(tensor:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `replace_linear_class(linear:nn.Linear, style:Literal['colwise', 'rowwise'], quant_config:Any) -> Union[ColumnParallelLinear, RowParallelLinear]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_all_gather_cat(tensor:torch.Tensor, dim:int=1, group:Optional[dist.ProcessGroup]=None, normal_len:int=0, last_len:int=0) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `rotate_half(x:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `apply_rotary_pos_emb(q:Any, k:Any, cos:Any, sin:Any, position_ids:Any=None, unsqueeze_dim:Any=1) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `repeat_kv(hidden_states:torch.Tensor, n_rep:int) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `show_modules(module:nn.Module, prefix:str='') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `H2Embed`
    Metodo: `__init__(self:Any, embedding:nn.Embedding, tau:float=1.0) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__call__(self:Any, x:torch.Tensor, mask_index:Optional[torch.Tensor]=None, logits:Optional[torch.Tensor]=None, iter_cont_weight:float=0.0) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `OlmoeRotaryEmbedding`
    Metodo: `__init__(self:Any, dim:Any=None, max_position_embeddings:Any=2048, base:Any=10000, device:Any=None, scaling_factor:Any=1.0, rope_type:Any='default', config:Optional[OlmoeConfig]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_dynamic_frequency_update(self:Any, position_ids:Any, device:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, x:Any, position_ids:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `OlmoeMLP`
    Metodo: `__init__(self:Any, config:Any, mlp_type:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, x:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `OlmoeAttention`
    Metodo: `__init__(self:Any, config:OlmoeConfig, layer_idx:Optional[int]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, hidden_states:torch.Tensor, attention_mask:Optional[torch.Tensor]=None, position_ids:Optional[torch.LongTensor]=None, past_key_value:Optional[Cache]=None, output_attentions:bool=False, use_cache:bool=False, cache_position:Optional[torch.LongTensor]=None, position_embeddings:Optional[Tuple[torch.Tensor, torch.Tensor]]=None, **kwargs:Any) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `OlmoeSdpaAttention`
    Metodo: `forward(self:Any, hidden_states:torch.Tensor, attention_mask:Optional[torch.Tensor]=None, position_ids:Optional[torch.LongTensor]=None, past_key_value:Optional[Cache]=None, output_attentions:bool=False, use_cache:bool=False, cache_position:Optional[torch.LongTensor]=None, position_embeddings:Optional[Tuple[torch.Tensor, torch.Tensor]]=None, replace_position:Any=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `OlmoeMoE`
    Metodo: `__init__(self:Any, config:Any, prefix:str='') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, hidden_states:torch.Tensor) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `OlmoeDecoderLayer`
    Metodo: `__init__(self:Any, config:OlmoeConfig, layer_idx:int) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, hidden_states:torch.Tensor, attention_mask:Optional[torch.Tensor]=None, position_ids:Optional[torch.LongTensor]=None, past_key_value:Optional[Cache]=None, output_attentions:Optional[bool]=False, output_router_logits:Optional[bool]=False, use_cache:Optional[bool]=False, cache_position:Optional[torch.LongTensor]=None, replace_position:Optional[torch.LongTensor]=None, position_embeddings:Optional[Tuple[torch.Tensor, torch.Tensor]]=None, **kwargs:Any) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `OlmoePreTrainedModel`
    Metodo: `_init_weights(self:Any, module:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `OlmoeModel`
    Metodo: `__init__(self:Any, config:OlmoeConfig) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_input_embeddings(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `set_input_embeddings(self:Any, value:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, input_ids:torch.LongTensor=None, attention_mask:Optional[torch.Tensor]=None, position_ids:Optional[torch.LongTensor]=None, past_key_values:Optional[Cache]=None, inputs_embeds:Optional[torch.FloatTensor]=None, use_cache:Optional[bool]=None, output_attentions:Optional[bool]=None, output_hidden_states:Optional[bool]=None, output_router_logits:Optional[bool]=None, return_dict:Optional[bool]=None, cache_position:Optional[torch.LongTensor]=None, replace_position:Optional[torch.LongTensor]=None) -> Union[Tuple, MoeModelOutputWithPast]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `FusedOlmoeForCausalLM`
    Metodo: `__init__(self:Any, config:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_keys_to_rename_on_load_unexpected(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `load_state_dict(self:Any, state_dict:Any, strict:Any=True, dtype:Any=torch.bfloat16) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
    Metodo: `load_sharded_safetensors(self:Any, model_dir:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
    Metodo: `load_weights(self:Any, model_path:Any, torch_dtype:Any=torch.bfloat16) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
    Metodo: `show_modules(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_input_embeddings(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `set_input_embeddings(self:Any, value:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_output_embeddings(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `set_output_embeddings(self:Any, new_embeddings:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `set_decoder(self:Any, decoder:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_decoder(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `init_h2e_module(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, input_ids:torch.LongTensor=None, attention_mask:Optional[torch.Tensor]=None, position_ids:Optional[torch.LongTensor]=None, past_key_values:Optional[KVCache]=None, inputs_embeds:Optional[torch.FloatTensor]=None, labels:Optional[torch.LongTensor]=None, use_cache:Optional[bool]=None, output_attentions:Optional[bool]=None, output_hidden_states:Optional[bool]=None, output_router_logits:Optional[bool]=None, return_dict:Optional[bool]=None, cache_position:Optional[torch.LongTensor]=None, replace_position:Optional[torch.LongTensor]=None, num_logits_to_keep:int=0) -> Union[Tuple, MoeCausalLMOutputWithPast]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `tensor_parallel(self:Any, tp_size:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\dinfer\python\dinfer\model\modeling_llada.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `_all_gather_cat(tensor:torch.Tensor, dim:int=1, group:Optional[dist.ProcessGroup]=None, normal_len:int=0, last_len:int=0) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `replace_linear_class(linear:nn.Linear, style:Literal['colwise', 'rowwise'], rank:int=0, world_size:int=1) -> Union[ColumnParallelLinear, RowParallelLinear]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `scaled_dot_product_attention(q:Any, k:Any, v:Any, mask:Any=None, attn_mask:Any=None, dropout_p:Any=0.0, is_causal:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `init_weights(config:ModelConfig, module:Union[nn.Linear, nn.Embedding], d:Optional[int]=None, layer_id:Optional[int]=None, std_factor:float=1.0, type_of_module:Optional[ModuleType]=None) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `ensure_finite_(x:torch.Tensor, check_neg_inf:bool=True, check_pos_inf:bool=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `activation_checkpoint_function(cfg:ModelConfig) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_non_meta_init_device(config:ModelConfig) -> torch.device`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `causal_attention_bias(seq_len:int, device:torch.device) -> torch.FloatTensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_causal_attention_bias(cache:BufferCache, seq_len:int, device:torch.device) -> torch.Tensor`
    Descrizione: Recupera valore/stato calcolato.
  - `alibi_attention_bias(seq_len:int, config:ModelConfig, device:torch.device) -> torch.FloatTensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `create_model_config_from_pretrained_config(config:LLaDAConfig) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
  Classi:
  - `H2Embed`
    Metodo: `__init__(self:Any, embedding:nn.Embedding, tau:float=1.0) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__call__(self:Any, x:torch.Tensor, mask_index:Optional[torch.Tensor]=None, logits:Optional[torch.Tensor]=None, iter_cont_weight:float=0.0) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `ModuleType`
  - `BufferCache`
  - `Dropout`
    Metodo: `forward(self:Any, input:torch.Tensor) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LayerNormBase`
    Metodo: `__init__(self:Any, config:ModelConfig, size:Optional[int]=None, elementwise_affine:Optional[bool]=True, eps:float=1e-05) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, x:torch.Tensor) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `build(cls:Any, config:ModelConfig, size:Optional[int]=None, **kwargs:Any) -> LayerNormBase`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
    Metodo: `_cast_if_autocast_enabled(self:Any, tensor:torch.Tensor, dtype:Optional[torch.dtype]=None) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `reset_parameters(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LayerNorm`
    Metodo: `__init__(self:Any, config:ModelConfig, size:Optional[int]=None, low_precision:bool=False, elementwise_affine:Optional[bool]=None, eps:float=1e-05) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, x:torch.Tensor) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `RMSLayerNorm`
    Metodo: `__init__(self:Any, config:ModelConfig, size:Optional[int]=None, elementwise_affine:Optional[bool]=None, eps:float=1e-05) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, x:torch.Tensor) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `GemmaRMSLayerNorm`
    Metodo: `__init__(self:Any, config:ModelConfig, size:Optional[int]=None, elementwise_affine:Optional[bool]=None, eps:float=1e-05) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, x:torch.Tensor) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `RotaryEmbedding`
    Metodo: `__init__(self:Any, config:ModelConfig, cache:BufferCache, tp_size:int=1) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_rotary_embedding(self:Any, seq_len:int, device:torch.device) -> Tuple[torch.Tensor, torch.Tensor]`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `rotate_half(self:Any, x:torch.Tensor) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `apply_rotary_pos_emb(self:Any, pos_sin:torch.Tensor, pos_cos:torch.Tensor, t:torch.Tensor) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, q:torch.Tensor, k:torch.Tensor, block_end_index:Optional[torch.Tensor]=None, start_pos:int=0) -> Tuple[torch.Tensor, torch.Tensor]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `Activation`
    Metodo: `__init__(self:Any, config:ModelConfig) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, x:torch.Tensor) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `output_multiplier(self:Any) -> float`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `build(cls:Any, config:ModelConfig) -> Activation`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
  - `GELU`
    Metodo: `output_multiplier(self:Any) -> float`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `ReLU`
    Metodo: `output_multiplier(self:Any) -> float`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `SiLU`
    Metodo: `output_multiplier(self:Any) -> float`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `SwiGLU`
    Metodo: `forward(self:Any, x:torch.Tensor) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `output_multiplier(self:Any) -> float`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LLaDABlock`
    Metodo: `__init__(self:Any, layer_id:int, config:ModelConfig, cache:BufferCache) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `reset_parameters(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `set_activation_checkpointing(self:Any, strategy:Optional[ActivationCheckpointingStrategy]) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_cast_attn_bias(cls:Any, bias:torch.Tensor, input_dtype:torch.dtype) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_scaled_dot_product_attention(self:Any, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, attn_mask:Optional[torch.Tensor]=None, dropout_p:float=0.0, is_causal:bool=False) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `attention(self:Any, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, mask:Optional[torch.Tensor]=None, attention_bias:Optional[torch.Tensor]=None, layer_past:Optional[Tuple[torch.Tensor, torch.Tensor]]=None, kv_cache:Optional[Cache]=None, use_cache:bool=False, replace_position:Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, x:torch.Tensor, attention_bias:Optional[torch.FloatTensor]=None, layer_past:Optional[Tuple[torch.Tensor, torch.Tensor]]=None, kv_cache:Any=None, use_cache:bool=False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `build(cls:Any, layer_id:int, config:ModelConfig, cache:BufferCache) -> LLaDABlock`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
  - `LLaDASequentialBlock`
    Metodo: `__init__(self:Any, layer_id:int, config:ModelConfig, cache:BufferCache) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `reset_parameters(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, x:torch.Tensor, attention_bias:Optional[torch.Tensor]=None, layer_past:Optional[Tuple[torch.Tensor, torch.Tensor]]=None, kv_cache:Optional[Cache]=None, use_cache:bool=False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LLaDALlamaBlock`
    Metodo: `__init__(self:Any, layer_id:int, config:ModelConfig, cache:BufferCache) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `reset_parameters(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, x:torch.Tensor, attention_bias:Optional[torch.Tensor]=None, layer_past:Optional[Tuple[torch.Tensor, torch.Tensor]]=None, kv_cache:Optional[Cache]=None, use_cache:bool=False, replace_position:Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LLaDABlockDiffBlock`
    Metodo: `__init__(self:Any, layer_id:int, config:ModelConfig, cache:BufferCache) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `reset_parameters(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `cross_attn_flex(self:Any, qkv:Any, mask:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, x:torch.Tensor, attention_bias:Optional[torch.Tensor]=None, layer_past:Optional[Tuple[torch.Tensor, torch.Tensor]]=None, use_cache:bool=False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LLaDAOutput`
  - `LLaDAGenerateOutput`
  - `LLaDABlockGroup`
    Metodo: `__init__(self:Any, config:ModelConfig, layer_offset:int, modules:Optional[Iterable[nn.Module]]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, x:torch.Tensor, attention_bias:Optional[torch.FloatTensor]=None, layers_past:Optional[List[Tuple[torch.Tensor, torch.Tensor]]]=None, kv_cache:Optional[Cache]=None, use_cache:bool=False) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `reset_parameters(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `set_activation_checkpointing(self:Any, strategy:Optional[ActivationCheckpointingStrategy]) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LLaDAModel`
    Metodo: `__init__(self:Any, config:ModelConfig, init_params:bool=True) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `set_activation_checkpointing(self:Any, strategy:Optional[ActivationCheckpointingStrategy]) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `device(self:Any) -> torch.device`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `reset_parameters(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_alibi_attention_bias(self:Any, seq_len:int, device:torch.device) -> torch.Tensor`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `forward(self:Any, input_ids:torch.LongTensor, input_embeddings:Optional[torch.FloatTensor]=None, attention_mask:Optional[torch.Tensor]=None, attention_bias:Optional[torch.Tensor]=None, past_key_values:Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]]=None, kv_cache:Optional[Cache]=None, use_cache:bool=False, last_logits_only:bool=False, output_hidden_states:Optional[bool]=None, replace_position:Optional[torch.Tensor]=None) -> LLaDAOutput`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LLaDAModelLM`
    Metodo: `__init__(self:Any, config:LLaDAConfig, model:Optional[LLaDAModel]=None, init_params:bool=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, input_ids:torch.LongTensor=None, inputs_embeds:Optional[torch.FloatTensor]=None, attention_mask:Optional[torch.Tensor]=None, attention_bias:Optional[torch.Tensor]=None, past_key_values:Optional[List[torch.FloatTensor]]=None, kv_cache:Optional[Cache]=None, labels:Optional[torch.LongTensor]=None, use_cache:Optional[bool]=None, output_attentions:Optional[bool]=None, output_hidden_states:Optional[bool]=None, return_dict:Optional[bool]=None, replace_position:Optional[torch.Tensor]=None) -> Union[Tuple, CausalLMOutputWithPast]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `can_generate(self:Any) -> bool`
    Descrizione: Genera output testo o sequenze.
    Metodo: `prepare_inputs_for_generation(self:Any, input_ids:torch.LongTensor, past_key_values:Optional[List[Tuple]]=None, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_input_embeddings(self:Any) -> torch.nn.Module`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `set_input_embeddings(self:Any, value:torch.nn.Module) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_output_embeddings(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `set_output_embeddings(self:Any, value:torch.nn.Module) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `tie_weights(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `init_h2e_module(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `tensor_parallel(self:Any, tp_size:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\dinfer\python\dinfer\model\modeling_llada2_moe.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `torch_all_reduce(tensor:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `roll_tensor(tensor:Any, shifts:Any=-1, dims:Any=-1, fill_value:Any=0) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `replace_linear_class(linear:nn.Linear, style:Literal['colwise', 'rowwise', 'qkv'], quant_config:Any, model_config:Any) -> Union[ColumnParallelLinear, RowParallelLinear]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_all_gather_cat(tensor:torch.Tensor, dim:int=1, group:Optional[dist.ProcessGroup]=None, normal_len:int=0, last_len:int=0) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_get_unpad_data(attention_mask:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_expand_mask(mask:torch.Tensor, dtype:torch.dtype, tgt_len:Optional[int]=None) -> Any`
    Descrizione: Gestisce logica di mascheratura/filtri di posizioni.
  - `_make_causal_mask(input_ids_shape:torch.Size, dtype:torch.dtype, device:torch.device, past_key_values_length:int=0) -> Any`
    Descrizione: Gestisce logica di mascheratura/filtri di posizioni.
  - `rotate_half(x:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `apply_rotary_pos_emb(q:Any, k:Any, cos:Any, sin:Any, unsqueeze_dim:Any=1) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `static_routing_function(gate:Any, hidden_states:Any, gating_output:Any, topk:Any, renormalize:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `repeat_kv(hidden_states:torch.Tensor, n_rep:int) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `H2Embed`
    Metodo: `__init__(self:Any, embedding:nn.Embedding, tau:float=1.0) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__call__(self:Any, x:torch.Tensor, mask_index:Optional[torch.Tensor]=None, logits:Optional[torch.Tensor]=None, iter_cont_weight:float=0.0) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `MoEV2CausalLMOutputWithPast`
  - `MoeV2ModelOutputWithPast`
    Metodo: `__init__(self:Any, mtp_hidden_states:Any=None, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LLaDA2MoeRMSNorm`
    Metodo: `__init__(self:Any, hidden_size:Any, eps:Any=1e-06) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, hidden_states:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LLaDA2MoeRotaryEmbedding`
    Metodo: `__init__(self:Any, config:LLaDA2MoeConfig, device:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, x:Any, position_ids:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LLaDA2MoeMLP`
    Metodo: `__init__(self:Any, config:LLaDA2MoeConfig, intermediate_size:int) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, x:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LLaDA2MoeGate`
    Metodo: `__init__(self:Any, config:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `reset_parameters(self:Any) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `group_limited_topk(self:Any, scores:torch.Tensor) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, hidden_states:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_logits(self:Any, hidden_states:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `routing(self:Any, hidden_states:Any, gating_output:Any, topk:Any, renormalize:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LLaDA2MoeSparseMoeBlock`
    Metodo: `__init__(self:Any, config:Any, prefix:str='') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, hidden_states:torch.Tensor) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LLaDA2MoeAttention`
    Metodo: `__init__(self:Any, config:LLaDA2MoeConfig, layer_idx:Optional[int]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_shape(self:Any, tensor:torch.Tensor, seq_len:int, bsz:int) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, hidden_states:torch.Tensor, attention_mask:Optional[torch.Tensor]=None, position_ids:Optional[torch.LongTensor]=None, past_key_value:Optional[Cache]=None, output_attentions:bool=False, use_cache:bool=False, position_embeddings:Optional[Tuple[torch.Tensor, torch.Tensor]]=None, **kwargs:Any) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LLaDA2MoeFlashAttention2`
    Metodo: `__init__(self:Any, *args:Any, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, hidden_states:torch.Tensor, attention_mask:Optional[torch.LongTensor]=None, position_ids:Optional[torch.LongTensor]=None, past_key_value:Optional[Cache]=None, output_attentions:bool=False, use_cache:bool=False, position_embeddings:Optional[Tuple[torch.Tensor, torch.Tensor]]=None, **kwargs:Any) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_flash_attention_forward(self:Any, query_states:Any, key_states:Any, value_states:Any, attention_mask:Any, query_length:Any, dropout:Any=0.0, softmax_scale:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_upad_input(self:Any, query_layer:Any, key_layer:Any, value_layer:Any, attention_mask:Any, query_length:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LLaDA2MoeSdpaAttention`
    Metodo: `forward(self:Any, hidden_states:torch.Tensor, attention_mask:Optional[torch.Tensor]=None, position_ids:Optional[torch.LongTensor]=None, past_key_value:Optional[Cache]=None, output_attentions:bool=False, use_cache:bool=False, position_embeddings:Optional[Tuple[torch.Tensor, torch.Tensor]]=None, cache_position:Optional[torch.LongTensor]=None, replace_position:Any=None, **kwargs:Any) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LLaDA2MoeMTPLayer`
    Metodo: `__init__(self:Any, config:LLaDA2MoeConfig, layer_idx:int) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, input_embeds:Any, hidden_states:torch.Tensor, attention_mask:Optional[torch.Tensor]=None, position_ids:Optional[torch.LongTensor]=None, past_key_value:Optional[Tuple[torch.Tensor]]=None, output_attentions:Optional[bool]=False, output_router_logits:Optional[bool]=False, use_cache:Optional[bool]=False, position_embeddings:Optional[Tuple[torch.Tensor, torch.Tensor]]=None, **kwargs:Any) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LLaDA2MoeDecoderLayer`
    Metodo: `__init__(self:Any, config:LLaDA2MoeConfig, layer_idx:int) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, hidden_states:torch.Tensor, attention_mask:Optional[torch.Tensor]=None, position_ids:Optional[torch.LongTensor]=None, past_key_value:Optional[Cache]=None, output_attentions:Optional[bool]=False, output_router_logits:Optional[bool]=False, use_cache:Optional[bool]=False, cache_position:Optional[torch.LongTensor]=None, replace_position:Optional[torch.LongTensor]=None, position_embeddings:Optional[Tuple[torch.Tensor, torch.Tensor]]=None, **kwargs:Any) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LLaDA2MoePreTrainedModel`
    Metodo: `_init_weights(self:Any, module:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LLaDA2MoeModel`
    Metodo: `__init__(self:Any, config:LLaDA2MoeConfig) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_input_embeddings(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `set_input_embeddings(self:Any, value:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, input_ids:torch.LongTensor=None, attention_mask:Optional[torch.Tensor]=None, position_ids:Optional[torch.LongTensor]=None, past_key_values:Optional[Cache]=None, inputs_embeds:Optional[torch.FloatTensor]=None, use_cache:Optional[bool]=None, output_attentions:Optional[bool]=None, output_hidden_states:Optional[bool]=None, output_router_logits:Optional[bool]=None, return_dict:Optional[bool]=None, cache_position:Optional[torch.LongTensor]=None, replace_position:Optional[torch.LongTensor]=None, **kwargs:Any) -> Union[Tuple, MoeV2ModelOutputWithPast]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LLaDA2MoeModelLM`
    Metodo: `__init__(self:Any, config:LLaDA2MoeConfig) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_input_embeddings(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `set_input_embeddings(self:Any, value:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_output_embeddings(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `set_output_embeddings(self:Any, new_embeddings:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `set_decoder(self:Any, decoder:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_decoder(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `load_state_dict(self:Any, model_dir:Any, strict:Any=True, dtype:Any=torch.bfloat16, device:Any=None) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
    Metodo: `load_sharded_safetensors(self:Any, model_dir:Any) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
    Metodo: `init_h2e_module(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `load_weights(self:Any, model_path:Any, torch_dtype:Any=torch.bfloat16, device:Any=None) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
    Metodo: `forward(self:Any, input_ids:torch.LongTensor=None, attention_mask:Optional[torch.Tensor]=None, position_ids:Optional[torch.LongTensor]=None, replace_position:Optional[torch.LongTensor]=None, past_key_values:Optional[KVCache]=None, inputs_embeds:Optional[torch.FloatTensor]=None, labels:Optional[torch.LongTensor]=None, use_cache:Optional[bool]=None, output_attentions:Optional[bool]=None, output_hidden_states:Optional[bool]=None, output_router_logits:Optional[bool]=None, return_dict:Optional[bool]=None, num_logits_to_keep:int=0, **kwargs:Any) -> Union[Tuple, MoEV2CausalLMOutputWithPast]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `tensor_parallel(self:Any, tp_size:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\dinfer\python\dinfer\model\modeling_llada2_moe_sglang.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `torch_all_reduce(tensor:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `torch_all_gather(input_:torch.Tensor) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_all_gather_cat(tensor:torch.Tensor, dim:int=1, group:Optional[dist.ProcessGroup]=None, normal_len:int=0, last_len:int=0) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `repeat_kv(hidden_states:torch.Tensor, n_rep:int) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `RMSNorm`
    Metodo: `__init__(self:Any, hidden_size:int, eps:float=1e-06, var_hidden_size:Optional[int]=None) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, x:torch.Tensor, residual:Optional[torch.Tensor]=None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LLaDA2RMSNorm`
    Metodo: `__init__(self:Any, hidden_size:Any, eps:Any=1e-06) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, hidden_states:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `H2Embed`
    Metodo: `__init__(self:Any, embedding:nn.Embedding, tau:float=1.0) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__call__(self:Any, x:torch.Tensor, mask_index:Optional[torch.Tensor]=None, logits:Optional[torch.Tensor]=None, iter_cont_weight:float=0.0) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LLaDA2MLP`
    Metodo: `__init__(self:Any, intermediate_size:int, config:PretrainedConfig, quant_config:Optional[QuantizationConfig]=None, reduce_results:Optional[bool]=True, prefix:str='', tp_rank:Optional[int]=None, tp_size:Optional[int]=None) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, hidden_states:torch.Tensor, forward_batch:Optional[ForwardBatch]=None, use_reduce_scatter:bool=False) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LLaDA2Gate`
    Metodo: `__init__(self:Any, config:Any, params_dtype:Optional[torch.dtype]=None, prefix:str='') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, hidden_states:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LLaDA2SparseMoeBlock`
    Metodo: `__init__(self:Any, layer_id:int, config:PretrainedConfig, quant_config:Optional[QuantizationConfig]=None, alt_stream:Optional[torch.cuda.Stream]=None, prefix:str='') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, hidden_states:torch.Tensor, forward_batch:Optional[ForwardBatch]=None, use_reduce_scatter:bool=False) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_moe_weights(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `_forward_shared_experts(self:Any, hidden_states:torch.Tensor) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_forward_router_experts(self:Any, hidden_states:torch.Tensor) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward_normal_dual_stream(self:Any, hidden_states:torch.Tensor) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward_normal(self:Any, hidden_states:torch.Tensor, use_reduce_scatter:bool=False) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward_deepep(self:Any, hidden_states:torch.Tensor, forward_batch:ForwardBatch) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LLaDA2Attention`
    Metodo: `__init__(self:Any, config:PretrainedConfig, layer_id:int=0, quant_config:Optional[QuantizationConfig]=None, reduce_results:bool=True, prefix:str='', alt_stream:Optional[torch.cuda.Stream]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_apply_q_norm(self:Any, q:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_apply_k_norm(self:Any, k:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_apply_qk_norm(self:Any, q:torch.Tensor, k:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_apply_repeat(self:Any, k:Any, v:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, positions:torch.Tensor, hidden_states:torch.Tensor, past_key_values:Any=None, replace_position:Optional[torch.LongTensor]=None, use_cache:Optional[bool]=None, attention_mask:Optional[torch.Tensor]=None) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LLaDA2Block`
    Metodo: `__init__(self:Any, config:PretrainedConfig, layer_id:int=0, quant_config:Optional[QuantizationConfig]=None, prefix:str='', alt_stream:Optional[torch.cuda.Stream]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_is_layer_sparse(self:Any, config:PretrainedConfig, layer_id:int, is_nextn:bool) -> bool`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, positions:torch.Tensor, hidden_states:torch.Tensor, residual:Optional[torch.Tensor], past_key_values:Any=None, replace_position:Optional[torch.LongTensor]=None, use_cache:Optional[bool]=None, attention_mask:Optional[torch.Tensor]=None) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LLaDA2Model`
    Metodo: `__init__(self:Any, config:PretrainedConfig, quant_config:Optional[QuantizationConfig]=None, alt_stream:Optional[torch.cuda.Stream]=None, prefix:str='.') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, input_ids:torch.Tensor, positions:torch.Tensor, past_key_values:Any=None, input_embeds:torch.Tensor=None, pp_proxy_tensors:Optional[PPProxyTensors]=None, replace_position:Optional[torch.LongTensor]=None, use_cache:Optional[bool]=None, attention_mask:Optional[torch.Tensor]=None) -> Union[torch.Tensor, PPProxyTensors]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LLaDA2SGLangLM`
    Metodo: `__init__(self:Any, config:PretrainedConfig, quant_config:Optional[QuantizationConfig]=None, prefix:str='', expert_map_path:str='') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `start_layer(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `end_layer(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_embed_and_head(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `set_embed_and_head(self:Any, embed:Any, head:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, input_ids:torch.Tensor=None, position_ids:torch.Tensor=None, inputs_embeds:torch.Tensor=None, pp_proxy_tensors:Optional[PPProxyTensors]=None, past_key_values:Any=None, replace_position:Optional[torch.LongTensor]=None, use_cache:Optional[bool]=None, attention_mask:Optional[torch.Tensor]=None) -> MoeCausalLMOutputWithPast`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `apply_state_dicts(self:Any, weights:Iterable[Tuple[str, torch.Tensor]], is_nextn:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_update_state_dict_for_fusemoe_quant(self:Any, state_dict:Any, num_layers:Any, dtype:Any, per_gpu_expert_mapping:Any, per_gpu_inverse_mapping:Any, device:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_update_state_dict_for_fusemoe(self:Any, state_dict:Any, num_layers:Any, dtype:Any, per_gpu_expert_mapping:Any, per_gpu_inverse_mapping:Any, device:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_tp_split(self:Any, tensor:torch.Tensor, dim:int, rank:int, world:int, is_w13:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `load_state_dict(self:Any, model_dir:Any, strict:Any=True, dtype:Any=torch.bfloat16, device:Any=None) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
    Metodo: `init_h2e_module(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `load_weights(self:Any, model_path:Any, torch_dtype:Any=torch.bfloat16, device:Any=None) -> Any`
    Descrizione: Carica dati o stato da sorgente esterna/file.
    Metodo: `get_model_config_for_expert_location(cls:Any, config:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `after_loading(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `after_processing(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\dinfer\python\dinfer\model\modeling_llada_fastdllm.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `replace_linear_class(linear:nn.Linear, style:Literal['colwise', 'rowwise'], quant_config:Any) -> Union[ColumnParallelLinear, RowParallelLinear]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `scaled_dot_product_attention(q:Any, k:Any, v:Any, mask:Any=None, attn_mask:Any=None, dropout_p:Any=0.0, is_causal:Any=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `init_weights(config:ModelConfig, module:Union[nn.Linear, nn.Embedding], d:Optional[int]=None, layer_id:Optional[int]=None, std_factor:float=1.0, type_of_module:Optional[ModuleType]=None) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `ensure_finite_(x:torch.Tensor, check_neg_inf:bool=True, check_pos_inf:bool=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `activation_checkpoint_function(cfg:ModelConfig) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `_non_meta_init_device(config:ModelConfig) -> torch.device`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `causal_attention_bias(seq_len:int, device:torch.device) -> torch.FloatTensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_causal_attention_bias(cache:BufferCache, seq_len:int, device:torch.device) -> torch.Tensor`
    Descrizione: Recupera valore/stato calcolato.
  - `alibi_attention_bias(seq_len:int, config:ModelConfig, device:torch.device) -> torch.FloatTensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `create_model_config_from_pretrained_config(config:LLaDAConfig) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
  Classi:
  - `ModuleType`
  - `BufferCache`
  - `Dropout`
    Metodo: `forward(self:Any, input:torch.Tensor) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LayerNormBase`
    Metodo: `__init__(self:Any, config:ModelConfig, size:Optional[int]=None, elementwise_affine:Optional[bool]=True, eps:float=1e-05) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, x:torch.Tensor) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `build(cls:Any, config:ModelConfig, size:Optional[int]=None, **kwargs:Any) -> LayerNormBase`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
    Metodo: `_cast_if_autocast_enabled(self:Any, tensor:torch.Tensor, dtype:Optional[torch.dtype]=None) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `reset_parameters(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LayerNorm`
    Metodo: `__init__(self:Any, config:ModelConfig, size:Optional[int]=None, low_precision:bool=False, elementwise_affine:Optional[bool]=None, eps:float=1e-05) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, x:torch.Tensor) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `RMSLayerNorm`
    Metodo: `__init__(self:Any, config:ModelConfig, size:Optional[int]=None, elementwise_affine:Optional[bool]=None, eps:float=1e-05) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, x:torch.Tensor) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `GemmaRMSLayerNorm`
    Metodo: `__init__(self:Any, config:ModelConfig, size:Optional[int]=None, elementwise_affine:Optional[bool]=None, eps:float=1e-05) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, x:torch.Tensor) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `RotaryEmbedding`
    Metodo: `__init__(self:Any, config:ModelConfig, cache:BufferCache) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_rotary_embedding(self:Any, seq_len:int, device:torch.device) -> Tuple[torch.Tensor, torch.Tensor]`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `rotate_half(self:Any, x:torch.Tensor) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `apply_rotary_pos_emb(self:Any, pos_sin:torch.Tensor, pos_cos:torch.Tensor, t:torch.Tensor) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, q:torch.Tensor, k:torch.Tensor, block_end_index:Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `Activation`
    Metodo: `__init__(self:Any, config:ModelConfig) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, x:torch.Tensor) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `output_multiplier(self:Any) -> float`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `build(cls:Any, config:ModelConfig) -> Activation`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
  - `GELU`
    Metodo: `output_multiplier(self:Any) -> float`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `ReLU`
    Metodo: `output_multiplier(self:Any) -> float`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `SiLU`
    Metodo: `output_multiplier(self:Any) -> float`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `SwiGLU`
    Metodo: `forward(self:Any, x:torch.Tensor) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `output_multiplier(self:Any) -> float`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LLaDABlock`
    Metodo: `__init__(self:Any, layer_id:int, config:ModelConfig, cache:BufferCache) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `reset_parameters(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `set_activation_checkpointing(self:Any, strategy:Optional[ActivationCheckpointingStrategy]) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_cast_attn_bias(cls:Any, bias:torch.Tensor, input_dtype:torch.dtype) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_scaled_dot_product_attention(self:Any, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, attn_mask:Optional[torch.Tensor]=None, dropout_p:float=0.0, is_causal:bool=False) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `attention(self:Any, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, mask:Optional[torch.Tensor]=None, attention_bias:Optional[torch.Tensor]=None, layer_past:Optional[Tuple[torch.Tensor, torch.Tensor]]=None, use_cache:bool=False, replace_position:Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, x:torch.Tensor, attention_bias:Optional[torch.FloatTensor]=None, layer_past:Optional[Tuple[torch.Tensor, torch.Tensor]]=None, use_cache:bool=False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `build(cls:Any, layer_id:int, config:ModelConfig, cache:BufferCache) -> LLaDABlock`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
  - `LLaDASequentialBlock`
    Metodo: `__init__(self:Any, layer_id:int, config:ModelConfig, cache:BufferCache) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `reset_parameters(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, x:torch.Tensor, attention_bias:Optional[torch.Tensor]=None, layer_past:Optional[Tuple[torch.Tensor, torch.Tensor]]=None, use_cache:bool=False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LLaDALlamaBlock`
    Metodo: `__init__(self:Any, layer_id:int, config:ModelConfig, cache:BufferCache) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `reset_parameters(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, x:torch.Tensor, attention_bias:Optional[torch.Tensor]=None, layer_past:Optional[Tuple[torch.Tensor, torch.Tensor]]=None, use_cache:bool=False, replace_position:Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LLaDABlockDiffBlock`
    Metodo: `__init__(self:Any, layer_id:int, config:ModelConfig, cache:BufferCache) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `reset_parameters(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `cross_attn_flex(self:Any, qkv:Any, mask:Any=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, x:torch.Tensor, attention_bias:Optional[torch.Tensor]=None, layer_past:Optional[Tuple[torch.Tensor, torch.Tensor]]=None, use_cache:bool=False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LLaDAOutput`
  - `LLaDAGenerateOutput`
  - `LLaDABlockGroup`
    Metodo: `__init__(self:Any, config:ModelConfig, layer_offset:int, modules:Optional[Iterable[nn.Module]]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, x:torch.Tensor, attention_bias:Optional[torch.FloatTensor]=None, layers_past:Optional[List[Tuple[torch.Tensor, torch.Tensor]]]=None, use_cache:bool=False) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `reset_parameters(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `set_activation_checkpointing(self:Any, strategy:Optional[ActivationCheckpointingStrategy]) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LLaDAModel`
    Metodo: `__init__(self:Any, config:ModelConfig, init_params:bool=True) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `set_activation_checkpointing(self:Any, strategy:Optional[ActivationCheckpointingStrategy]) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `device(self:Any) -> torch.device`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `reset_parameters(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_alibi_attention_bias(self:Any, seq_len:int, device:torch.device) -> torch.Tensor`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `forward(self:Any, input_ids:torch.LongTensor, input_embeddings:Optional[torch.FloatTensor]=None, attention_mask:Optional[torch.Tensor]=None, attention_bias:Optional[torch.Tensor]=None, past_key_values:Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]]=None, use_cache:bool=False, last_logits_only:bool=False, output_hidden_states:Optional[bool]=None, replace_position:Optional[torch.Tensor]=None) -> LLaDAOutput`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LLaDAModelLM`
    Metodo: `__init__(self:Any, config:LLaDAConfig, model:Optional[LLaDAModel]=None, init_params:bool=False) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, input_ids:torch.LongTensor=None, inputs_embeds:Optional[torch.FloatTensor]=None, attention_mask:Optional[torch.Tensor]=None, attention_bias:Optional[torch.Tensor]=None, past_key_values:Optional[List[torch.FloatTensor]]=None, labels:Optional[torch.LongTensor]=None, use_cache:Optional[bool]=None, output_attentions:Optional[bool]=None, output_hidden_states:Optional[bool]=None, return_dict:Optional[bool]=None, replace_position:Optional[torch.Tensor]=None) -> Union[Tuple, CausalLMOutputWithPast]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `can_generate(self:Any) -> bool`
    Descrizione: Genera output testo o sequenze.
    Metodo: `prepare_inputs_for_generation(self:Any, input_ids:torch.LongTensor, past_key_values:Optional[List[Tuple]]=None, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_input_embeddings(self:Any) -> torch.nn.Module`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `set_input_embeddings(self:Any, value:torch.nn.Module) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_output_embeddings(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `set_output_embeddings(self:Any, value:torch.nn.Module) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `tie_weights(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `tensor_parallel(self:Any, tp_size:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\dinfer\python\dinfer\model\tp_linear.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `tensor_model_parallel_all_gather(tensor:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `tensor_model_parallel_all_reduce(tensor:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `set_weight_attrs(weight:torch.Tensor, weight_attrs:Optional[dict[str, Any]]) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `divide(numerator:Any, denominator:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `split_tensor_along_last_dim(tensor:torch.Tensor, num_partitions:int, contiguous_split_chunks:bool=False) -> Sequence[torch.Tensor]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `adjust_bitblas_shard(param:Any, shard_size:Any, shard_offset:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `adjust_marlin_shard(param:Any, shard_size:Any, shard_offset:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `adjust_bitsandbytes_4bit_shard(param:Parameter, shard_offsets:dict[str, tuple[int, int]], loaded_shard_id:str) -> tuple[int, int]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `adjust_scalar_to_fused_array(param:Any, loaded_weight:Any, shard_id:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `left_shift_bitsandbytes_4bit_shard(bnb_weight_attrs:dict[str, Any]) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `UnquantizedLinearMethod`
    Metodo: `create_weights(self:Any, layer:torch.nn.Module, input_size_per_partition:int, output_partition_sizes:list[int], input_size:int, output_size:int, params_dtype:torch.dtype, **extra_weight_attrs:Any) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.
    Metodo: `apply(self:Any, layer:torch.nn.Module, x:torch.Tensor, bias:Optional[torch.Tensor]=None) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LinearBase`
    Metodo: `__init__(self:Any, input_size:int, output_size:int, skip_bias_add:bool=False, params_dtype:Optional[torch.dtype]=None, prefix:str='', return_bias:bool=True) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, x:torch.Tensor) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `ReplicatedLinear`
    Metodo: `__init__(self:Any, input_size:int, output_size:int, bias:bool=True, skip_bias_add:bool=False, params_dtype:Optional[torch.dtype]=None, prefix:str='', tp_rank:Any=0, tp_size:Any=1, return_bias:bool=True) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `weight_loader(self:Any, param:Parameter, loaded_weight:torch.Tensor) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, x:torch.Tensor) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `extra_repr(self:Any) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `ColumnParallelLinear`
    Metodo: `__init__(self:Any, input_size:int, output_size:int, bias:bool=True, gather_output:bool=False, skip_bias_add:bool=False, params_dtype:Optional[torch.dtype]=None, output_sizes:Optional[list[int]]=None, prefix:str='', tp_rank:Any=0, tp_size:Any=1, return_bias:bool=True) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `weight_loader(self:Any, param:Parameter, loaded_weight:torch.Tensor) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, input_:Any) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `extra_repr(self:Any) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `RowParallelLinear`
    Metodo: `__init__(self:Any, input_size:int, output_size:int, bias:bool=True, input_is_parallel:bool=True, skip_bias_add:bool=False, params_dtype:Optional[torch.dtype]=None, reduce_results:bool=True, prefix:str='', tp_rank:Any=0, tp_size:Any=1, return_bias:bool=True) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `weight_loader(self:Any, param:Parameter, loaded_weight:torch.Tensor) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, input_:Any) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `extra_repr(self:Any) -> str`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\dinfer\setup.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `read(*names:Any, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `find_version(*file_paths:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\dinfer\tests\test.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `test_block_iterator() -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `test_token_array() -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `get_prompts(tokenizer:Any, mask_id:Any, device:Any, num:Any=1) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `check_iteration(model:Any, decoder:Any, input_ids:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `test_moe_diffusion() -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `test_diffusion() -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `setup_distributed(rank:Any, world_size:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `check_worker(rank:Any, world_size:Any, gpu:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `test_dist() -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `check_diffusion_worker(rank:Any, world_size:Any, gpu:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `test_diffusion_sp() -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `test_moe_server(require_init:Any=True) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `test_server() -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  Classi:
  - `SimulateBlockIterator`
    Metodo: `__init__(self:Any, x:Any, block_length:Any, mask_id:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__iter__(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `move_next(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__next__(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `SimulateBlockIteratorFactory`
    Metodo: `create(self:Any, x:Any, block_length:Any) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.

- File: `vendor\dinfer\tests\test_bd.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `init_vllm_dist(worker_id:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `run_bd(use_kvcache:Any) -> Any`
    Descrizione: Esegue pipeline o job completo.
  - `batchinfer_diverse_length() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `test_bd() -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `vendor\dinfer\tests\test_bd_serving.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `generate_test(*args:Any, tout:Any=10, **kwargs:Any) -> Any`
    Descrizione: Genera output testo o sequenze.
  - `init_sglang_dist() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `run_bd(use_kvcache:Any) -> Any`
    Descrizione: Esegue pipeline o job completo.
  - `run_bd_serving(use_kvcache:Any) -> Any`
    Descrizione: Esegue pipeline o job completo.
  - `run_bd_serving_error(use_kvcache:Any) -> Any`
    Descrizione: Esegue pipeline o job completo.
  - `test_bd() -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `vendor\dinfer\tests\test_bd_serving_batching.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `test_bd_tpep() -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `vendor\dinfer\tests\test_bd_serving_tpep.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `test_bd_tpep() -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `vendor\dinfer\tests\test_generate.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `IterSmoothDiffusionLLM`
    Metodo: `__init__(self:Any, model:Any, decoder:Any, iterator_factory:Any, early_stop:Any=True, cache_factory:Any=None, maximum_unroll:Any=4, expected_tpf:Any=8, cont_weight:Any=0.3, cont_weight_init:Any=0.15, cont_weight_growth:Any=0.02, threshold_decay:Any=0.02) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, prompt:Any, gen_length:Any=128, block_length:Any=128) -> Any`
    Descrizione: Genera output testo o sequenze.
  - `IterSmoothWithVicinityCacheDiffusionLLM`
    Metodo: `__init__(self:Any, model:Any, decoder:Any, iterator_factory:Any, cache_factory:Any, maximum_unroll:Any=4, expected_tpf:Any=8, prefix_look:Any=0, after_look:Any=0, warmup_steps:Any=0, early_stop:Any=True, cont_weight:Any=0.3, cont_weight_init:Any=0.15, cont_weight_growth:Any=0.02, threshold_decay:Any=0.02) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `generate(self:Any, prompt:Any, gen_length:Any=128, block_length:Any=128) -> Any`
    Descrizione: Genera output testo o sequenze.

- File: `vendor\dinfer\tests\test_llada.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_prompts(tokenizer:Any, mask_id:Any, device:Any, num:Any=1) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `test_sw_dual_cache() -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `test_prefix_cache() -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `test_dual_cache() -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `test_itersmooth() -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  Classi:
  - `SimulateBlockIterator`
    Metodo: `__init__(self:Any, x:Any, block_length:Any, mask_id:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__iter__(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `move_next(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `__next__(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `SimulateBlockIteratorFactory`
    Metodo: `create(self:Any, x:Any, block_length:Any) -> Any`
    Descrizione: Costruisce oggetto/struttura derivata dai parametri.

- File: `vendor\dinfer\tests\test_llada_moe.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_prompts(tokenizer:Any, mask_id:Any, device:Any, num:Any=1) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `init_vllm_dist(worker_id:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `test_llada_moe_hierarchy() -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `test_llada_moe_blockwise() -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `test_llada_moe_batching() -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `test_llada_moe_itersmooth() -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `test_llada_moe_dual_cache() -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `test_llada_moe_dual_cache_batching() -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `test_llada_moe_itersmooth_cache() -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `test_llada_moe_itersmooth_vicinity_cache() -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `vendor\dinfer\tests\test_serving.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_prompts(tokenizer:Any, mask_id:Any, device:Any, num:Any=1) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `setup_llada_reference() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `setup_moe_reference() -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `test_llada_server(setup_llada_reference:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `test_moe_server(setup_moe_reference:Any) -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `vendor\dinfer\tests\test_serving_sglang.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `get_prompts(tokenizer:Any, mask_id:Any, device:Any, num:Any=1) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `get_reference_response(master_port:Any, input_ids:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `test_server_sglang() -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `vendor\dinfer\tests\test_wo_model.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `test_block_iterator() -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `test_token_array() -> Any`
    Descrizione: Test automatico di regressione/comportamento.
  - `setup_distributed(rank:Any, world_size:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `check_worker(rank:Any, world_size:Any, gpu:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `test_dist() -> Any`
    Descrizione: Test automatico di regressione/comportamento.

- File: `vendor\dinfer\tools\__init__.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Nessuna funzione/classe top-level.

- File: `vendor\dinfer\tools\configuration_lladamoe.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Classi:
  - `LLaDAConfig`
    Metodo: `__init__(self:Any, vocab_size:Any=-1, hidden_size:Any=-1, dense_intermediate_size:Any=-1, expert_intermediate_size:Any=-1, shared_expert_intermediate_size:Any=-1, num_hidden_layers:Any=-1, num_attention_heads:Any=-1, num_key_value_heads:Any=None, hidden_act:Any='silu', max_position_embeddings:Any=4096, initializer_range:Any=0.02, rms_norm_eps:Any=1e-05, use_cache:Any=False, pad_token_id:Any=1, bos_token_id:Any=None, eos_token_id:Any=50279, tie_word_embeddings:Any=False, rope_theta:Any=-1, partial_rotary_factor:Any=-1, rope_scaling:Any=None, attention_bias:Any=False, attention_dropout:Any=0.0, clip_qkv:Any=None, num_experts_per_tok:Any=-1, num_experts:Any=-1, output_router_logits:Any=False, router_aux_loss_coef:Any=0.01, norm_topk_prob:Any=None, qk_layernorm:Any=None, moe_layer_freq:Any=[], moe_router_enable_expert_bias:Any=None, moe_router_score_function:Any=None, routed_scaling_factor:Any=1, router_num_group:Any=-2, router_topk_group:Any=-2, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\dinfer\tools\fuse_moe.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `fused_moe_kernel_gptq_awq(a_ptr:Any, b_ptr:Any, c_ptr:Any, b_scale_ptr:Any, b_zp_ptr:Any, topk_weights_ptr:Any, sorted_token_ids_ptr:Any, expert_ids_ptr:Any, num_tokens_post_padded_ptr:Any, N:tl.constexpr, K:tl.constexpr, EM:Any, num_valid_tokens:Any, stride_am:Any, stride_ak:Any, stride_be:Any, stride_bk:Any, stride_bn:Any, stride_cm:Any, stride_cn:Any, stride_bse:Any, stride_bsk:Any, stride_bsn:Any, stride_bze:Any, stride_bzk:Any, stride_bzn:Any, block_k_diviable:tl.constexpr, group_size:tl.constexpr, BLOCK_SIZE_M:tl.constexpr, BLOCK_SIZE_N:tl.constexpr, BLOCK_SIZE_K:tl.constexpr, GROUP_SIZE_M:tl.constexpr, MUL_ROUTED_WEIGHT:tl.constexpr, top_k:tl.constexpr, compute_type:tl.constexpr, has_zp:tl.constexpr, use_int4_w4a16:tl.constexpr, use_int8_w8a16:tl.constexpr) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `fused_moe_kernel(a_ptr:Any, b_ptr:Any, c_ptr:Any, a_scale_ptr:Any, b_scale_ptr:Any, topk_weights_ptr:Any, sorted_token_ids_ptr:Any, expert_ids_ptr:Any, num_tokens_post_padded_ptr:Any, N:Any, K:Any, EM:Any, num_valid_tokens:Any, stride_am:Any, stride_ak:Any, stride_be:Any, stride_bk:Any, stride_bn:Any, stride_cm:Any, stride_cn:Any, stride_asm:Any, stride_ask:Any, stride_bse:Any, stride_bsk:Any, stride_bsn:Any, group_n:tl.constexpr, group_k:tl.constexpr, BLOCK_SIZE_M:tl.constexpr, BLOCK_SIZE_N:tl.constexpr, BLOCK_SIZE_K:tl.constexpr, GROUP_SIZE_M:tl.constexpr, MUL_ROUTED_WEIGHT:tl.constexpr, top_k:tl.constexpr, compute_type:tl.constexpr, use_fp8_w8a8:tl.constexpr, use_int8_w8a16:tl.constexpr) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `ceil_div(a:Any, b:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `moe_align_block_size_stage1(topk_ids_ptr:Any, tokens_cnts_ptr:Any, num_experts:tl.constexpr, numel:tl.constexpr, tokens_per_thread:tl.constexpr) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `moe_align_block_size_stage2(tokens_cnts_ptr:Any, num_experts:tl.constexpr) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `moe_align_block_size_stage3(total_tokens_post_pad_ptr:Any, tokens_cnts_ptr:Any, cumsum_ptr:Any, num_experts:tl.constexpr, block_size:tl.constexpr) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `moe_align_block_size_stage4(topk_ids_ptr:Any, sorted_token_ids_ptr:Any, expert_ids_ptr:Any, tokens_cnts_ptr:Any, cumsum_ptr:Any, num_experts:tl.constexpr, block_size:tl.constexpr, numel:tl.constexpr, tokens_per_thread:tl.constexpr) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `moe_align_block_size_triton(topk_ids:torch.Tensor, num_experts:int, block_size:int, sorted_token_ids:torch.Tensor, expert_ids:torch.Tensor, num_tokens_post_pad:torch.Tensor) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `moe_align_block_size(topk_ids:torch.Tensor, block_size:int, num_experts:int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `invoke_fused_moe_kernel(A:torch.Tensor, B:torch.Tensor, C:torch.Tensor, A_scale:Optional[torch.Tensor], B_scale:Optional[torch.Tensor], B_zp:Optional[torch.Tensor], topk_weights:torch.Tensor, topk_ids:torch.Tensor, sorted_token_ids:torch.Tensor, expert_ids:torch.Tensor, num_tokens_post_padded:torch.Tensor, mul_routed_weight:bool, top_k:int, config:Dict[str, Any], compute_type:tl.dtype, use_fp8_w8a8:bool, use_int8_w8a16:bool, use_int4_w4a16:bool, block_shape:Optional[List[int]]=None) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_config_file_name(E:int, N:int, dtype:Optional[str], block_shape:Optional[List[int]]=None) -> str`
    Descrizione: Recupera valore/stato calcolato.
  - `get_moe_configs(E:int, N:int, dtype:Optional[str], block_n:Optional[int]=None, block_k:Optional[int]=None) -> Optional[Dict[int, Any]]`
    Descrizione: Recupera valore/stato calcolato.
  - `get_default_config(M:int, E:int, N:int, K:int, topk:int, dtype:Optional[str], is_marlin:bool, block_shape:Optional[List[int]]=None) -> Dict[str, int]`
    Descrizione: Recupera valore/stato calcolato.
  - `try_get_optimal_moe_config(w1_shape:Tuple[int, ...], w2_shape:Tuple[int, ...], top_k:int, dtype:Optional[str], M:int, is_marlin:bool=False, block_shape:Optional[List[int]]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `fused_topk(hidden_states:torch.Tensor, gating_output:torch.Tensor, topk:int, renormalize:bool) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `grouped_topk(hidden_states:torch.Tensor, gating_output:torch.Tensor, topk:int, renormalize:bool, num_expert_group:int=0, topk_group:int=0, scoring_func:str='softmax', e_score_correction_bias:Optional[torch.Tensor]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `get_config_dtype_str(dtype:torch.dtype, use_int4_w4a16:Optional[bool]=False, use_int8_w8a16:Optional[bool]=False, use_fp8_w8a8:Optional[bool]=False) -> Any`
    Descrizione: Recupera valore/stato calcolato.
  - `inplace_fused_experts(hidden_states:torch.Tensor, w1:torch.Tensor, w2:torch.Tensor, topk_weights:torch.Tensor, topk_ids:torch.Tensor, use_fp8_w8a8:bool=False, use_int8_w8a16:bool=False, use_int4_w4a16:bool=False, w1_scale:Optional[torch.Tensor]=None, w2_scale:Optional[torch.Tensor]=None, w1_zp:Optional[torch.Tensor]=None, w2_zp:Optional[torch.Tensor]=None, a1_scale:Optional[torch.Tensor]=None, a2_scale:Optional[torch.Tensor]=None, block_shape:Optional[List[int]]=None) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `inplace_fused_experts_fake(hidden_states:torch.Tensor, w1:torch.Tensor, w2:torch.Tensor, topk_weights:torch.Tensor, topk_ids:torch.Tensor, use_fp8_w8a8:bool=False, use_int8_w8a16:bool=False, use_int4_w4a16:bool=False, w1_scale:Optional[torch.Tensor]=None, w2_scale:Optional[torch.Tensor]=None, w1_zp:Optional[torch.Tensor]=None, w2_zp:Optional[torch.Tensor]=None, a1_scale:Optional[torch.Tensor]=None, a2_scale:Optional[torch.Tensor]=None, block_shape:Optional[List[int]]=None) -> None`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `outplace_fused_experts(hidden_states:torch.Tensor, w1:torch.Tensor, w2:torch.Tensor, topk_weights:torch.Tensor, topk_ids:torch.Tensor, use_fp8_w8a8:bool=False, use_int8_w8a16:bool=False, use_int4_w4a16:bool=False, w1_scale:Optional[torch.Tensor]=None, w2_scale:Optional[torch.Tensor]=None, w1_zp:Optional[torch.Tensor]=None, w2_zp:Optional[torch.Tensor]=None, a1_scale:Optional[torch.Tensor]=None, a2_scale:Optional[torch.Tensor]=None, block_shape:Optional[List[int]]=None) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `outplace_fused_experts_fake(hidden_states:torch.Tensor, w1:torch.Tensor, w2:torch.Tensor, topk_weights:torch.Tensor, topk_ids:torch.Tensor, use_fp8_w8a8:bool=False, use_int8_w8a16:bool=False, use_int4_w4a16:bool=False, w1_scale:Optional[torch.Tensor]=None, w2_scale:Optional[torch.Tensor]=None, w1_zp:Optional[torch.Tensor]=None, w2_zp:Optional[torch.Tensor]=None, a1_scale:Optional[torch.Tensor]=None, a2_scale:Optional[torch.Tensor]=None, block_shape:Optional[List[int]]=None) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `fused_experts(hidden_states:torch.Tensor, w1:torch.Tensor, w2:torch.Tensor, topk_weights:torch.Tensor, topk_ids:torch.Tensor, inplace:bool=False, use_fp8_w8a8:bool=False, use_int8_w8a16:bool=False, use_int4_w4a16:bool=False, w1_scale:Optional[torch.Tensor]=None, w2_scale:Optional[torch.Tensor]=None, w1_zp:Optional[torch.Tensor]=None, w2_zp:Optional[torch.Tensor]=None, a1_scale:Optional[torch.Tensor]=None, a2_scale:Optional[torch.Tensor]=None, block_shape:Optional[List[int]]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `fused_experts_impl(hidden_states:torch.Tensor, w1:torch.Tensor, w2:torch.Tensor, topk_weights:torch.Tensor, topk_ids:torch.Tensor, inplace:bool=False, use_fp8_w8a8:bool=False, use_int8_w8a16:bool=False, use_int4_w4a16:bool=False, w1_scale:Optional[torch.Tensor]=None, w2_scale:Optional[torch.Tensor]=None, w1_zp:Optional[torch.Tensor]=None, w2_zp:Optional[torch.Tensor]=None, a1_scale:Optional[torch.Tensor]=None, a2_scale:Optional[torch.Tensor]=None, block_shape:Optional[List[int]]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `fused_moe(hidden_states:torch.Tensor, w1:torch.Tensor, w2:torch.Tensor, topk_weights:torch.Tensor, topk_ids:torch.Tensor, renormalize:bool=False, inplace:bool=False, use_grouped_topk:bool=False, num_expert_group:Optional[int]=None, topk_group:Optional[int]=None, custom_routing_function:Optional[Callable]=None, use_fp8_w8a8:bool=False, use_int8_w8a16:bool=False, use_int4_w4a16:bool=False, w1_scale:Optional[torch.Tensor]=None, w2_scale:Optional[torch.Tensor]=None, w1_zp:Optional[torch.Tensor]=None, w2_zp:Optional[torch.Tensor]=None, a1_scale:Optional[torch.Tensor]=None, a2_scale:Optional[torch.Tensor]=None, block_shape:Optional[List[int]]=None) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\dinfer\tools\modeling_fused_lladamoe.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `load_balancing_loss_func(gate_logits:torch.Tensor, num_experts:torch.Tensor=None, top_k:Any=2, attention_mask:Optional[torch.Tensor]=None) -> float`
    Descrizione: Carica dati o stato da sorgente esterna/file.
  - `rotate_half(x:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `apply_rotary_pos_emb(q:Any, k:Any, cos:Any, sin:Any, position_ids:Any=None, unsqueeze_dim:Any=1) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `repeat_kv(hidden_states:torch.Tensor, n_rep:int) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  Classi:
  - `LLaDAMoERMSNorm`
    Metodo: `__init__(self:Any, hidden_size:Any, eps:Any=1e-05) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, hidden_states:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `extra_repr(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LLaDAMoERotaryEmbedding`
    Metodo: `__init__(self:Any, dim:Any=None, max_position_embeddings:Any=2048, base:Any=10000, device:Any=None, scaling_factor:Any=1.0, rope_type:Any='default', config:Optional[LLaDAConfig]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `_dynamic_frequency_update(self:Any, position_ids:Any, device:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, x:Any, position_ids:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LLaDAMoEMLP`
    Metodo: `__init__(self:Any, config:Any, mlp_type:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, x:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LLaDAMoEAttention`
    Metodo: `__init__(self:Any, config:LLaDAConfig, layer_idx:Optional[int]=None) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, hidden_states:torch.Tensor, attention_mask:Optional[torch.Tensor]=None, position_ids:Optional[torch.LongTensor]=None, past_key_value:Optional[Cache]=None, output_attentions:bool=False, use_cache:bool=False, cache_position:Optional[torch.LongTensor]=None, position_embeddings:Optional[Tuple[torch.Tensor, torch.Tensor]]=None, **kwargs:Any) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LLaDAMoEFlashAttention2`
    Metodo: `__init__(self:Any, *args:Any, **kwargs:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, hidden_states:torch.Tensor, attention_mask:Optional[torch.LongTensor]=None, position_ids:Optional[torch.LongTensor]=None, past_key_value:Optional[Cache]=None, output_attentions:bool=False, use_cache:bool=False, cache_position:Optional[torch.LongTensor]=None, position_embeddings:Optional[Tuple[torch.Tensor, torch.Tensor]]=None, **kwargs:Any) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LLaDAMoESdpaAttention`
    Metodo: `forward(self:Any, hidden_states:torch.Tensor, attention_mask:Optional[torch.Tensor]=None, position_ids:Optional[torch.LongTensor]=None, past_key_value:Optional[Cache]=None, output_attentions:bool=False, use_cache:bool=False, cache_position:Optional[torch.LongTensor]=None, position_embeddings:Optional[Tuple[torch.Tensor, torch.Tensor]]=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LLaDAMoESparseMoeBlock`
    Metodo: `__init__(self:Any, config:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, hidden_states:torch.Tensor) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `FusedLLaDAMoESparseMoeBlock`
    Metodo: `__init__(self:Any, config:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `reset_parameters(self:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, hidden_states:torch.Tensor) -> torch.Tensor`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LLaDAMoEDecoderLayer`
    Metodo: `__init__(self:Any, config:LLaDAConfig, layer_idx:int) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, hidden_states:torch.Tensor, attention_mask:Optional[torch.Tensor]=None, position_ids:Optional[torch.LongTensor]=None, past_key_value:Optional[Cache]=None, output_attentions:Optional[bool]=False, output_router_logits:Optional[bool]=False, use_cache:Optional[bool]=False, cache_position:Optional[torch.LongTensor]=None, position_embeddings:Optional[Tuple[torch.Tensor, torch.Tensor]]=None, **kwargs:Any) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LLaDAMoEPreTrainedModel`
    Metodo: `_init_weights(self:Any, module:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `LLaDAMoEModel`
    Metodo: `__init__(self:Any, config:LLaDAConfig) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_input_embeddings(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `set_input_embeddings(self:Any, value:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `forward(self:Any, input_ids:torch.LongTensor=None, attention_mask:Optional[torch.Tensor]=None, position_ids:Optional[torch.LongTensor]=None, past_key_values:Optional[Union[Cache, List[torch.FloatTensor]]]=None, inputs_embeds:Optional[torch.FloatTensor]=None, use_cache:Optional[bool]=None, output_attentions:Optional[bool]=None, output_hidden_states:Optional[bool]=None, output_router_logits:Optional[bool]=None, return_dict:Optional[bool]=None, cache_position:Optional[torch.LongTensor]=None) -> Union[Tuple, MoeModelOutputWithPast]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
  - `FusedLLaDAMoEModelLM`
    Metodo: `__init__(self:Any, config:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_input_embeddings(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `set_input_embeddings(self:Any, value:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_output_embeddings(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `set_output_embeddings(self:Any, new_embeddings:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `set_decoder(self:Any, decoder:Any) -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
    Metodo: `get_decoder(self:Any) -> Any`
    Descrizione: Recupera valore/stato calcolato.
    Metodo: `forward(self:Any, input_ids:torch.LongTensor=None, attention_mask:Optional[torch.Tensor]=None, position_ids:Optional[torch.LongTensor]=None, past_key_values:Optional[List[torch.FloatTensor]]=None, inputs_embeds:Optional[torch.FloatTensor]=None, labels:Optional[torch.LongTensor]=None, use_cache:Optional[bool]=None, output_attentions:Optional[bool]=None, output_hidden_states:Optional[bool]=None, output_router_logits:Optional[bool]=None, return_dict:Optional[bool]=None, cache_position:Optional[torch.LongTensor]=None, num_logits_to_keep:int=0) -> Union[Tuple, MoeCausalLMOutputWithPast]`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.

- File: `vendor\dinfer\tools\transfer.py`
  Logica d'uso: Modulo di supporto nella pipeline SAFE.
  Funzioni:
  - `convert_and_save(input_path:str, output_path:str, modeling_file_name:str, device:str='cpu') -> Any`
    Descrizione: Funzione di utilita' usata nel flusso SAFE.
