# Test Output
Generated: 2026-02-22 04:04:55 +01:00

## Command 1
python e:\DIFFUSION\HildaNext\hildanext\test\run_tests.py
Exit code: 0

```text
test_ar_generation_dummy (test_ar.ARTests.test_ar_generation_dummy) ... ok
test_llada21_gamma_delta_sets (test_formulas.FormulaTests.test_llada21_gamma_delta_sets) ... ok
test_llada2_wsd_phase_boundaries (test_formulas.FormulaTests.test_llada2_wsd_phase_boundaries) ... ok
test_llada_m2t_loss_matches_masked_ce (test_formulas.FormulaTests.test_llada_m2t_loss_matches_masked_ce) ... ok
test_batch_doc_attention_mask_shape (test_masks.MaskTests.test_batch_doc_attention_mask_shape) ... ok
test_doc_attention_mask_blocks_cross_doc (test_masks.MaskTests.test_doc_attention_mask_blocks_cross_doc) ... ok
test_dtype_mapping_cpu (test_precision.PrecisionTests.test_dtype_mapping_cpu) ... ok
test_tiny_forward_fp16_if_cuda (test_precision.PrecisionTests.test_tiny_forward_fp16_if_cuda) ... skipped 'cuda not available'
test_tiny_forward_fp32 (test_precision.PrecisionTests.test_tiny_forward_fp32) ... ok
test_sft_one_step (test_sft_smoke.SFTSmokeTests.test_sft_one_step) ... ok
test_local_tokenizer_mask (test_vocab_mask.VocabMaskTests.test_local_tokenizer_mask) ... ok
test_simple_tokenizer_mask (test_vocab_mask.VocabMaskTests.test_simple_tokenizer_mask) ... ok

----------------------------------------------------------------------
Ran 12 tests in 23.011s

OK (skipped=1)

```

## Command 2
python -m hildanext.cli smoke-test --config e:\DIFFUSION\HildaNext\hildanext\runs\configs\smoke.json
Exit code: 0

```text
{
  "ok": true,
  "load_model": {
    "dummy_model": true,
    "load_error": ""
  },
  "prepare_data": {
    "dolma": 64,
    "tinystories": 64,
    "train": 51,
    "eval": 13,
    "sft_train": 51,
    "sft_eval": 13
  },
  "tokenize": {
    "train": 409,
    "eval": 105,
    "sft_train": 11,
    "sft_eval": 3
  },
  "cpt": {
    "kind": "cpt",
    "steps": 1,
    "loss_last": 12.07774543762207,
    "dummy_model": true,
    "token_seen": 128,
    "tokens_per_sec": 24.003856921034746,
    "log_path": "e:\\DIFFUSION\\HildaNext\\hildanext\\runs\\logs\\cpt.jsonl",
    "checkpoints_dir": "e:\\DIFFUSION\\HildaNext\\hildanext\\runs\\checkpoints\\cpt"
  },
  "sft": {
    "kind": "sft",
    "steps": 1,
    "loss_last": 12.235666275024414,
    "dummy_model": true,
    "token_seen": 128,
    "tokens_per_sec": 23.48926453136269,
    "log_path": "e:\\DIFFUSION\\HildaNext\\hildanext\\runs\\logs\\sft.jsonl",
    "checkpoints_dir": "e:\\DIFFUSION\\HildaNext\\hildanext\\runs\\checkpoints\\sft"
  },
  "sample_generation": "dummy-output",
  "benchmarks": {
    "tinystories": {
      "items": 4,
      "non_empty": 4,
      "avg_words": 1.0
    },
    "humaneval_dummy": {
      "items": 4,
      "non_empty": 4
    }
  }
}
OK

```
