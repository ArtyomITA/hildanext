# Audit report generation tests for formula-paper mapping.
# Entrypoint: unittest.
# Validates md/json outputs and core key coverage.
from __future__ import annotations
from pathlib import Path
import json
import tempfile
import unittest
from hildanext.audit import run_audit
from reporting import emit_payload

ROOT=Path(__file__).resolve().parents[1]

class AuditReportTests(unittest.TestCase):
    def test_audit_outputs_exist_and_have_expected_keys(self):
        with tempfile.TemporaryDirectory(prefix="hildanext_audit_") as td:
            td_path=Path(td)
            out_md=td_path/"formula_audit.md"
            out_json=td_path/"formula_audit.json"
            rep=run_audit(ROOT,out_md,out_json)
            self.assertTrue(out_md.exists())
            self.assertTrue(out_json.exists())
            self.assertIn("summary",rep)
            self.assertIn("components",rep)
            data=json.loads(out_json.read_text(encoding="utf-8"))
            self.assertIn("summary",data)
            self.assertIn("components",data)
            symbols={x["code_symbol"] for x in data["components"]}
            needed={
                "formulas.llada_m2t_loss",
                "formulas.llada2_wsd_block",
                "formulas.llada21_sets",
                "formulas.llada21_apply",
                "diffusion.compute_m2t_t2t_losses",
                "masks.doc_attention_mask",
                "masks.batch_doc_attention_mask"
            }
            self.assertTrue(needed.issubset(symbols))
            self.assertEqual(int(data["summary"].get("FAIL",0)),0)
            emit_payload(
                "test_audit_outputs_exist_and_have_expected_keys",
                "Runs audit and checks required formula symbols with no FAIL.",
                {"summary":data["summary"],"symbols":sorted(list(symbols))}
            )

if __name__=="__main__":
    unittest.main()
