# Unified unittest runner for HildaNext test pack.
# Entry point: python hildanext/test/run_tests.py.
# Runs all test_*.py files in this folder.
from __future__ import annotations
from pathlib import Path
from datetime import datetime,timezone
import os
import sys
import unittest
import json
from reporting import reset_payload_log

ROOT=Path(__file__).resolve().parents[1]
LOG_DIR=ROOT/"runs"/"reports"/"logs"
RESULT_JSON=LOG_DIR/"unittest_result_mdm.json"

def _is_mdm()->bool:
    env=os.environ.get("CONDA_DEFAULT_ENV","").lower()
    exe=str(Path(sys.executable)).replace("/","\\").lower()
    return env=="mdm" or "\\envs\\mdm\\" in exe

class DetailedTextResult(unittest.TextTestResult):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.items=[]
        self._start={}
    def startTest(self,test):
        self._start[id(test)]=datetime.now(timezone.utc).timestamp()
        super().startTest(test)
    def _append(self,test,status,err=None):
        t0=self._start.get(id(test),datetime.now(timezone.utc).timestamp())
        t1=datetime.now(timezone.utc).timestamp()
        row={"test_id":str(test),"status":status,"duration_sec":round(float(max(0.0,t1-t0)),6)}
        if err is not None:
            row["error"]="".join(self._exc_info_to_string(err,test))
        self.items.append(row)
    def addSuccess(self,test):
        self._append(test,"ok")
        super().addSuccess(test)
    def addSkip(self,test,reason):
        self._append(test,"skipped",None)
        self.items[-1]["skip_reason"]=str(reason)
        super().addSkip(test,reason)
    def addFailure(self,test,err):
        self._append(test,"failure",err)
        super().addFailure(test,err)
    def addError(self,test,err):
        self._append(test,"error",err)
        super().addError(test,err)

class DetailedRunner(unittest.TextTestRunner):
    resultclass=DetailedTextResult

def main()->int:
    if not _is_mdm():
        print("ERROR: tests must run in conda env 'mdm'.")
        print("Use: conda run -n mdm python hildanext/test/run_tests.py")
        return 2
    LOG_DIR.mkdir(parents=True,exist_ok=True)
    reset_payload_log()
    root=Path(__file__).resolve().parent
    suite=unittest.defaultTestLoader.discover(str(root),pattern="test_*.py")
    runner=DetailedRunner(verbosity=2)
    res=runner.run(suite)
    payload={
        "ts":datetime.now(timezone.utc).isoformat(),
        "python_exe":sys.executable,
        "env":{"CONDA_DEFAULT_ENV":os.environ.get("CONDA_DEFAULT_ENV","")},
        "summary":{
            "testsRun":int(res.testsRun),
            "failures":len(res.failures),
            "errors":len(res.errors),
            "skipped":len(res.skipped),
            "ok":bool(res.wasSuccessful())
        },
        "tests":getattr(res,"items",[])
    }
    RESULT_JSON.write_text(json.dumps(payload,ensure_ascii=True,indent=2),encoding="utf-8")
    return 0 if res.wasSuccessful() else 1

if __name__=="__main__":
    raise SystemExit(main())
