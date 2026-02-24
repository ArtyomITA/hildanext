# Minimal tests for package wiring.
# Main focus: import and parser shape.
# Heavy smoke is executed via CLI command.
from hildanext.cli import build_parser

def test_required_commands_present():
    parser=build_parser()
    names=set(parser._subparsers._group_actions[0].choices.keys())
    assert {"prepare-data","tokenize","convert-wsd","sft","serve","smoke-test","audit","quant-bench"}.issubset(names)
