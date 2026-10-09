"""Fault injection: the V3 test suite must reject deliberately broken copies of accounting.py (one per subprocess).

Each fault replaces one exact source fragment (asserted unique) in an in-memory copy registered as the package's
``accounting`` module; the unchanged test module is then run. F15 (owner shortcut from any caller chain) is an
EQUIVALENT mutant: the whole-graph witness alone makes any reuse exact, so the chain restriction is defence in depth
and no test can (or should) fail. Usage (from the checkout):

  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 /usr/bin/python3 -B -m \\
      rl_agent.splitfusion_gt_storage_accounting_v3.fault_injection --out <new json file>
"""
import importlib.util, json, subprocess, sys, types, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; PKG = "rl_agent.splitfusion_gt_storage_accounting_v3"
SRC = (ROOT / "rl_agent/splitfusion_gt_storage_accounting_v3/accounting.py").read_text()
FAULTS = {
 "F1_ignore_prior_context": ("sum(map(own_by_id.__getitem__, own_by_id.keys() - seen))", "sum(own_by_id.values())"),
 "F2_witness_always_unchanged": ("        if self.ineligible or self.dict_tags is None:\n            return False\n        try:", "        return True\n        try:"),
 "F3_row_memo_ignores_state": ("row_tags = list(map(_dict_version, rows))", "row_tags = list(map(id, rows))"),
 "F4_no_list_element_identity": ("                    and all(map(is_, chain.from_iterable(self.seqs), self.seq_items))\n", ""),
 "F5_pair_history_overlap_type_unchecked": ("if not set(map(type, map(_PH_COUNT, histories))) <= _NUMERIC:", "if False:"),
 "F6_no_pair_history_class_guard": ("        if not _pair_history_class_unchanged():\n            return self._decline(\"pair_history_class_changed\")\n", ""),
 "F7_tuple_memo_not_pinned": ("memo.tuple_pins.update(zip(tuple_ids, tuples))", "pass"),
 "F8_seen_not_updated": ("        seen.update(own_by_id)\n        self.snapshot_fast += 1", "        self.snapshot_fast += 1"),
 "F9_plan_delta_ignored": ("result = previous[2] - prior_size + current_size", "result = previous[2]"),
 "F10_ineligible_scalars_allowed": ("                if record and type(value) is not str and type(value) is not bytes:\n                    witness.ineligible = True\n", ""),
 "F11_no_dataclass_witness": ("                    and all(type(value) is kind and all(map(is_, _dataclass_children(value), children))\n                            and len(children) == len(fields(value))\n                            for value, kind, children in self.objects)\n", ""),
 "F12_no_array_state_witness": ("                    and list(map(_array_state, self.arrays)) == self.array_states\n", ""),
 "F16_array_hasobject_not_witnessed": ("    return array.dtype.hasobject, array.nbytes", "    return array.nbytes"),
 "F13_no_pair_history_tags_in_witness": ("list(map(_dict_version, map(vars, self.pair_histories))) == self.pair_history_tags", "True"),
 "F14_samples_not_recounted": ("own_by_id[id(samples)] = own_samples", "own_by_id[id(samples)] = 16"),
 "F15_shortcut_any_caller": ("(previous[0], caller) in _OWNER_CHAINS and ", ""),
}
def child(name):
    old, new = FAULTS[name]
    assert SRC.count(old) == 1, name
    import rl_agent.splitfusion_gt_storage_accounting_v3 as pkg
    module = types.ModuleType(PKG + ".accounting"); module.__package__ = PKG; module.__file__ = str(ROOT / "rl_agent/splitfusion_gt_storage_accounting_v3/accounting.py")
    sys.modules[PKG + ".accounting"] = module
    exec(compile(SRC.replace(old, new), f"<{name}>", "exec"), module.__dict__)
    pkg.accounting = module
    from rl_agent.splitfusion_gt_storage_accounting_v3 import test_accounting as T
    result = unittest.TextTestRunner(stream=open("/dev/null", "w")).run(unittest.defaultTestLoader.loadTestsFromModule(T))
    failed = sorted({t.id().rsplit(".", 1)[1] for t, _ in result.failures + result.errors})
    print(json.dumps(dict(fault=name, tests=result.testsRun, failed=len(failed), failing_tests=failed)))
EQUIVALENT = {"F15_shortcut_any_caller"}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.child:
        child(args.child)
        return 0
    rows = []
    for name in FAULTS:
        out = subprocess.run([sys.executable, "-B", "-m", PKG + ".fault_injection", "--child", name],
                             capture_output=True, text=True, cwd=ROOT)
        try:
            rows.append(json.loads(out.stdout.strip().splitlines()[-1]))
        except (ValueError, IndexError):
            rows.append(dict(fault=name, error=out.stderr.strip()[-600:]))
        print(json.dumps(rows[-1]), flush=True)
    detected = [r for r in rows if r.get("failed")]
    status = ("PASS" if all(r.get("failed") for r in rows if r["fault"] not in EQUIVALENT)
              and all(not r.get("failed") for r in rows if r["fault"] in EQUIVALENT) else "FAIL")
    summary = dict(schema="splitfusion.gt_storage_accounting_v3.fault_injection.v1", status=status,
                   faults=len(rows), detected=len(detected), equivalent=sorted(EQUIVALENT), rows=rows)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "x") as stream:
            json.dump(summary, stream, indent=1)
            stream.write("\n")
    print(json.dumps(dict(status=status, faults=len(rows), detected=len(detected))))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
