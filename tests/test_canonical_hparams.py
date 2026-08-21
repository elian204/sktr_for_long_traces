"""Regression guard: entry-point decode defaults must equal CANONICAL_DECODE.

This exists because kfold_learning_curve_experiment.py silently drifted to
candidate_top_k=15, top_m=3 while every real run used 3 / 1 — costing a wasted
Breakfast measurement. See SKTR_HYPERPARAMETERS.md.

Run:  python3 tests/test_canonical_hparams.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sktr_hparams import CANONICAL_DECODE, deviations_from_canonical  # noqa: E402

ENTRY_POINTS = ['kfold_learning_curve_experiment.py', 'eval_diffact_sktr_fold1_paper.py']

# CLI flag -> CANONICAL_DECODE key
FLAGS = {
    '--candidate-top-k': 'candidate_top_k',
    '--candidate-top-p': 'candidate_top_p',
    '--candidate-min-k': 'candidate_min_k',
    '--state-mode': 'conditioning_state_mode',
    '--top-m': 'conditioning_top_m',
    '--chunk-size': 'chunk_size',
    '--prob-threshold': 'prob_threshold',
}


def defaults_in(path):
    """Extract each flag's `default=` expression from the argparse source."""
    src = path.read_text()
    found = {}
    for flag, key in FLAGS.items():
        for quoted in (f"'{flag}'", f'"{flag}"'):
            i = src.find(quoted)
            if i != -1:
                m = re.search(r'default=([^,\n)]+)', src[i:i + 400])
                if m:
                    found[key] = m.group(1).strip()
                break
    return found


def main():
    failures = []
    for name in ENTRY_POINTS:
        p = ROOT / name
        if not p.exists():
            continue
        for key, expr in defaults_in(p).items():
            # Accept only a reference to the canonical table -- a hardcoded
            # literal is exactly the drift this test is here to catch.
            if 'CANONICAL_DECODE' not in expr:
                failures.append(
                    f'{name}: {key} default is the literal `{expr}`, not '
                    f"CANONICAL_DECODE['{key}'] (canonical = {CANONICAL_DECODE[key]!r})"
                )

    # The canonical values themselves must not silently change.
    expected = {'candidate_top_k': 3, 'conditioning_top_m': 1,
                'conditioning_state_mode': 'topm', 'chunk_size': 11}
    for k, v in expected.items():
        if CANONICAL_DECODE[k] != v:
            failures.append(f'CANONICAL_DECODE[{k!r}] changed: {CANONICAL_DECODE[k]!r} != {v!r}. '
                            'If deliberate, update this test AND SKTR_HYPERPARAMETERS.md.')

    # Sanity: the detector must catch the historical mis-launch.
    if deviations_from_canonical({'candidate_top_k': 15, 'top_m': 3}) != {
            'candidate_top_k': (15, 3), 'conditioning_top_m': (3, 1)}:
        failures.append('deviations_from_canonical() no longer detects the 15/3 mis-launch')

    if failures:
        print('FAIL — canonical hyperparameter drift detected:')
        for f in failures:
            print('  -', f)
        return 1
    print(f'PASS — {len(ENTRY_POINTS)} entry points use CANONICAL_DECODE; values intact.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
