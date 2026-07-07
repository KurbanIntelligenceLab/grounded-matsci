# Splits

This study has no train/validation/test split in the ML sense: every headline number is
computed over the frozen holdout prompt corpus (committed at
`data/frozen/corpus_final_v2.json`, registered short sha `66365980`) with a development
half used only to freeze detector thresholds ex ante (`CONSISTENCY_THRESHOLD`, the Platt
gate constants — see `src/grounded_matsci/domain/calibration.py` and
`verification/two_stage.py`). The dev/holdout partition is defined inside the corpus
file itself.
