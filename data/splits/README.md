# Splits

This study has no train/validation/test split in the ML sense: every headline number is
computed over the frozen holdout prompt corpus (registered short sha `66365980`, see
`data/manifests/handoff_missing.yaml`) with a development half used only to freeze
detector thresholds ex ante (`CONSISTENCY_THRESHOLD`, the Platt gate constants — see
`src/grounded_matsci/domain/calibration.py` and `verification/two_stage.py`). The
dev/holdout partition is defined inside the corpus file, which is part of the handoff
data release and not in this repository.
