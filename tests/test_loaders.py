from __future__ import annotations

import pytest

from mosaic.eval.loaders.registry import load_eval_bundle


def test_registry_unknown_dataset() -> None:
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_eval_bundle("not-a-dataset")
