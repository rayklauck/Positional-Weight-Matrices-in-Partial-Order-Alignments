from pot_correction import *
import pytest


@pytest.mark.parametrize("read1, read2, expected", [
    ("ACT", "ACT", 3),
    ("ACT", "ACG", 2),
    ("ACT", "GCT", 2),
    ("ACT", "GCA", 1),
    ("ACT", "GCG", 1),
    ("AAATTAA", "TCTAAAT", 4),
    ("AAATTAA", "TCTAAAG", 3),
])
def test_best_alignment_similarity(read1, read2, expected):
    assert best_alignment_similarity(
        list(map(certain_uncertainty_generator,read1)),
        list(map(certain_uncertainty_generator,read2)),
        length_correction=lambda v, l: v) == expected
    