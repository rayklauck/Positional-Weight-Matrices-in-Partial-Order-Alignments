from pot_correction import *
import pytest


@pytest.mark.parametrize(
    "read1, read2, expected",
    [
        ("ACT", "ACT", 3),
        ("ACT", "ACG", 2),
        ("ACT", "GCT", 2),
        ("ACT", "GCA", 1),
        ("ACT", "GCG", 1),
        ("AAATTAA", "TCTAAAT", 4),
        ("AAATTAA", "TCTAAAG", 3),
    ],
)
def test_best_alignment_similarity(read1, read2, expected):
    assert (
        best_alignment_similarity(
            list(map(certain_uncertainty_generator, read1)),
            list(map(certain_uncertainty_generator, read2)),
            length_correction=lambda v, l: v,
        )[0]
        == expected
    )


def test_best_ensemble_alignment():
    read = Read("AATCGA", 3, 9, certain_uncertainty_generator)
    reference_reads = [
        (Read("TCGAGG", 5, 11, certain_uncertainty_generator), 5),
        (Read("CCCAAT", 0, 6, certain_uncertainty_generator), 0),
    ]
    assert best_ensemble_alignment(reference_reads, read)[0] == 3


def test_most_likely_pot_string():
    get_in_pot_alignment = {
        Read("AATCGA", 3, 9, certain_uncertainty_generator): 3,
        Read("TCGAGG", 5, 11, certain_uncertainty_generator): 5,
    }
    assert most_likely_pot_string(get_in_pot_alignment) == ("AATCGAGG", 3)
