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
    read = Read("AATCGA", 3)
    reference_reads = [
        (Read("TCGAGG", 5), 5),
        (Read("CCCAAT", 0), 0),
    ]
    assert best_ensemble_alignment(reference_reads, read)[0] == 3


def test_most_likely_pot_string():
    get_in_pot_alignment = {
        Read("AATCGA", 3): 3,
        Read("TCGAGG", 5): 5,
    }
    assert most_likely_pot_string(get_in_pot_alignment) == ("AATCGAGG", 3)


def test_uncertain_text():
    Read("AATCGA", 3).predicted_text == "AATCGA"


def test_round_uncertainty_matrix_one_block():
    assert round_uncertainty_matrix([0.1, 0.2, 0.3, 0.4], M=1) == [0, 0, 0, 1]


def test_round_uncertainty_matrix_two_blocks():
    assert round_uncertainty_matrix([0.1, 0.2, 0.3, 0.4], M=2) == [0, 0, 0.5, 0.5]


def test_round_uncertainty_matrix_many_blocks_precise():
    assert round_uncertainty_matrix([0.1, 0.2, 0.3, 0.4], M=10) == [0.1, 0.2, 0.3, 0.4]


def test_round_uncertainty_matrix_many_blocks():
    assert round_uncertainty_matrix([0.1, 0.2, 0.3, 0.4], M=5) == [0.2, 0.2, 0.2, 0.4]


def test_round_uncertainty_base():
    assert round_uncertainty_matrix(just(A), M=1) == just(A)


def test_read_copy_independend():
    read = Read("AATCGA", 3)
    read_copy = read.copy()
    read_copy.uncertain_text[0] = [[0, 0, 2, 1]]
    assert read.uncertain_text == make_certain_uncertain("AATCGA")


def test_rounded_read_does_not_change_read():
    read = Read("AATCGA", 3)
    rounded_r = rounded_read(read, 1)
    assert read.uncertain_text == make_certain_uncertain("AATCGA")
