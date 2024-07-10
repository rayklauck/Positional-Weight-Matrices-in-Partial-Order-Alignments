from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pot_correction import *
from uncertain_dp2 import *


@dataclass
class Environment:
    read_count: int = 10
    read_length: int = 35
    dna_length: int = 50
    normal_alignment_bonus: int = 0.7
    probabilistic_alignment_bonus: int = 2
    gauss_unsharpness: float = 0.47

    # def define_globally(self):
    # global READ_COUNT, READ_lENGTH, DNA_LENGTH, NORMAL_ALIGNMENT_BONUS
    # READ_COUNT = self.read_count
    # READ_lENGTH = self.read_length
    # DNA_LENGTH = self.dna_length
    # NORMAL_ALIGNMENT_BONUS = self.normal_alignment_bonus


def compare_and_plot(
    read_correctors: list[t.Callable[[list[Read]], list[Read]]],
    *,
    environment: Environment,
    names: list[str] | None = None,
    iterations=4
):
    if names is None:
        names = [str(i) for i in range(len(read_correctors))]

    all: list[list[float]] = [[] for _ in read_correctors]
    for _ in range(iterations):

        dna = generate_dna(environment.dna_length)
        reads = [
            generate_read(
                dna,
                environment.read_length,
                lambda base: gauss_unsharp_uncertainty_generator(
                    base, environment.gauss_unsharpness
                ),
            )
            for _ in range(environment.read_count)
        ]

        for j, corrector in enumerate(read_correctors):
            corrected_reads = corrector(reads)
            all[j].append(
                dna_distance_error_rate(
                    dna,
                    corrected_reads,
                    alignment_bonus=environment.normal_alignment_bonus,
                )
            )

    plt.bar(
        names,
        [np.mean(measurement_row) for measurement_row in all],
        yerr=[pd.Series(measurement_row).sem() for measurement_row in all],
        capsize=6,
    )

    return all
