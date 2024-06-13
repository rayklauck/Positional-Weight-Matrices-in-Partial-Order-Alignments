
from uncertain_dp2 import *
import random
random.seed(1)


DNA_LENGTH = 30           
READ_lENGTH = 20
READ_COUNT = 10

PROBABILISTIC_ALIGNMENT_BONUS = 2
NORMAL_ALIGNMENT_BONUS = 0.7


before_correction = []
normal_correction = []
probabilistic_correction = []

for i in range(3):

    dna = generate_dna(DNA_LENGTH)
    reads = [generate_read(dna, READ_lENGTH, lambda base: gauss_unsharp_uncertainty_generator(base, 0.47)) for _ in range(READ_COUNT)]

    corrected_reads = correct_reads_with_consens(reads,probabilistic=False, alignment_bonus=NORMAL_ALIGNMENT_BONUS)
    probabilistic_corrected_reads = correct_reads_with_consens(reads,probabilistic=True, alignment_bonus=PROBABILISTIC_ALIGNMENT_BONUS)

    before_correction.append(dna_distance_error_rate(dna, reads, alignment_bonus=NORMAL_ALIGNMENT_BONUS))
    normal_correction.append(dna_distance_error_rate(dna, corrected_reads, alignment_bonus=NORMAL_ALIGNMENT_BONUS))
    probabilistic_correction.append(dna_distance_error_rate(dna, probabilistic_corrected_reads, alignment_bonus=PROBABILISTIC_ALIGNMENT_BONUS))

print(before_correction, normal_correction, probabilistic_correction)
