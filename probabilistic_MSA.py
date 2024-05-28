from pot_correction import *


DELETION = 1
INSERTION = 2
SUBSTITUTION = 3

NOT_SET = -1


def dp_with_probabilities(
    sequence1: list[UncertainBase], sequence2: list[UncertainBase]
):
    dp_table: list[list[int]] = [
        [NOT_SET for _ in range(len(sequence1) + 1)] for _ in range(len(sequence2) + 1)
    ]
    backtrace: list[list[int]] = [
        [NOT_SET for _ in range(len(sequence1) + 1)] for _ in range(len(sequence2) + 1)
    ]

    # base cases
    for i in range(len(sequence1) + 1):
        dp_table[0][i] = i
        backtrace[0][i] = DELETION

    for i in range(len(sequence2) + 1):
        dp_table[i][0] = i
        backtrace[i][0] = INSERTION

    for i in range(1, len(sequence2) + 1):
        for j in range(1, len(sequence1) + 1):
            # deletion
            deletion = dp_table[i - 1][j] + 1
            # insertion
            insertion = dp_table[i][j - 1] + 1
            # substitution
            substitution = (
                dp_table[i - 1][j - 1]
                + 1
                - uncertain_base_scalar_product(sequence1[j - 1], sequence2[i - 1])
            )
            dp_table[i][j] = min(deletion, insertion, substitution)

            if dp_table[i][j] == deletion:
                backtrace[i][j] = DELETION
            elif dp_table[i][j] == insertion:
                backtrace[i][j] = INSERTION
            else:
                backtrace[i][j] = SUBSTITUTION

    #print(backtrace)
    # print the changes with colors
    queue = []
    i = len(sequence2)
    j = len(sequence1)
    while i > 0 or j > 0:
        if backtrace[i][j] == DELETION:
            i -= 1
            queue.append((sequence2[i], RED))
        elif backtrace[i][j] == INSERTION:
            j -= 1
            queue.append((sequence1[j], GREEN))
        else:
            i -= 1
            j -= 1
            if uncertain_base_scalar_product(sequence1[j], sequence2[i]) > 0.6:
                queue.append((sequence2[i], WHITE))
            else:
                queue.append((sequence2[i], BLUE))

    for base, color in reversed(queue):
        colored_print(f"{most_likely_base_restorer(base)}", color, end="")

        # print_color(f"{most_likely_base_restorer(sequence1[j-1])}", RED, end='')

    return dp_table[len(sequence2)][len(sequence1)]
