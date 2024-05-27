# import toyplot
from functools import cached_property
from random import choice, randint, uniform, random, gauss
from dataclasses import dataclass
import typing as t

# generate dna, reads, reads with errors
# similarity func?
# implement pot evaluation / sorting
# analyse results
#

A = "A"
T = "T"
C = "C"
G = "G"


def chance(p):
    return random() <= p


def clip(value, min_, max_):
    return min(max_, max(min_, value))


def string_similarity(s1, s2):
    assert len(s1) == len(s2)
    return sum([1 for i in range(len(s1)) if s1[i] == s2[i]]) / len(s1)

def normalize(vector: list[float]) -> list[float]:
    sum_ = sum(vector)
    if sum_ == 0:
        return [1 / len(vector) for _ in vector]
    return [a / sum_ for a in vector]

#def majority_vote(votes: list[t.Any]) -> t.Any:
#    return max(set(votes), key=votes.count)

def majority_vote(votes: list[t.Any]) -> t.Any:
    return max(votes, key=votes.count)

alphabet = ["A", "T", "C", "G"]
base_to_index = {"A": 0, "T": 1, "C": 2, "G": 3}
index_to_base = {0: "A", 1: "T", 2: "C", 3: "G"}

def generate_dna(length):
    return "".join([choice(alphabet) for _ in range(length)])


UncertainBase = tuple[float, float, float, float]

BLUE = 34
RED = 31
GREEN = 32
YELLOW = 33
WHITE = 37


def colored_print(text: str, color_code: int, *args, **kwargs):
    print(colored_string(text,color_code), *args, **kwargs)

def colored_string(text: str, color_code: int) -> str:
    return f"\033[1;{color_code}m{text}\033[0m"

def uncertain_base_scalar_product(base1: UncertainBase, base2: UncertainBase):
    return sum([base1[i] * base2[i] for i in range(4)])


def just(base: str) -> UncertainBase:
    return certain_uncertainty_generator(base)


def non_deterministic_probably(base: str, p=0.7) -> UncertainBase:
    return gauss_unsharp_uncertainty_generator(base, 1 - p)


def probably(base: str, p=0.7) -> UncertainBase:
    r = [(1 - p) / 3 for _ in range(4)]
    r[base_to_index[base]] = p
    return r


def probably_those(*bases: list[str], p=0.7) -> list[UncertainBase]:
    return [probably(base, p) for base in bases]


def just_those(*bases: list[str]) -> list[UncertainBase]:
    return [certain_uncertainty_generator(base) for base in bases]


def mix(*bases: list[str]) -> UncertainBase:
    n = len(bases)
    counts = [0, 0, 0, 0]
    for base in bases:
        counts[base_to_index[base]] += 1
    return [count / n for count in counts]


def certain(uncertain_base: UncertainBase) -> str:
    return alphabet[max(range(4), key=lambda i: uncertain_base[i])]


class Read:
    def __init__(
        self,
        original_text: str,
        start_position: int,
        end_position: int,
        uncertainty_generator: t.Callable[[str], list[UncertainBase]],
    ):
        self.original_text = original_text
        self.start_position = start_position
        self.end_position = end_position
        self.uncertainty_generator = uncertainty_generator
        self.uncertain_text: list[UncertainBase] = list(
            map(self.uncertainty_generator, self.original_text)
        )

    def __repr__(self):
        result = "<"
        for i in range(len(self.uncertain_text)):
            b = most_likely_base_restorer(self.uncertain_text[i])
            if b == self.original_text[i]:
                result += b
            else:
                result += colored_string(b, RED)
        return result + ">"

    def __hash__(self) -> int:
        return id(self)


def percent_most_likely_uncertainty_generator(base, inprecision_rate=0.15):
    """Imperfect unprecition generator

    Probability that correct base does not have highest probability is 'inprecision_rate'
    """
    index = base_to_index[base]
    w1 = 1 / 3 * (1 / (1 / inprecision_rate - 1))
    # print(w1)
    # result = [0,0,0,0]
    # for i in range(len(result)):
    #    result[i] = uniform(0, 2*inprecision_rate / (len(result)-1) )
    # result[index] = 1 - sum(result) + result[index]
    # return result
    while True:
        result = [random() for _ in range(4)]
        # print(result)
        result = [a / sum(result) for a in result]
        # normalize

        if max(result) == result[index]:  # p=0.25
            return result
        else:
            if chance(w1):
                return result


def certain_uncertainty_generator(base):
    """Simulating perfect measurements"""
    index = base_to_index[base]
    result = [0, 0, 0, 0]
    result[index] = 1
    return result


def unsharp_uncertainty_generator(base, inprecision_rate=0.15):
    """caution: can generate negative values"""
    result = certain_uncertainty_generator(base)
    for i in range(len(result)):
        result[i] += clip(
            result[i] + uniform(-inprecision_rate, inprecision_rate), 0, 1
        )

    # normalize
    sum_ = sum(result)
    if sum_ == 0:
        return [0.25, 0.25, 0.25, 0.25]
    result = [a / sum_ for a in result]
    return result


def gauss_unsharp_uncertainty_generator(base, inprecision_rate=0.15):
    """caution: can generate negative values"""
    result = certain_uncertainty_generator(base)
    for i in range(len(result)):
        result[i] = clip(result[i] + gauss(0, inprecision_rate), 0, 1)
    # normalize
    sum_ = sum(result)
    if sum_ == 0:
        return [0.25, 0.25, 0.25, 0.25]
    result = [a / sum_ for a in result]
    return result


def generate_read(dna, length, uncertainty_generator=certain_uncertainty_generator):
    start = randint(0, len(dna) - length)
    end = start + length
    return Read(dna[start:end], start, end, uncertainty_generator)


def spot_similarity(base_dist1, base_dist2):
    return sum([base_dist1[i] * base_dist2[i] for i in range(len(base_dist1))])


def piece_similarity(piece1, piece2):
    return sum([spot_similarity(piece1[i], piece2[i]) for i in range(len(piece1))])


def invers_sqrt_length_correction(value, length):
    return value / (length ** 0.5)


def best_alignment_similarity(
    read1,
    read2,
    return_all_results=False,
    min_considered_overlap=2,
    length_correction=invers_sqrt_length_correction,
) -> tuple[float, int] | list[tuple[float, int]]:
    """Try to align reads and find best similarity score.

    Offset defined as:
    the number of bases the second read starts after the first read (may be negative to indicate it starts before the first read)

    returns: (similarity_score, offset)
    """
    length_corrected_piece_similarity = lambda p1, p2: length_correction(
        piece_similarity(p1, p2), len(p1)
    )

    assert len(read1) == len(read2)
    size = len(read1)
    scores_and_alignment_offset: list[tuple[float, int]] = []

    for i in range(min_considered_overlap, size + 1):
        scores_and_alignment_offset.append(
            (length_corrected_piece_similarity(read1[:i], read2[size - i :]), -size + i)
        )
        scores_and_alignment_offset.append(
            (length_corrected_piece_similarity(read1[size - i :], read2[:i]), size - i)
        )
    if return_all_results:
        return scores_and_alignment_offset
    return max(scores_and_alignment_offset, key=lambda x: x[0])


def similarity_score(read1, read2) -> float:
    return best_alignment_similarity(read1, read2)[0]


def best_ensemble_alignment(
    reference_reads: list[tuple[Read, int]], read: Read
) -> tuple[int, float]:
    """Find the best alignment offset for a read based on multiple reference reads."""
    all_alignment_scores = []
    for reference_read, alignment_offset in reference_reads:
        similarity_scores = best_alignment_similarity(
            reference_read.uncertain_text, read.uncertain_text, return_all_results=True
        )
        similarity_scores = [
            (score, score_offset + alignment_offset)
            for score, score_offset in similarity_scores
        ]
        all_alignment_scores.extend(similarity_scores)
    # print(all_alignment_scores)

    alignment_options = {}
    for score, offset in all_alignment_scores:
        if offset not in alignment_options:
            alignment_options[offset] = []
        alignment_options[offset].append(score)

    combined_alignment_options = {
        offset: sum(scores)
        / len(scores)
        * len(scores) ** 0.1  # small reward for more frequent alignments
        for offset, scores in alignment_options.items()
    }
    # print(alignment_options)
    # print(combined_alignment_options)
    # return max([(x,y) for x,y in d.items()], key=lambda x: x[1])
    return max(
        [(x, y) for x, y in combined_alignment_options.items()], key=lambda x: x[1]
    )


class Pot:
    def __init__(self):
        self.members: t.Set[Read] = set()

    def similarity(self, read: Read, sample_size=5):
        if not self.members:
            return 1e30
        return sum(
            [
                similarity_score(
                    read.uncertain_text,
                    choice([m.uncertain_text for m in self.members]),
                )
                for _ in range(sample_size)
            ]
        )

    def __lt__(self, other):
        return choice([self, other])

    def __repr__(self):
        return str(self.members)


def get_pots(count, reads):
    pots = [Pot() for _ in range(count)]
    for i in range(len(pots)):  # distribute initial elements
        pots[i].members.add(reads[i])
    return pots


def pot_iteration(pots: list[Pot], reads, pot_sample_size=5):
    for read in reads:
        # for i in range(len(pots)):

        # remove read from the pot it was in
        for pot in pots:
            if read in pot.members:
                pot.members.remove(read)
                break

        _, best_pot = max(
            [(pot.similarity(read, pot_sample_size), pot) for pot in pots]
        )
        best_pot.members.add(read)


def histogram_like(positions, end):
    positions = set(positions)
    for i in range(end):
        print("|" if i in positions else ".", end="")
    print()


def most_likely_restorer(uncertain_text: list[UncertainBase]):
    return "".join(
        [
            alphabet[max(range(4), key=lambda i: uncertain_text[j][i])]
            for j in range(len(uncertain_text))
        ]
    )


def most_likely_base_restorer(uncertain_base: UncertainBase):
    return alphabet[max(range(4), key=lambda i: uncertain_base[i])]


def string_similarity(s1, s2):
    return sum([1 for i in range(len(s1)) if s1[i] == s2[i]]) / len(s1)


def get_in_pot_alignment(pot: Pot) -> tuple[dict[Read, int], dict[Read, float]]:
    read_to_start_position = {}

    # one read as a reference
    reference_read = list(pot.members)[0]
    read_to_start_position[reference_read] = 0
    reads_to_alignment_certainty = {
        reference_read: 100
    }  # this number is arbitrary, but should be high reletive to the other scores

    for _ in range(
        2
    ):  # iterativly improve (not only one round, as this would make the first to insert error-prone)
        for read in pot.members:
            for i in range(10):
                # randomly choose 5 reference reads
                reference_reads = [
                    choice(list(read_to_start_position.items())) for _ in range(5)
                ]
                alignment_offset, score = best_ensemble_alignment(reference_reads, read)
                reads_to_alignment_certainty[read] = score
                read_to_start_position[read] = alignment_offset
    return read_to_start_position, reads_to_alignment_certainty


def show_read_alignment(read_to_start_position, highest_minus=20):
    for read, start_position in read_to_start_position.items():
        print(
            " " * (start_position + highest_minus),
            most_likely_restorer(read.uncertain_text),
        )
    print()


def most_likely_pot_string(in_pot_alignment: dict[Read, int]) -> tuple[str, int]:
    """Return the most likely string and the relative alignment offset relative to the reads"""
    votes_per_position = {}

    for read, start_position in in_pot_alignment.items():
        for i in range(len(read.uncertain_text)):
            if i + start_position not in votes_per_position:
                votes_per_position[i + start_position] = []
            votes_per_position[i + start_position].append(read.uncertain_text[i])

    highest_position = max(votes_per_position.keys())
    lowest_position = min(votes_per_position.keys())

    restored_string = ["-" for _ in range(highest_position - lowest_position + 1)]

    for position, votes in votes_per_position.items():
        sum_base_distribution = [0, 0, 0, 0]
        for vote in votes:
            for i in range(4):
                sum_base_distribution[i] += vote[i]
        avg_base_distribution = [a / len(votes) for a in sum_base_distribution]

        restored_string[position - lowest_position] = most_likely_restorer(
            [avg_base_distribution]
        )[0]
    return "".join(restored_string), lowest_position


def correct_reads_with_restored_text(
    in_pot_alignment: dict[Read, int],
    restored_text: tuple[str, int],
    only_correct_if_change_maximal_percent=0.3,
):
    """todo: only correct if the difference is not too high (avoid correcting those who do not belong to the pot)"""
    restored_string, lowest_position = restored_text
    corrected_reads = []
    for read, start_position in in_pot_alignment.items():
        corrected_read = Read(
            read.original_text,
            read.start_position,
            read.end_position,
            read.uncertainty_generator,
        )
        corrected_read.uncertain_text = list(
            map(
                certain_uncertainty_generator,
                restored_string[
                    start_position
                    - lowest_position : start_position
                    - lowest_position
                    + len(read.uncertain_text)
                ],
            )
        )

        # do not correct reads if it does not belong to the pot
        if (
            string_similarity(
                most_likely_restorer(read.uncertain_text),
                most_likely_restorer(corrected_read.uncertain_text),
            )
            < 1 - only_correct_if_change_maximal_percent
        ):
            corrected_reads.append(read)
            continue
        # print(f"correct \n{most_likely_restorer(read.uncertain_text)} to \n{most_likely_restorer(corrected_read.uncertain_text)} while real is \n{read.original_text}")
        corrected_reads.append(corrected_read)
    return corrected_reads


def most_likely_restorer_error_rate(reads: list[Read]):
    similarities = [
        string_similarity(most_likely_restorer(r.uncertain_text), r.original_text)
        for r in reads
    ]
    print(similarities)
    return 1 - sum(similarities) / len(similarities)


def pot_correction(reads: list[Read], pot_count=5, iterations=2):
    pots = get_pots(pot_count, reads)
    for _ in range(iterations):
        pot_iteration(pots, reads)
    all_corrected_reads = []
    for pot in pots:
        read_to_start_position, _ = get_in_pot_alignment(pot)
        restored_text = most_likely_pot_string(read_to_start_position)
        corrected_reads = correct_reads_with_restored_text(
            read_to_start_position, restored_text
        )
        all_corrected_reads += corrected_reads
    return all_corrected_reads


def recursive_pot_sorting(reads: list[Read], subdivide_levels: int) -> list[list[Read]]:
    if subdivide_levels == 0:
        return [reads]

    pots = get_pots(2, reads)
    for _ in range(2):
        pot_iteration(pots, reads)

    return [
        e
        for pot in pots
        for e in recursive_pot_sorting(list(pot.members), subdivide_levels - 1)
    ]


##### kmer_hash


class KmerHash:
    def __init__(self, kmer_length, reads: list[Read]):
        self.kmer_length = kmer_length
        self.kmer_hash = {}
        self.reads = reads

        for read in reads:
            text = most_likely_restorer(read.uncertain_text)
            for i in range(len(text) - kmer_length + 1):
                kmer = text[i : i + kmer_length]
                if kmer not in self.kmer_hash:
                    self.kmer_hash[kmer] = []
                self.kmer_hash[kmer].append(read)

    def get_adjacent_reads(self, kmer: str, max_distance) -> t.Iterator[Read]:
        for query in get_max_n_error_ensemble(kmer, max_distance):
            for result in self.kmer_hash.get(query, []):
                yield result


def get_one_error_ensemble(text: str):
    """Generate all possible strings which have one error compared to the input string"""
    for i in range(len(text)):
        for base in alphabet:
            if base != text[i]:
                yield text[:i] + base + text[i + 1 :]


def get_n_error_ensemble(text: str, n: int) -> t.Iterator[str]:
    """Generate all possible strings which have n errors compared to the input string"""
    if n == 0:
        yield text
        return
    result = set()
    for i in range(len(text)):
        for base in alphabet:
            if base != text[i]:
                for result in get_n_error_ensemble(
                    text[:i] + base + text[i + 1 :], n - 1
                ):
                    yield result


def get_max_n_error_ensemble(text: str, n: int) -> t.Iterator[str]:
    """Generate all possible strings which have max n errors compared to the input string"""
    for i in range(n + 1):
        for result in get_n_error_ensemble(text, i):
            yield result


# typedef
V = t.TypeVar("V")


def take_n_unique(iterable: t.Iterable[V], n: int) -> t.Iterator[V]:
    """Take n unique elements from an iterable"""
    seen = set()
    for element in iterable:
        if element not in seen:
            seen.add(element)
            yield element
            if len(seen) == n:
                break
