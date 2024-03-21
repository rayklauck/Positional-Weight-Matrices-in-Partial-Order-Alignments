
#import toyplot
from functools import cached_property
from random import choice, randint, uniform, random, gauss
from dataclasses import dataclass
import typing as t

# generate dna, reads, reads with errors
# similarity func?
# implement pot evaluation / sorting
# analyse results
#

def chance(p):
    return random() <= p

def clip(value, min_, max_):
    return min(max_, max(min_, value))



alphabet = ["A","T","C","G"]
base_index = {"A":0, "T":1, "C":2, "G":3}


def generate_dna(length):
    return "".join([choice(alphabet) for _ in range(length)])



class Read:
    def __init__(self, original_text, start_position, end_position, uncertainty_generator: t.Callable[[str], list[tuple[float, float, float, float]]]):
        self.original_text = original_text
        self.start_position = start_position
        self.end_position = end_position
        self.uncertainty_generator = uncertainty_generator
        self.uncertain_text = list(map(self.uncertainty_generator, self.original_text))


    def __repr__(self):
        return f"<{self.original_text}>"
    
    def __hash__(self) -> int:
        return id(self)




def percent_most_likely_uncertainty_generator(base, inprecision_rate=0.15):
    """Imperfect unprecition generator
    
    Probability that correct base does not have highest probability is 'inprecision_rate'
    """
    index = base_index[base]
    w1 = 1/3 * (1/(1/inprecision_rate - 1))
    #print(w1)
    #result = [0,0,0,0]
    #for i in range(len(result)):
    #    result[i] = uniform(0, 2*inprecision_rate / (len(result)-1) )
    #result[index] = 1 - sum(result) + result[index]
    #return result
    while True:
        result = [random() for _ in range(4)]
        #print(result)
        result = [a/sum(result) for a in result]
        # normalize

        if max(result) == result[index]:  # p=0.25
            return result
        else:
            if chance(w1):
                return result
            

def certain_uncertainty_generator(base):
    """Simulating perfect measurements"""
    index = base_index[base]
    result = [0,0,0,0]
    result[index] = 1
    return result


def unsharp_uncertainty_generator(base, inprecision_rate=0.15):
    """caution: can generate negative values"""
    result = certain_uncertainty_generator(base)
    for i in range(len(result)):
        result[i] += clip(result[i] + uniform(-inprecision_rate, inprecision_rate), 0, 1)

    # normalize
    sum_ = sum(result)
    if sum_ == 0:
        return [0.25, 0.25, 0.25, 0.25]
    result = [a/sum_ for a in result]
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
    result = [a/sum_ for a in result]
    return result



def generate_read(dna, length, uncertainty_generator=certain_uncertainty_generator):
    start = randint(0, len(dna) - length)
    end = start + length
    return Read(dna[start: end], start, end, uncertainty_generator)







def spot_similarity(base_dist1, base_dist2):
    return sum([base_dist1[i]*base_dist2[i] for i in range(len(base_dist1))])

def piece_similarity(piece1, piece2):
    return sum([spot_similarity(piece1[i], piece2[i]) for i in range(len(piece1))])


def invers_sqrt_length_correction(value, length):
    return value / (length**0.5)

def best_alignment_similarity(read1, read2, min_considered_overlap=2, length_correction=invers_sqrt_length_correction):
    """Try to align reads and find best similarity score."""
    length_corrected_piece_similarity = lambda p1, p2: length_correction(piece_similarity(p1, p2), len(p1))

    assert len(read1) == len(read2)
    size = len(read1)
    scores = []
    for i in range(min_considered_overlap,size+1):
        scores.append(
            length_corrected_piece_similarity(read1[:i], read2[size-i:])
        )
        scores.append(
            length_corrected_piece_similarity(read1[size-i:], read2[:i])
        )
    return max(scores)

def similarity_score(read1, read2):
    return best_alignment_similarity(read1, read2)



class Pot:
    def __init__(self):
        self.members: t.Set[Read] = set()

    def similarity(self, read: Read, sample_size=5):
        if not self.members:
            return 1e30
        return sum([similarity_score(read.uncertain_text, choice([m.uncertain_text for m in self.members])) for _ in range(sample_size)])
    
    def __lt__(self, other):
        return choice([self,other])
    
    def __repr__(self):
        return str(self.members)

def get_pots(count, reads):
    pots = [Pot() for _ in range(count)]
    for i in range(len(pots)): # distribute initial elements
        pots[i].members.add(reads[i])
    return pots



def pot_iteration(pots: list[Pot], reads, pot_sample_size=5):
    for read in reads:
        #for i in range(len(pots)):

        # remove read from the pot it was in
        for pot in pots:
            if read in pot.members:
                pot.members.remove(read)
                break
        
        _, best_pot = max([(pot.similarity(read, pot_sample_size), pot) for pot in pots])
        best_pot.members.add(read)



def histogram_like(positions, end):
    positions = set(positions)
    for i in range(end):
        print("|" if i in positions else ".", end="")
    print()


def most_likely_restorer(read: Read):
    return "".join([alphabet[max(range(4), key=lambda i: read.uncertain_text[j][i])] for j in range(len(read.uncertain_text))])


def string_similarity(s1, s2):
    return sum([1 for i in range(len(s1)) if s1[i] == s2[i]]) / len(s1)