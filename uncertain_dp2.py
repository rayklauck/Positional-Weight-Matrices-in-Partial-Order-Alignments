from dataclasses import field
from enum import Enum
from functools import cache
from pot_correction import *
from abc import ABC, abstractmethod

DELETION_PENALTY = 1
INSERTION_PENALTY = 1
# ALIGNMENT_BONUS = 2 # 0.7  # adjust based on the consideration: how many aligned bases is one unaligned worth?


class BaseLike(ABC):
    """Can play the role of a base"""

    @abstractmethod
    def __eq__(self, value: object) -> bool:
        ...

    @abstractmethod
    def dp_conversion_penalty(self, value: object) -> float:
        ...

    @classmethod
    def consent(cls, sequence: list["BaseLike"]) -> "BaseLike":
        return majority_vote(sequence)


class ReadNode:
    """LinkedList for one read"""

    def __init__(self, base: BaseLike, next: t.Union["ReadNode", None] = None):
        self.next = next
        self.base = base
        self.graph_node = None

    def __repr__(self):
        return f"R({self.base})"


class GraphNode:
    """Node representing vertex for specific base at certain position."""

    def __init__(self):
        self.read_nodes: list[ReadNode] = []
        self.layer: t.Union[None, AssoziatedLayer] = None

    def add(self, read_node: ReadNode):
        self.read_nodes.append(read_node)
        read_node.graph_node = self

    @property
    def successors(self) -> list["GraphNode"]:
        return list(
            {
                read_node.next.graph_node
                for read_node in self.read_nodes
                if read_node.next is not None
            }
        )

    @property
    def base(self) -> BaseLike:
        """Guaranteed that all read_nodes will have the same base here.
        And there will be never a GraphNode with no ReadNode."""
        return self.read_nodes[0].base

    def __repr__(self):
        return f"{self.base}-{self.successors if len(self.successors)!=1 else self.successors[0]}"


class AssoziatedLayer:
    """Layer for specific position. Contains GraphNodes for occuring bases"""

    def __init__(self):
        self.graph_nodes: list[GraphNode] = []
        self.graph: Graph = None

    def add(self, graph_node: GraphNode):
        self.graph_nodes.append(graph_node)
        graph_node.layer = self  # set backreference


@dataclass
class Graph:
    start_nodes: list[GraphNode]
    reads: list[ReadNode] = field(default_factory=list)


class RegularBase(BaseLike):
    def __init__(self, base: str):
        self.base = base

    def __hash__(self) -> int:
        return hash(self.base)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, RegularBase):
            return False
        return self.base == value.base

    def dp_conversion_penalty(self, value: object) -> float:
        return self.base != value.base

    @staticmethod
    def consent(sequence: list["BaseLike"]) -> "BaseLike":
        histogram = {}
        for base in sequence:
            histogram[base] = histogram.get(base, 0) + 1
        return max(histogram, key=histogram.get)

    def __repr__(self):
        return self.base


class PositionalWeightMatrixBase(BaseLike):
    """Base with associated probability"""

    def __init__(self, base: UncertainBase) -> None:
        self.base = base

    def __hash__(self) -> int:
        return hash(self.base)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, PositionalWeightMatrixBase):
            return False
        return self.base == value.base

    def dp_conversion_penalty(self, value: object) -> float:
        return 1 - uncertain_base_scalar_product(self.base, value.base)

    @staticmethod
    def consent(
        sequence: list["PositionalWeightMatrixBase"],
    ) -> "PositionalWeightMatrixBase":
        histogram = [0, 0, 0, 0]
        for b in sequence:
            for i, v in enumerate(b.base):
                histogram[i] += v

        return PositionalWeightMatrixBase(normalize(histogram))

    def __repr__(self):
        # return "~" + most_likely_base_restorer(self.base)
        return str(self.base)


def make_regular(*bases: list[str]) -> list[RegularBase]:
    return [RegularBase(base) for base in bases]


def make_uncertain_regular(
    *bases: list[UncertainBase],
) -> list[PositionalWeightMatrixBase]:
    return [PositionalWeightMatrixBase(base) for base in bases]


class DpOperation(Enum):
    """Enum for DP operations"""

    DELETE = 1
    INSERT = 2
    REPLACE = 3
    MATCH = 4
    INSERT_ALL_END = 5
    DELETE_ALL_END = 6
    START = 7
    DELETE_START = 8
    INSERT_START = 9


@dataclass
class DpOption:
    """Option for DP"""

    cost: float
    trace: list["TracePoint"]

    def matching_adjusted_cost(self, alignment_bonus):
        total_alignment_bonus = 0
        for trace_point in self.trace:
            if trace_point.operation == DpOperation.MATCH:
                total_alignment_bonus += alignment_bonus
            elif trace_point.operation == DpOperation.REPLACE:
                total_alignment_bonus += alignment_bonus * (
                    1 - trace_point.replacement_cost
                )

        return self.cost - total_alignment_bonus


@dataclass
class TracePoint:
    operation: DpOperation
    graph_node: GraphNode
    replacement_cost: float | None = None  # for replace operations only
    end_cut_out_sequence: list[BaseLike] | None = None  # for end operations only


def carefully_chached(func: t.Callable[[GraphNode, int, bool, bool, float], DpOption]):
    cache: dict[tuple[GraphNode, int, bool, bool, float], DpOption] = {}

    def wrapper(
        graph_position: GraphNode | None,
        sequence_still_to_align: list[BaseLike],
        at_graph_start: bool,
        at_sequence_start: bool,
        alignment_bonus: float,
    ):
        fingerprint = (
            graph_position,
            len(sequence_still_to_align),
            at_graph_start,
            at_sequence_start,
            alignment_bonus,
        )
        if fingerprint in cache:
            return cache[fingerprint]
        result = func(
            graph_position,
            sequence_still_to_align,
            at_graph_start,
            at_sequence_start,
            alignment_bonus,
        )
        cache[fingerprint] = result
        return result

    def wiper():
        nonlocal cache
        cache = {}

    return wrapper, wiper


def dp_memoized_function(
    graph_position: GraphNode | None,
    sequence_still_to_align: list[BaseLike],
    at_graph_start: bool,
    at_sequence_start: bool,
    alignment_bonus: float,
) -> DpOption:
    """
    returns:
        - minimum cost, way through the graph
    """
    # Base cases
    if sequence_still_to_align == []:
        # choose end statement
        return DpOption(
            0,
            [TracePoint(DpOperation.INSERT_ALL_END, graph_position)]
            if graph_position is not None
            else [],
        )

    if graph_position is None:
        # choose end statement
        return DpOption(
            0,
            [
                TracePoint(
                    DpOperation.DELETE_ALL_END,
                    graph_position,
                    end_cut_out_sequence=sequence_still_to_align,
                )
            ],
        )

    successors = (
        graph_position.successors if graph_position.successors != [] else [None]
    )

    # Recursive case
    options: list[DpOption] = []

    # for all graph-successors
    for next_graph_position in successors:
        # delete from sequence
        sub_option = dp_memoized_function(
            graph_position,
            sequence_still_to_align[1:],
            at_graph_start=at_graph_start,
            at_sequence_start=False,
            alignment_bonus=alignment_bonus,
        )
        sub_cost, trace = sub_option.cost, sub_option.trace
        if at_graph_start:
            # if at graph_start, we can delete for free (meaning a flexible starting position)
            trace = [
                TracePoint(DpOperation.DELETE_START, graph_position)
            ] + trace.copy()

            options.append(
                DpOption(
                    sub_cost,
                    trace,
                )
            )
        else:
            trace = [TracePoint(DpOperation.DELETE, graph_position)] + trace.copy()

            options.append(
                DpOption(
                    DELETION_PENALTY + sub_cost,
                    trace,
                )
            )

        # insert into sequence
        sub_option = dp_memoized_function(
            next_graph_position,
            sequence_still_to_align,
            at_graph_start=False,
            at_sequence_start=at_sequence_start,
            alignment_bonus=alignment_bonus,
        )
        sub_cost, trace = sub_option.cost, sub_option.trace

        if at_sequence_start:
            # if at sequence start, we can insert for free (meaning a flexible starting position)
            trace = [
                TracePoint(DpOperation.INSERT_START, graph_position)
            ] + trace.copy()
            options.append(
                DpOption(
                    sub_cost,
                    trace,
                )
            )
        else:
            trace = [TracePoint(DpOperation.INSERT, graph_position)] + trace.copy()

            options.append(
                DpOption(
                    INSERTION_PENALTY + sub_cost,
                    trace,
                )
            )

        # replace in sequence
        sub_option = dp_memoized_function(
            next_graph_position,
            sequence_still_to_align[1:],
            at_graph_start=False,
            at_sequence_start=False,
            alignment_bonus=alignment_bonus,
        )
        sub_cost, trace = sub_option.cost, sub_option.trace

        cost_here = graph_position.base.dp_conversion_penalty(
            sequence_still_to_align[0]
        )

        trace = [
            TracePoint(
                DpOperation.MATCH  # same event. Does not mean there is no penalty
                if graph_position.base == sequence_still_to_align[0]
                else DpOperation.REPLACE,
                graph_position,
                replacement_cost=cost_here,
            )
        ] + trace.copy()

        options.append(
            DpOption(
                cost_here + sub_cost,
                trace,
            )
        )

        # flexible start option

    # return min of all possibilities
    # give advantage to longer actually aligned sequences
    return min(options, key=lambda x: x.matching_adjusted_cost(alignment_bonus))


dp_memoized_function, wiper = carefully_chached(dp_memoized_function)


def wiped_dp_memoized_function(
    graph_position: GraphNode | None,
    sequence_still_to_align: list[BaseLike],
    at_graph_start: bool,
    at_sequence_start: bool,
    alignment_bonus: float,
) -> DpOption:
    wiper()
    return dp_memoized_function(
        graph_position,
        sequence_still_to_align,
        at_graph_start,
        at_sequence_start,
        alignment_bonus=alignment_bonus,
    )


def initial_graph_of(sequence: list[BaseLike]) -> Graph:
    read_node = create_read_node_chain(sequence)
    return initial_graph_of_read_node(read_node)


def initial_graph_of_read_node(read_node: ReadNode) -> Graph:
    """Creates initial graph from sequence"""
    graph = Graph([])
    graph_nodes = []

    current_read_node = read_node
    while current_read_node is not None:
        layer = AssoziatedLayer()
        layer.graph = graph
        graph_node = GraphNode()
        layer.add(graph_node)
        graph_node.add(current_read_node)
        graph_nodes.append(graph_node)
        current_read_node = current_read_node.next

    graph.start_nodes.append(graph_nodes[0])
    return graph


def create_read_node_chain(sequence: list[BaseLike]) -> ReadNode:
    """Creates a chain of read nodes from sequence"""
    current_read_node = ReadNode(sequence[0])
    head = current_read_node
    for base in sequence[1:]:
        next_read_node = ReadNode(base)
        current_read_node.next = next_read_node
        current_read_node = next_read_node
    return head


def add_trace_to_graph(sequence: list[BaseLike], trace: list[TracePoint]):
    read_node = create_read_node_chain(sequence)
    return add_trace_read_node_to_graph(read_node=read_node, trace=trace)


def add_trace_read_node_to_graph(read_node: ReadNode, trace: list[TracePoint]):
    already_new_start_node_added = False

    graph = trace[0].graph_node.layer.graph

    for trace_node in trace:
        if trace_node.operation == DpOperation.MATCH:
            # add to existing graph node
            trace_node.graph_node.add(read_node)
            read_node = read_node.next

        elif trace_node.operation == DpOperation.REPLACE:
            # create (new) parallel graph node
            new_graph_node = GraphNode()
            trace_node.graph_node.layer.add(new_graph_node)
            new_graph_node.add(read_node)
            read_node = read_node.next

        elif trace_node.operation in [DpOperation.INSERT, DpOperation.INSERT_START]:
            # ignore the existing graph node (jump over it)
            pass

        elif trace_node.operation in [DpOperation.DELETE, DpOperation.DELETE_START]:
            # create new layer before existing one
            new_layer = AssoziatedLayer()
            new_layer.graph = graph
            new_graph_node = GraphNode()
            new_layer.add(new_graph_node)
            new_graph_node.add(read_node)
            # should be automatically connected to the other layers via the read-linked-list
            read_node = read_node.next

            # if we have multiple delete_starts in a row, only add the first one
            if not already_new_start_node_added:
                new_layer.graph.start_nodes.append(new_graph_node)
                already_new_start_node_added = True

        elif trace_node.operation == DpOperation.INSERT_ALL_END:
            pass

        elif trace_node.operation == DpOperation.DELETE_ALL_END:
            for base in trace_node.end_cut_out_sequence:
                new_layer = AssoziatedLayer()
                new_layer.graph = graph
                new_graph_node = GraphNode()
                new_layer.add(new_graph_node)
                new_graph_node.add(read_node)
                # should be automatically connected to the other layers via the read-linked-list
                read_node = read_node.next
                # should be an invariant that this is never none and finished at the end

        else:
            raise ValueError(f"Unknown operation: {trace_node.operation}")


def dp_with_graph(
    sequence: list[BaseLike], graph: Graph, alignment_bonus: float
) -> DpOption:
    """Deals with multiple start-points.
    Chooses best start-point based on alignment_bonus.

    Important: Save partial results from the handling of one start-node for the other, to avoid massive recalculation.
    """
    wiper()  # important to reset cache
    options = []
    for start_node in graph.start_nodes:
        option = dp_memoized_function(
            start_node, sequence, True, True, alignment_bonus=alignment_bonus
        )
        options.append(option)
    return min(options, key=lambda x: x.matching_adjusted_cost(alignment_bonus))


def add_to_graph(
    sequence: list[BaseLike], graph: Graph, alignment_bonus: float
) -> DpOption:
    option = dp_with_graph(sequence, graph, alignment_bonus=alignment_bonus)
    add_trace_to_graph(sequence, option.trace)
    return option


def sequence_of_read(read: Read, *, probabilistic: bool) -> list[BaseLike]:
    return (
        make_regular(*read.predicted_text)
        if not probabilistic
        else make_uncertain_regular(*read.uncertain_text)
    )


def create_linked_read_node(read: Read, *, probabilistic: bool) -> ReadNode:
    sequence = sequence_of_read(read, probabilistic=probabilistic)
    read_node = create_read_node_chain(sequence)
    read.read_node_pointer = read_node
    return read_node


def add_read_to_graph(
    read: Read, graph: Graph, *, alignment_bonus: float, probabilistic: bool
) -> DpOption:
    sequence = sequence_of_read(read, probabilistic=probabilistic)
    option = dp_with_graph(sequence, graph, alignment_bonus=alignment_bonus)
    add_trace_read_node_to_graph(
        create_linked_read_node(read, probabilistic=probabilistic), option.trace
    )
    return option


def consent_of_graph(graph: Graph) -> list[BaseLike]:
    """Returns the most likely sequence from the graph"""
    current_graph_node = graph.start_nodes[0]  # TODO: assumption only one start
    sequence = []

    current_layer = current_graph_node.layer

    while current_layer is not None:
        layer_consent_votes = []
        for graph_node in current_layer.graph_nodes:
            for read_node in graph_node.read_nodes:
                layer_consent_votes.append(read_node.base)
        class_of_base = layer_consent_votes[
            0
        ].__class__  # find out which classes consent function to use
        sequence.append(class_of_base.consent(layer_consent_votes))

        next_layer_candidates = []
        for graph_node in current_layer.graph_nodes:
            for read_node in graph_node.read_nodes:
                next_layer_candidates.append(
                    read_node.next.graph_node.layer
                    if read_node.next is not None
                    else None
                )
        next_layer = majority_vote_none_only_if_no_other(next_layer_candidates)
        current_layer = next_layer
    return sequence


def consent_at_read(read: Read) -> list[BaseLike]:
    if read.read_node_pointer is None:
        raise ValueError("No read node pointer found at call of consent_at_read")

    end_pointer: ReadNode = read.read_node_pointer
    while end_pointer.next is not None:
        end_pointer = end_pointer.next

    return consent_at_read_nodes(read.read_node_pointer, end_pointer)


def consent_at_read_nodes(
    start_read_node: ReadNode, end_read_node: ReadNode
) -> list[BaseLike]:
    current_graph_node = start_read_node.graph_node
    sequence: list[BaseLike] = []

    current_layer: AssoziatedLayer = current_graph_node.layer

    while True:
        layer_consent_votes = []
        for graph_node in current_layer.graph_nodes:
            for read_node in graph_node.read_nodes:
                layer_consent_votes.append(read_node.base)
        class_of_base = layer_consent_votes[
            0
        ].__class__  # find out which classes consent function to use
        sequence.append(class_of_base.consent(layer_consent_votes))

        next_layer_candidates = []
        for graph_node in current_layer.graph_nodes:
            for read_node in graph_node.read_nodes:
                next_layer_candidates.append(
                    read_node.next.graph_node.layer
                    if read_node.next is not None
                    else None
                )
        next_layer = majority_vote_none_only_if_no_other(next_layer_candidates)

        # do ... while - construction doing a consent step until (including) the end read node is found
        if end_read_node in [
            r for g in current_layer.graph_nodes for r in g.read_nodes
        ]:
            break

        current_layer = next_layer
        if current_layer is  None:   #todo: why is this happening? Should never occur, but it does. Is the solution correct??
            print(next_layer_candidates)
            break
            #raise ValueError("No next layer found")

    return sequence


def consent_corrected_read(read: Read,*, probabilistic:bool) -> Read:
    result = read.copy()
    consent = consent_at_read(result)
    result.uncertain_text = (
        [c.base for c in consent]
        if probabilistic
        else list(map(certain_uncertainty_generator,[c.base for c in consent]))
    )
    return result


def multiple_sequence_alignment(
    sequences: list[list[BaseLike]], alignment_bonus: float
) -> Graph:
    graph = initial_graph_of(sequences[0])
    for sequence in sequences[1:]:
        add_to_graph(sequence, graph, alignment_bonus=alignment_bonus)
    return graph


def read_multiple_sequence_alignment(
    reads: list[Read], *, alignment_bonus: float, probabilistic: bool
):
    graph = initial_graph_of_read_node(
        create_linked_read_node(reads[0], probabilistic=probabilistic)
    )
    for read in reads[1:]:
        add_read_to_graph(
            read, graph, alignment_bonus=alignment_bonus, probabilistic=probabilistic
        )
    return graph


def read_consent(
    reads: list[Read], *, alignment_bonus: float, as_read=False, probabilistic=False
):
    # print("reads: ", reads)
    if probabilistic:
        base_like_reads = list(
            map(
                lambda read: [
                    PositionalWeightMatrixBase(b) for b in read.uncertain_text
                ],
                reads,
            )
        )
    else:
        base_like_reads = list(
            map(
                lambda read: [
                    RegularBase(b) for b in most_likely_restorer(read.uncertain_text)
                ],
                reads,
            )
        )

    graph = multiple_sequence_alignment(
        base_like_reads, alignment_bonus=alignment_bonus
    )

    consent = consent_of_graph(graph)
    consent = [c.base for c in consent]
    if not as_read:
        return consent
    # raise NotImplementedError("Not implemented yet")
    # TODO: may go outsidethe read dna if something was misaligned

    start = reads[0].start_position
    end = start + len(consent)
    try:
        result = create_read_with_dna_pointer(reads[0].dna_pointer, start, end)
    except IndexError:
        print("Failed to read what real dna was")
        result = Read(C * len(consent), start, end)  # good idea to fill with Cs ?

    if probabilistic:
        result.uncertain_text = consent[: len(reads[0].original_text)]
    else:
        result.uncertain_text = list(map(certain_uncertainty_generator, consent))[
            : len(reads[0].original_text)
        ]  # TODO: always correct?
    return result


def correct_reads_with_consens(
    reads: list[Read], *, alignment_bonus: float, probabilistic: bool
):

    graph = read_multiple_sequence_alignment(
        reads, alignment_bonus=alignment_bonus, probabilistic=probabilistic
    )
    return [consent_corrected_read(r, probabilistic=probabilistic) for r in reads] #todo

    ### Old naive version

    # results = []
    # for i in range(len(reads)):
    #    ordered_reads = ([reads[i]] + reads[:i] + reads[i + 1 :]).copy()
    #    results.append(
    #        read_consent(ordered_reads, as_read=True, probabilistic=probabilistic, alignment_bonus=alignment_bonus)
    #    )
    #    # print(results)
    # return results


def edit_distance(s1: list[BaseLike], s2: list[BaseLike],*, alignment_bonus):
    start_graph_node = initial_graph_of(s1).start_nodes[0]
    option = wiped_dp_memoized_function(
        start_graph_node, s2, True, True, alignment_bonus=alignment_bonus
    )
    return option.cost

def read_edit_distance(read:Read,*, alignment_bonus:float):
    return edit_distance(make_regular(*read.original_text), make_regular(*read.predicted_text), alignment_bonus=alignment_bonus)

def colored_edit_string(s1: list[BaseLike], s2: list[BaseLike],*, alignment_bonus:float):
    start_graph_node = initial_graph_of(s1).start_nodes[0]
    option = wiped_dp_memoized_function(
        start_graph_node, s2, True, True, alignment_bonus=alignment_bonus
    )
    edit_string = ""
    for t in option.trace:
        if t.operation in [DpOperation.DELETE, DpOperation.DELETE_START]:
            edit_string += colored_string("X", RED)
        elif t.operation in [DpOperation.INSERT, DpOperation.INSERT_START]:
            edit_string += colored_string("X",GREEN)
        elif t.operation == DpOperation.MATCH:
            edit_string += colored_string("X",WHITE)
        elif t.operation == DpOperation.REPLACE:
            edit_string += colored_string("X",BLUE)
        elif t.operation == DpOperation.INSERT_ALL_END:
            edit_string += colored_string("...",GREEN)
        elif t.operation == DpOperation.DELETE_ALL_END:
            edit_string += colored_string("...",RED)
        elif t.operation == DpOperation.DELETE_START:
            edit_string += colored_string("X",GREEN)
        else:
            raise("Unknown DpOperation")
    return edit_string


def print_it(f:t.Callable[[t.Any],str]):
    def wrapper(*args,**kwargs):
        print(f(*args,**kwargs))
    return wrapper


def show_read_errors(read:Read,*, alignment_bonus:float):
    return colored_edit_string(make_regular(*read.original_text), make_regular(*read.predicted_text), alignment_bonus=alignment_bonus)


def show_read_intendet_errors(read:Read,*, alignment_bonus:float):
    return " "*read.start_position + show_read_errors(read, alignment_bonus=alignment_bonus)

print_read_errors = print_it(show_read_errors)
print_read_intendet_errors = print_it(show_read_intendet_errors)

def dna_distance_error_rate(dna: str, reads: list[Read], alignment_bonus: float):
    individual_error_rates = []
    total_distance = 0
    total_length = 0
    for read in reads:
        cost = edit_distance(
            make_regular(*dna),
            make_regular(*read.predicted_text),
            alignment_bonus=alignment_bonus,
        )
        length = len(read.original_text)
        individual_error_rates.append(cost / length)
        total_distance += cost
        total_length += length
    print(individual_error_rates)
    return total_distance / total_length


def percent_similarity_func(value1, value2, *, percent=0.2):
    return value1 / value2 <= 1 + percent and value2 / value1 <= 1 + percent


def check_dna_distance_error_rate_suitable(
    dna: str,
    reads: list[Read],
    alignment_bonus: float,
    similarity_criterium=percent_similarity_func,
):
    real_result = most_likely_restorer_error_rate(reads)
    arbitrary_alignment_result = dna_distance_error_rate(
        dna, reads, alignment_bonus=alignment_bonus
    )
    if not similarity_criterium(real_result, arbitrary_alignment_result):
        raise ValueError(
            "Assumption broken: Allowing reads to align arbitrarely significantly reduces perceived error rate. In this case,"
            + f"arbitrary alignment ({arbitrary_alignment_result}) and actual read origin alignment scores ({real_result}) cannot be used interchangeably."
        )


def most_likely_restorer_error_rate(reads: list[Read], *, alignment_bonus: float=2): # todo: hardcoded alignment bonus!!!
    distances = [
       read_edit_distance(r, alignment_bonus=alignment_bonus) / len(r.original_text) # this is not any more counting actual mistakes, but best available alignment... (problematic??)
       for r in reads
    ]
    print(distances)
    return sum(distances) / len(distances)


def tes_dp_same():
    # graph = initial_graph_of(make_regular(A, T, C, G, T, T, C))
    # add_to_graph(            make_regular(A, T, C, A, T, T, C), graph, alignment_bonus=TEST_ALINGMENT_BONUS)
    # add_to_graph(            make_regular(A, T, C, A, T, T, C), graph, alignment_bonus=TEST_ALINGMENT_BONUS)
    graph = read_multiple_sequence_alignment(
        [
            Read("ATCGTTC", 0, 7, uncertainty_generator=certain_uncertainty_generator),
            Read("ATCATTC", 0, 7, uncertainty_generator=certain_uncertainty_generator),
            Read("ATCATTC", 0, 7, uncertainty_generator=certain_uncertainty_generator),
        ],
        alignment_bonus=0.7,
        probabilistic=False,
    )


if __name__ == "__main__":
    tes_dp_same()
