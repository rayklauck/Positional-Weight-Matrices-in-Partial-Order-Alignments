from enum import Enum
from functools import cache
from pot_correction import *
from abc import ABC, abstractmethod

DELETION_PENALTY = 1
INSERTION_PENALTY = 1
ALIGNMENT_BONUS = 0.9  # adjust based on the consideration: how many aligned bases is one unaligned worth?


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

    def add(self, graph_node: GraphNode):
        self.graph_nodes.append(graph_node)
        graph_node.layer = self  # set backreference


@dataclass
class Graph:
    start_nodes: list[GraphNode]


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

    def matching_adjusted_cost(self):
        alignment_bonus = 0
        for trace_point in self.trace:
            if trace_point.operation == DpOperation.MATCH:
                alignment_bonus += ALIGNMENT_BONUS
            elif trace_point.operation == DpOperation.REPLACE:
                alignment_bonus += ALIGNMENT_BONUS * (1 - trace_point.replacement_cost)

        return self.cost - alignment_bonus


@dataclass
class TracePoint:
    operation: DpOperation
    graph_node: GraphNode
    replacement_cost: float | None = None  # for replace operations only
    end_cut_out_sequence: list[BaseLike] | None = None  # for end operations only


def carefully_chached(
    func: t.Callable[[GraphNode, int, bool, bool], tuple[float, list[TracePoint]]]
):
    cache: dict[tuple[GraphNode, int, bool, bool], tuple[float, list[TracePoint]]] = {}

    def wrapper(
        graph_position: GraphNode | None,
        sequence_still_to_align: list[BaseLike],
        at_graph_start: bool,
        at_sequence_start: bool,
    ):
        fingerprint = (
            graph_position,
            len(sequence_still_to_align),
            at_graph_start,
            at_sequence_start,
        )
        if fingerprint in cache:
            return cache[fingerprint]
        result = func(
            graph_position, sequence_still_to_align, at_graph_start, at_sequence_start
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
) -> tuple[float, list[TracePoint]]:
    """
    returns:
        - minimum cost, way through the graph
    """
    # Base cases
    if sequence_still_to_align == []:
        # choose end statement
        return (
            0,
            [TracePoint(DpOperation.INSERT_ALL_END, graph_position)]
            if graph_position is not None
            else [],
        )

    if graph_position is None:
        # choose end statement
        return 0, [
            TracePoint(
                DpOperation.DELETE_ALL_END,
                graph_position,
                end_cut_out_sequence=sequence_still_to_align,
            )
        ]

    successors = (
        graph_position.successors if graph_position.successors != [] else [None]
    )

    # Recursive case
    options: list[DpOption] = []

    # for all graph-successors
    for next_graph_position in successors:
        # delete from sequence
        sub_cost, trace = dp_memoized_function(
            graph_position,
            sequence_still_to_align[1:],
            at_graph_start=at_graph_start,
            at_sequence_start=False,
        )
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
        sub_cost, trace = dp_memoized_function(
            next_graph_position,
            sequence_still_to_align,
            at_graph_start=False,
            at_sequence_start=at_sequence_start,
        )

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
        sub_cost, trace = dp_memoized_function(
            next_graph_position,
            sequence_still_to_align[1:],
            at_graph_start=False,
            at_sequence_start=False,
        )

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
    best_option = min(options, key=lambda x: x.matching_adjusted_cost())
    return best_option.cost, best_option.trace


dp_memoized_function, wiper = carefully_chached(dp_memoized_function)


def wiped_dp_memoized_function(
    graph_position: GraphNode | None,
    sequence_still_to_align: list[BaseLike],
    at_graph_start: bool,
    at_sequence_start: bool,
) -> tuple[float, list[TracePoint]]:
    wiper()
    return dp_memoized_function(
        graph_position, sequence_still_to_align, at_graph_start, at_sequence_start
    )


def initial_graph_of(sequence: list[BaseLike]) -> Graph:
    """Creates initial graph from sequence"""
    read_node_chain = create_read_node_chain(sequence)
    graph = Graph([])
    graph_nodes = []

    current_read_node = read_node_chain
    while current_read_node is not None:
        layer = AssoziatedLayer()
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
            new_graph_node = GraphNode()
            new_layer.add(new_graph_node)
            new_graph_node.add(read_node)
            # should be automatically connected to the other layers via the read-linked-list
            read_node = read_node.next

        elif trace_node.operation == DpOperation.INSERT_ALL_END:
            pass

        elif trace_node.operation == DpOperation.DELETE_ALL_END:
            for base in trace_node.end_cut_out_sequence:
                new_layer = AssoziatedLayer()
                new_graph_node = GraphNode()
                new_layer.add(new_graph_node)
                new_graph_node.add(read_node)
                # should be automatically connected to the other layers via the read-linked-list
                read_node = read_node.next
                # should be an invariant that this is never none and finished at the end

        else:
            raise ValueError(f"Unknown operation: {trace_node.operation}")


def add_to_graph_start_node(sequence: list[BaseLike], start_graph_node: GraphNode):
    _, trace = wiped_dp_memoized_function(start_graph_node, sequence, True, True)
    # print([t.operation.name for t in trace])
    add_trace_to_graph(sequence, trace)


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


def multiple_sequence_alignment(sequences: list[list[BaseLike]]) -> Graph:
    graph = initial_graph_of(sequences[0])
    for sequence in sequences[1:]:
        add_to_graph_start_node(sequence, graph.start_nodes[0])
    return graph


def read_consent(reads: list[Read], as_read=False, probabilistic=False):
    print("reads: ", reads)
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

    print("breads: ", base_like_reads)
    graph = multiple_sequence_alignment(base_like_reads)
    print("gr: ", graph)

    consent = consent_of_graph(graph)
    print("consent: ", consent)
    consent = [c.base for c in consent]
    if not as_read:
        return consent
    raise NotImplementedError("Not implemented yet")
    # TODO: may go outsidethe read dna if something was misaligned
    # try:
        # result = create_read_with_dna_pointer(
            # reads[0].dna_pointer, reads[0].start_position, reads[0].start_position + len(consent))
    # except IndexError:
        # result = 
# 
    # if probabilistic:
        # result.uncertain_text = consent
    # else:
        # result.uncertain_text = list(map(certain_uncertainty_generator, consent))[
            # : len(result.original_text)
        # ]  # TODO: always correct?
    # return result
# 

def correct_reads_with_consens(reads: list[Read], probabilistic=False):
    results = []
    for i in range(len(reads)):
        ordered_reads = ([reads[i]] + reads[:i] + reads[i + 1 :]).copy()
        results.append(
            read_consent(ordered_reads, as_read=True, probabilistic=probabilistic)
        )
        # print(results)
    return results


def tes_dp_same():
    g = multiple_sequence_alignment([
            make_regular(C, T, G, G, A, C),
            make_regular(G, G, A, C, C, T),
    ]
        )
    print(g)
    assert consent_of_graph(
        g
    ) == make_regular(C, T, G, G, A, C, C, T)


if __name__ == "__main__":
    tes_dp_same()
