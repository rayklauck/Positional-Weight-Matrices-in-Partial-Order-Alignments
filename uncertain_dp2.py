from enum import Enum
from functools import cache
from pot_correction import *
from abc import ABC, abstractmethod

DELETION_PENALTY = 1
INSERTION_PENALTY = 1


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
        return f"N({self.base}, {self.read_nodes}) -> {self.successors}"


class AssoziatedLayer:
    """Layer for specific position. Contains GraphNodes for occuring bases"""

    def __init__(self):
        self.graph_nodes: list[GraphNode] = []

    def add(self, graph_node: GraphNode):
        self.graph_nodes.append(graph_node)

        # set backreference
        graph_node.layer = self


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
    def consent(sequence: list['PositionalWeightMatrixBase']) -> 'PositionalWeightMatrixBase':
        histogram = [0,0,0,0]
        for b in sequence:
            for i, v in enumerate(b.base):
                histogram[i] += v
        
        return PositionalWeightMatrixBase(normalize(histogram))


    def __repr__(self):
        #return "~" + most_likely_base_restorer(self.base)
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
    END = 5
    START = 6


@dataclass
class DpOption:
    """Option for DP"""

    cost: float
    trace: list["TracePoint"]


@dataclass
class TracePoint:
    operation: DpOperation
    graph_node: GraphNode


def carefully_chached(func):
    cache: dict[tuple[GraphNode, int], tuple[float, list[TracePoint]]] = {}

    def wrapper(
        graph_position: GraphNode | None, sequence_still_to_align: list[BaseLike]
    ):
        fingerprint = graph_position, len(sequence_still_to_align)
        if fingerprint in cache:
            return cache[fingerprint]
        result = func(graph_position, sequence_still_to_align)
        cache[fingerprint] = result
        return result

    def wiper():
        nonlocal cache
        cache = {}

    return wrapper, wiper


def dp_memoized_function(
    graph_position: GraphNode | None, sequence_still_to_align: list[BaseLike]
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
            [TracePoint(DpOperation.END, graph_position)]
            if graph_position is not None
            else [],
        )

    if graph_position is None:
        # must insert all remaining bases

        sub_cost, trace = dp_memoized_function(None, sequence_still_to_align[1:])
        return (
            DELETION_PENALTY + sub_cost,
            [TracePoint(DpOperation.DELETE, None)] + trace.copy(),
        )

        # todo
        # or should this case also fall under a free end statement (other relative to this?)

    successors = (
        graph_position.successors if graph_position.successors != [] else [None]
    )

    # Recursive case
    options: list[DpOption] = []

    # for all graph-successors
    for next_graph_position in successors:
        # delete from sequence
        sub_cost, trace = dp_memoized_function(
            graph_position, sequence_still_to_align[1:]
        )
        trace = [TracePoint(DpOperation.DELETE, graph_position)] + trace.copy()

        options.append(
            DpOption(
                DELETION_PENALTY + sub_cost,
                trace,
            )
        )

        # insert into sequence
        sub_cost, trace = dp_memoized_function(
            next_graph_position, sequence_still_to_align
        )
        trace = [TracePoint(DpOperation.INSERT, graph_position)] + trace.copy()

        options.append(
            DpOption(
                INSERTION_PENALTY + sub_cost,
                trace,
            )
        )

        # replace in sequence
        sub_cost, trace = dp_memoized_function(
            next_graph_position, sequence_still_to_align[1:]
        )

        cost_here = graph_position.base.dp_conversion_penalty(
            sequence_still_to_align[0]
        )

        trace = [
            TracePoint(
                DpOperation.MATCH # same event. Does not mean there is no penalty
                if graph_position.base == sequence_still_to_align[0]
                else DpOperation.REPLACE,
                graph_position,
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
    best_option = min(options, key=lambda x: x.cost)
    return best_option.cost, best_option.trace


dp_memoized_function, wiper = carefully_chached(dp_memoized_function)


def wiped_dp_memoized_function(
    graph_position: GraphNode | None, sequence_still_to_align: list[BaseLike]
) -> tuple[float, list[TracePoint]]:
    wiper()
    return dp_memoized_function(graph_position, sequence_still_to_align)


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

        elif trace_node.operation == DpOperation.INSERT:
            # ignore the existing graph node (jump over it)
            pass

        elif trace_node.operation == DpOperation.DELETE:
            # create new layer before existing one
            new_layer = AssoziatedLayer()
            new_graph_node = GraphNode()
            new_layer.add(new_graph_node)
            new_graph_node.add(read_node)
            # should be automatically connected to the other layers via the read-linked-list
            read_node = read_node.next

        elif trace_node.operation == DpOperation.END:
            # sequence successfully aligned
            return

        else:
            raise ValueError(f"Unknown operation: {trace_node.operation}")


def add_to_graph_start_node(sequence: list[BaseLike], start_graph_node: GraphNode):
    _, trace = wiped_dp_memoized_function(start_graph_node, sequence)
    add_trace_to_graph(sequence, trace)


def consent_of_graph(graph: Graph) -> t.Iterator[BaseLike]:
    """Returns the most likely sequence from the graph"""
    current_graph_node = graph.start_nodes[0]  # TODO: assumption only one start
    sequence = []

    current_layer = current_graph_node.layer

    while current_layer is not None:
        layer_consent_votes = []
        for graph_node in current_layer.graph_nodes:
            for read_node in graph_node.read_nodes:
                layer_consent_votes.append(read_node.base)
        class_of_base = layer_consent_votes[0].__class__ # find out which classes consent function to use
        sequence.append(class_of_base.consent(layer_consent_votes))

        next_layer_candidates = []
        for graph_node in current_layer.graph_nodes:
            for read_node in graph_node.read_nodes:
                next_layer_candidates.append(read_node.next.graph_node.layer if read_node.next is not None else None)
        next_layer = majority_vote(next_layer_candidates)
        current_layer = next_layer
    return sequence
        

def multiple_sequence_alignment(sequences: list[list[BaseLike]]) -> Graph:
    graph = initial_graph_of(sequences[0])
    for sequence in sequences[1:]:
        add_to_graph_start_node(sequence, graph.start_nodes[0])
    return graph


def read_consent(reads: list[Read]):
    rounded_base_like_reads = list(map(
        lambda read: [RegularBase(b)for b in most_likely_restorer(read.uncertain_text)]
        ,reads))
    graph = multiple_sequence_alignment(rounded_base_like_reads)
    return consent_of_graph(graph)

def probabilistic_read_consens():
    pass


def tes_dp_same():
    print(consent_of_graph(
        multiple_sequence_alignment([
            make_uncertain_regular(just(A), just(T), just(C)),
            make_uncertain_regular(just(A), just(G), just(C)),
        ])
    )[2].base )
    assert False



if __name__ == "__main__":
    tes_dp_same()
