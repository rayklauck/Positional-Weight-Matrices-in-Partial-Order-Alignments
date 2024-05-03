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


class ReadNode:
    """LinkedList for one read"""

    def __init__(self, base: BaseLike, next: t.Union["ReadNode" , None] = None):
        self.next = next
        self.base = base
        self.graph_node = None


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
        return list({read_node.next.graph_node for read_node in self.read_nodes if read_node.next is not None})

    @property
    def base(self) -> BaseLike:
        """Guaranteed that all read_nodes will have the same base here.
        And there will be never a GraphNode with no ReadNode."""
        return self.read_nodes[0].base
    
    def __repr__(self):
        return f"GraphNode({self.base})"


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
    """Base with associated probability"""

    def __init__(self, base: str):
        self.base = base

    def __eq__(self, value: object) -> bool:
        return self.base == value.base

    def dp_conversion_penalty(self, value: object) -> float:
        return self.base != value.base
    
    def __repr__(self):
        return self.base


def make_regular(*bases: list[str]) -> list[RegularBase]:
    return [RegularBase(base) for base in bases]

class DpOperation(Enum):
    """Enum for DP operations"""
    DELETE = 1
    INSERT = 2
    REPLACE = 3
    MATCH = 4
    END = 5


@dataclass
class DpOption:
    """Option for DP"""
    cost: float
    trace: list['TracePoint']


@dataclass
class TracePoint:
    operation: DpOperation
    graph_node: GraphNode


def carefully_chached(func):
    cache: dict[ tuple[GraphNode, int], tuple[float,list[TracePoint]]] = {}
    def wrapper(graph_position: GraphNode | None, sequence_still_to_align: list[BaseLike]):
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
) -> tuple[float,list[TracePoint]]:
    """
    returns:
        - minimum cost, way through the graph
    """
    print('CALL: ',graph_position, sequence_still_to_align)

    # Base cases
    if sequence_still_to_align == []:
        # choose end statement
        return 0, [TracePoint(DpOperation.END, graph_position)] if graph_position is not None else []

    if graph_position is None:
        # must insert all remaining bases

        sub_cost, trace = dp_memoized_function(None, sequence_still_to_align[1:])
        return DELETION_PENALTY + sub_cost, [TracePoint(DpOperation.DELETE, None)] + trace.copy()

        # todo
        # or should this case also fall under a free end statement (other relative to this?)
        
       
        
    successors = graph_position.successors if graph_position.successors != [] else [None]

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

        trace = [TracePoint(DpOperation.REPLACE if cost_here > 0 else DpOperation.MATCH, graph_position)] + trace.copy()

        options.append(
            DpOption(
                cost_here + sub_cost,
                trace,
            )
        )

    # return min of all possibilities
    best_option = min(options, key=lambda x: x.cost)
    return best_option.cost, best_option.trace


dp_memoized_function, wiper = carefully_chached(dp_memoized_function)

def wiped_dp_memoized_function(
    graph_position: GraphNode | None, sequence_still_to_align: list[BaseLike]
) -> tuple[float,list[TracePoint]]:
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
    




def tes_dp_same():
    start_graph_node = initial_graph_of(make_regular(A, T, G)).start_nodes[0]
    sequence = make_regular(A, C, G)
    add_to_graph_start_node(sequence, start_graph_node)
    sequence = make_regular(A, C, G)
    cost, trace = wiped_dp_memoized_function(start_graph_node, sequence)
    print(trace)
    assert cost == 0


if __name__ == "__main__":
    tes_dp_same()
