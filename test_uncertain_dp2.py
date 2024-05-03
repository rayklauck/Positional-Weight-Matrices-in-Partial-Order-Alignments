from uncertain_dp2 import *


def test_graph_initialization():
    graph = initial_graph_of([RegularBase(C), RegularBase(G), RegularBase(T)]) 

    assert len(graph.start_nodes) == 1
    start_node = graph.start_nodes[0]

    assert start_node.base == RegularBase(C)
    assert start_node.successors[0].base == RegularBase(G)
    assert type(start_node.successors[0]) == GraphNode
    assert start_node.successors[0].successors[0].base == RegularBase(T)
    assert start_node.successors[0].successors[0].successors == []
    assert start_node.layer != start_node.successors[0].layer


def test_graph_initialization_edge():
    graph = initial_graph_of([RegularBase(C)]) 

    assert len(graph.start_nodes) == 1
    start_node = graph.start_nodes[0]

    assert start_node.base == RegularBase(C)
    assert start_node.successors == []


def test_graph_initialization_layer():
    start_graph_node = initial_graph_of([RegularBase(C)]).start_nodes[0]
    start_graph_node.layer.graph_nodes == [start_graph_node]



def test_dp_same_one_base():
    start_graph_node = initial_graph_of(make_regular(A)).start_nodes[0]
    sequence = make_regular(A)
    cost, trace = dp_memoized_function(start_graph_node, sequence)
    assert cost == 0
    assert len(trace) == 1
    assert trace[0].operation == DpOperation.MATCH
    assert trace[0].graph_node == start_graph_node


def test_dp_same_two_bases():
    start_graph_node = initial_graph_of(make_regular(C, G)).start_nodes[0]
    sequence = make_regular(C, G)
    cost, trace = dp_memoized_function(start_graph_node, sequence)
    assert cost == 0
    assert len(trace) == 2
    assert trace[0].operation == DpOperation.MATCH
    assert trace[0].graph_node.base == RegularBase(C)
    assert trace[1].operation == DpOperation.MATCH
    assert trace[1].graph_node.base == RegularBase(G)


def test_dp_same_multiple_bases():
    start_graph_node = initial_graph_of(make_regular(C, G, A, T)).start_nodes[0]
    sequence = make_regular(C, G, A, T)
    cost, trace = dp_memoized_function(start_graph_node, sequence)
    assert cost == 0
    assert len(trace) == 4
    assert trace[0].operation == DpOperation.MATCH
    assert trace[0].graph_node.base == RegularBase(C)
    assert trace[1].operation == DpOperation.MATCH
    assert trace[1].graph_node.base == RegularBase(G)
    assert trace[2].operation == DpOperation.MATCH
    assert trace[2].graph_node.base == RegularBase(A)
    assert trace[3].operation == DpOperation.MATCH
    assert trace[3].graph_node.base == RegularBase(T)


def test_dp_one_replacement():
    start_graph_node = initial_graph_of(make_regular(C, G, A)).start_nodes[0]
    sequence = make_regular(C, T, A)
    cost, trace = dp_memoized_function(start_graph_node, sequence)
    assert cost == RegularBase(T).dp_conversion_penalty(RegularBase(G))
    assert len(trace) == 3
    assert trace[0].operation == DpOperation.MATCH
    assert trace[0].graph_node.base == RegularBase(C)
    assert trace[1].operation == DpOperation.REPLACE
    assert trace[1].graph_node.base == RegularBase(G)
    assert trace[2].operation == DpOperation.MATCH
    assert trace[2].graph_node.base == RegularBase(A)


def test_dp_end():
    start_graph_node = initial_graph_of(make_regular(A)).start_nodes[0]
    sequence = []
    cost, trace = dp_memoized_function(start_graph_node, sequence)
    assert cost == 0
    assert len(trace) == 1
    assert trace[0].operation == DpOperation.END
    assert trace[0].graph_node == start_graph_node


def test_dp_one_insertion_scenario():
    start_graph_node = initial_graph_of(make_regular(G, T, A, C)).start_nodes[0]
    sequence = make_regular(G, A, C)
    cost, trace = dp_memoized_function(start_graph_node, sequence)
    assert cost == INSERTION_PENALTY
    assert len(trace) == 4
    assert trace[0].operation == DpOperation.MATCH
    assert trace[0].graph_node.base == RegularBase(G)
    assert trace[1].operation == DpOperation.INSERT
    assert trace[1].graph_node.base == RegularBase(T)
    assert trace[2].operation == DpOperation.MATCH
    assert trace[2].graph_node.base == RegularBase(A)
    assert trace[3].operation == DpOperation.MATCH
    assert trace[3].graph_node.base == RegularBase(C)


def test_dp_one_deletion_scenario():
    start_graph_node = initial_graph_of(make_regular(G, A, C)).start_nodes[0]
    sequence = make_regular(G, T, A, C)
    cost, trace = dp_memoized_function(start_graph_node, sequence)
    assert cost == DELETION_PENALTY
    assert len(trace) == 4
    assert trace[0].operation == DpOperation.MATCH
    assert trace[0].graph_node.base == RegularBase(G)
    assert trace[1].operation == DpOperation.DELETE
    assert trace[1].graph_node.base == RegularBase(A)
    assert trace[2].operation == DpOperation.MATCH
    assert trace[2].graph_node.base == RegularBase(A)
    assert trace[3].operation == DpOperation.MATCH
    assert trace[3].graph_node.base == RegularBase(C)


def test_dp_mixed_scenario():
    start_graph_node = initial_graph_of(make_regular(G, A, C, T, T, G, C)).start_nodes[0]
    sequence = make_regular(G, A, T, T, G, A, C)
    cost, trace = dp_memoized_function(start_graph_node, sequence)
    assert cost == INSERTION_PENALTY + DELETION_PENALTY
    assert len(trace) == 8
    assert trace[0].operation == DpOperation.MATCH
    assert trace[0].graph_node.base == RegularBase(G)
    assert trace[1].operation == DpOperation.MATCH
    assert trace[1].graph_node.base == RegularBase(A)
    assert trace[2].operation == DpOperation.INSERT
    assert trace[2].graph_node.base == RegularBase(C)
    assert trace[3].operation == DpOperation.MATCH
    assert trace[3].graph_node.base == RegularBase(T)
    assert trace[4].operation == DpOperation.MATCH
    assert trace[4].graph_node.base == RegularBase(T)
    assert trace[5].operation == DpOperation.MATCH
    assert trace[5].graph_node.base == RegularBase(G)
    assert trace[6].operation == DpOperation.DELETE
    assert trace[6].graph_node.base == RegularBase(C)
    assert trace[7].operation == DpOperation.MATCH
    assert trace[7].graph_node.base == RegularBase(C)
    

def test_dp_mixed_scenario2():
    start_graph_node = initial_graph_of(make_regular(G, A, C, T, T, G, C, T, A, A)).start_nodes[0]
    sequence = make_regular(G, A, G, T, T, G, T, A, A)
    cost, trace = dp_memoized_function(start_graph_node, sequence)
    assert cost == RegularBase(C).dp_conversion_penalty(RegularBase(G)) + INSERTION_PENALTY
    assert len(trace) == 10
    assert trace[0].operation == DpOperation.MATCH
    assert trace[0].graph_node.base == RegularBase(G)
    assert trace[1].operation == DpOperation.MATCH
    assert trace[1].graph_node.base == RegularBase(A)
    assert trace[2].operation == DpOperation.REPLACE
    assert trace[2].graph_node.base == RegularBase(C)
    assert trace[3].operation == DpOperation.MATCH
    assert trace[3].graph_node.base == RegularBase(T)
    assert trace[4].operation == DpOperation.MATCH
    assert trace[4].graph_node.base == RegularBase(T)
    assert trace[5].operation == DpOperation.MATCH
    assert trace[5].graph_node.base == RegularBase(G)
    assert trace[6].operation == DpOperation.INSERT
    assert trace[6].graph_node.base == RegularBase(C)
    assert trace[7].operation == DpOperation.MATCH
    assert trace[7].graph_node.base == RegularBase(T)
    assert trace[8].operation == DpOperation.MATCH
    assert trace[8].graph_node.base == RegularBase(A)
    assert trace[9].operation == DpOperation.MATCH
    assert trace[9].graph_node.base == RegularBase(A)


def test_add_to_graph_single_match():
    start_graph_node = initial_graph_of(make_regular(A)).start_nodes[0]
    sequence = make_regular(A)
    cost, trace = dp_memoized_function(start_graph_node, sequence)
    add_trace_to_graph(sequence, trace)
    len(start_graph_node.read_nodes) == 2

'''
def test_add_to_graph_single_replace():
    start_graph_node = initial_graph_of(make_regular(A, T, G)).start_nodes[0]
    sequence = make_regular(G, T, G)
    cost, trace = dp_memoized_function(start_graph_node, sequence)
    add_trace_to_graph(sequence, trace)
    len(start_graph_node.read_nodes) == 1
    len(start_graph_node.layer.graph_nodes) == 2
'''

# todo: account for flexible start (not necessarily at the start of the graph)
