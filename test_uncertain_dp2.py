from uncertain_dp2 import *

TEST_ALINGMENT_BONUS = 0.7


def test_create_read_node_chain():
    read_node_chain = create_read_node_chain(make_regular(C, G, T))
    assert read_node_chain.base == RegularBase(C)
    assert read_node_chain.next.base == RegularBase(G)
    assert read_node_chain.next.next.base == RegularBase(T)
    assert read_node_chain.next.next.next == None


def test_graph_initialization():
    graph = initial_graph_of(make_regular(C, G, T))
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
    start_graph_node = initial_graph_of(make_regular(C, T)).start_nodes[0]
    start_graph_node.layer.graph_nodes == [start_graph_node]
    start_graph_node.successors[0].layer.graph_nodes == [start_graph_node.successors[0]]


def test_dp_same_one_base():
    start_graph_node = initial_graph_of(make_regular(A)).start_nodes[0]
    sequence = make_regular(A)
    option = wiped_dp_memoized_function(
        start_graph_node, sequence, True, True, alignment_bonus=TEST_ALINGMENT_BONUS
    )
    cost, trace = option.cost, option.trace
    assert cost == 0
    assert len(trace) == 1
    assert trace[0].operation == DpOperation.MATCH
    assert trace[0].graph_node == start_graph_node


def test_dp_same_two_bases():
    start_graph_node = initial_graph_of(make_regular(C, G)).start_nodes[0]
    sequence = make_regular(C, G)
    option = wiped_dp_memoized_function(
        start_graph_node, sequence, True, True, alignment_bonus=TEST_ALINGMENT_BONUS
    )
    cost, trace = option.cost, option.trace
    assert cost == 0
    assert len(trace) == 2
    assert trace[0].operation == DpOperation.MATCH
    assert trace[0].graph_node.base == RegularBase(C)
    assert trace[1].operation == DpOperation.MATCH
    assert trace[1].graph_node.base == RegularBase(G)


def test_dp_same_multiple_bases():
    start_graph_node = initial_graph_of(make_regular(C, G, A, T)).start_nodes[0]
    sequence = make_regular(C, G, A, T)
    option = wiped_dp_memoized_function(
        start_graph_node, sequence, True, True, alignment_bonus=TEST_ALINGMENT_BONUS
    )
    cost, trace = option.cost, option.trace
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
    option = wiped_dp_memoized_function(
        start_graph_node, sequence, True, True, alignment_bonus=TEST_ALINGMENT_BONUS
    )
    cost, trace = option.cost, option.trace
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
    option = wiped_dp_memoized_function(
        start_graph_node, sequence, True, True, alignment_bonus=TEST_ALINGMENT_BONUS
    )
    cost, trace = option.cost, option.trace
    assert cost == 0
    assert len(trace) == 1
    assert trace[0].operation == DpOperation.INSERT_ALL_END
    assert trace[0].graph_node == start_graph_node


def test_dp_one_insertion_scenario():
    start_graph_node = initial_graph_of(make_regular(G, T, A, C)).start_nodes[0]
    sequence = make_regular(G, A, C)
    option = wiped_dp_memoized_function(
        start_graph_node, sequence, True, True, alignment_bonus=TEST_ALINGMENT_BONUS
    )
    cost, trace = option.cost, option.trace
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
    option = wiped_dp_memoized_function(
        start_graph_node, sequence, True, True, alignment_bonus=TEST_ALINGMENT_BONUS
    )
    cost, trace = option.cost, option.trace
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
    start_graph_node = initial_graph_of(
        make_regular(G, A, C, T, T, G, C, C, C)
    ).start_nodes[0]
    sequence = make_regular(G, A, T, T, G, A, C, C, C)
    option = wiped_dp_memoized_function(
        start_graph_node, sequence, True, True, alignment_bonus=TEST_ALINGMENT_BONUS
    )
    # assert [t.operation.name for t in trace] == False
    cost, trace = option.cost, option.trace
    assert cost == INSERTION_PENALTY + DELETION_PENALTY
    assert len(trace) == 10
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
    assert trace[8].operation == DpOperation.MATCH
    assert trace[8].graph_node.base == RegularBase(C)
    assert trace[9].operation == DpOperation.MATCH
    assert trace[9].graph_node.base == RegularBase(C)


def test_dp_mixed_scenario2():
    start_graph_node = initial_graph_of(
        make_regular(G, A, C, T, T, G, C, T, A, A)
    ).start_nodes[0]
    sequence = make_regular(G, A, G, T, T, G, T, A, A)
    option = wiped_dp_memoized_function(
        start_graph_node, sequence, True, True, alignment_bonus=TEST_ALINGMENT_BONUS
    )
    cost, trace = option.cost, option.trace
    assert (
        cost == RegularBase(C).dp_conversion_penalty(RegularBase(G)) + INSERTION_PENALTY
    )
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
    option = wiped_dp_memoized_function(
        start_graph_node, sequence, True, True, alignment_bonus=TEST_ALINGMENT_BONUS
    )
    cost, trace = option.cost, option.trace
    add_trace_to_graph(sequence, trace)
    assert len(start_graph_node.read_nodes) == 2


def test_add_to_graph_multiple_match():
    start_graph_node = initial_graph_of(make_regular(A, T, G)).start_nodes[0]
    sequence = make_regular(A, T, G)
    option = wiped_dp_memoized_function(
        start_graph_node, sequence, True, True, alignment_bonus=TEST_ALINGMENT_BONUS
    )
    cost, trace = option.cost, option.trace
    add_trace_to_graph(sequence, trace)
    assert len(start_graph_node.read_nodes) == 2
    assert len(start_graph_node.layer.graph_nodes) == 1
    assert start_graph_node.layer.graph_nodes[0].base == RegularBase(A)
    assert len(start_graph_node.successors[0].layer.graph_nodes) == 1
    assert len(start_graph_node.successors[0].successors[0].layer.graph_nodes) == 1


def test_add_to_graph_replace():
    start_graph_node = initial_graph_of(make_regular(C, C, A, T, G)).start_nodes[0]
    sequence = make_regular(C, C, G, T, G)
    option = wiped_dp_memoized_function(
        start_graph_node, sequence, True, True, alignment_bonus=TEST_ALINGMENT_BONUS
    )
    cost, trace = option.cost, option.trace
    add_trace_to_graph(sequence, trace)

    node = start_graph_node.successors[0].successors[0]
    assert len(node.read_nodes) == 1
    assert len(node.layer.graph_nodes) == 2
    bases_here = list(map(lambda x: x.base, node.layer.graph_nodes))
    assert RegularBase(G) in bases_here
    assert RegularBase(A) in bases_here


def test_add_to_graph_replace2():
    start_graph_node = initial_graph_of(make_regular(A, T, G)).start_nodes[0]
    sequence = make_regular(A, C, G)
    option = wiped_dp_memoized_function(
        start_graph_node, sequence, True, True, alignment_bonus=TEST_ALINGMENT_BONUS
    )
    cost, trace = option.cost, option.trace
    add_trace_to_graph(sequence, trace)
    assert len(start_graph_node.successors) == 2


def test_add_to_graph_delete():
    graph = initial_graph_of(make_regular(A, T, G))
    sequence = make_regular(A, C, T, G)
    add_to_graph(sequence, graph, alignment_bonus=TEST_ALINGMENT_BONUS)
    start_graph_node = graph.start_nodes[0]
    assert len(start_graph_node.layer.graph_nodes) == 1
    assert len(start_graph_node.successors) == 2
    c_node = [s for s in start_graph_node.successors if s.base == RegularBase(C)][0]
    t_node = [s for s in start_graph_node.successors if s.base == RegularBase(T)][0]
    assert c_node.layer != t_node.layer
    assert c_node.successors[0].layer == t_node.layer


def test_add_to_graph_insert():
    graph = initial_graph_of(make_regular(G, T, A, C))
    sequence = make_regular(G, A, C)
    add_to_graph(sequence, graph, alignment_bonus=TEST_ALINGMENT_BONUS)
    start_graph_node = graph.start_nodes[0]
    assert len(start_graph_node.layer.graph_nodes) == 1
    assert len(start_graph_node.successors) == 2
    a_node = [s for s in start_graph_node.successors if s.base == RegularBase(A)][0]
    t_node = [s for s in start_graph_node.successors if s.base == RegularBase(T)][0]
    assert a_node.layer != t_node.layer
    assert t_node.successors[0].layer == a_node.layer


def test_add_to_graph_end():
    graph = initial_graph_of(make_regular(A, T))
    sequence = make_regular(A)
    add_to_graph(sequence, graph, alignment_bonus=TEST_ALINGMENT_BONUS)
    start_graph_node = graph.start_nodes[0]
    assert len(start_graph_node.layer.graph_nodes) == 1
    assert len(start_graph_node.successors) == 1
    assert start_graph_node.successors[0].base == RegularBase(T)


def test_multiple_sequences_alignment():
    start_graph_node = multiple_sequence_alignment(
        [make_regular(A, T), make_regular(A)], alignment_bonus=TEST_ALINGMENT_BONUS
    ).start_nodes[0]
    assert len(start_graph_node.layer.graph_nodes) == 1
    assert len(start_graph_node.successors) == 1
    assert start_graph_node.successors[0].base == RegularBase(T)


def test_add_three_same_sequences():
    graph = initial_graph_of(make_regular(A, T, C, G))
    sequence = make_regular(A, T, C, G)
    add_to_graph(sequence, graph, alignment_bonus=TEST_ALINGMENT_BONUS)
    add_to_graph(sequence, graph, alignment_bonus=TEST_ALINGMENT_BONUS)
    start_graph_node = graph.start_nodes[0]
    assert len(start_graph_node.layer.graph_nodes) == 1
    assert len(start_graph_node.successors) == 1
    assert start_graph_node.successors[0].base == RegularBase(T)
    assert len(start_graph_node.successors[0].layer.graph_nodes) == 1
    assert len(start_graph_node.successors[0].successors) == 1
    assert start_graph_node.successors[0].successors[0].base == RegularBase(C)
    assert len(start_graph_node.read_nodes) == 3


def test_add_three_different_sequences():
    graph = initial_graph_of(make_regular(A, T, C, G))
    sequence = make_regular(A, C, C, G)
    add_to_graph(sequence, graph, alignment_bonus=TEST_ALINGMENT_BONUS)
    sequence = make_regular(A, G, C, G)
    add_to_graph(sequence, graph, alignment_bonus=TEST_ALINGMENT_BONUS)
    start_graph_node = graph.start_nodes[0]
    assert len(start_graph_node.successors) == 3
    assert len({s.layer for s in start_graph_node.successors}) == 1


def test_arbitrary_alignment_to_graph():
    graph = initial_graph_of(make_regular(A, T, G))
    sequence = make_regular(A, C, G)
    add_to_graph(sequence, graph, alignment_bonus=TEST_ALINGMENT_BONUS)
    sequence = make_regular(A, C, G)
    start_graph_node = graph.start_nodes[0]
    option = wiped_dp_memoized_function(
        start_graph_node, sequence, True, True, alignment_bonus=TEST_ALINGMENT_BONUS
    )
    cost, trace = option.cost, option.trace
    assert cost == 0


def test_arbitrary_alignment_to_graph2():
    graph = initial_graph_of(make_regular(C, A, T, G, C, G))
    sequence = make_regular(C, A, G, C, G)
    add_to_graph(sequence, graph, alignment_bonus=TEST_ALINGMENT_BONUS)
    start_graph_node = graph.start_nodes[0]

    sequence = make_regular(C, A, C, G)
    option = wiped_dp_memoized_function(
        start_graph_node, sequence, True, True, alignment_bonus=TEST_ALINGMENT_BONUS
    )
    cost, trace = option.cost, option.trace
    assert cost == 1

    sequence = make_regular(C, A, T, G, G)
    option = wiped_dp_memoized_function(
        start_graph_node, sequence, True, True, alignment_bonus=TEST_ALINGMENT_BONUS
    )
    cost, trace = option.cost, option.trace
    assert cost == 1


def test_arbitrary_alignment_to_graph_not_branch_jumping():
    graph = initial_graph_of(make_regular(C, A, T, G, C, G))
    sequence = make_regular(C, A, A, C, C, G)
    add_to_graph(sequence, graph, alignment_bonus=TEST_ALINGMENT_BONUS)
    start_graph_node = graph.start_nodes[0]

    sequence = make_regular(C, A, T, C, C, G)
    option = wiped_dp_memoized_function(
        start_graph_node, sequence, True, True, alignment_bonus=TEST_ALINGMENT_BONUS
    )
    cost, trace = option.cost, option.trace
    assert cost == 1  # would be 0 if it could jump branches


def test_arbitrary_alignment_to_graph_take_multiple_branches():
    graph = initial_graph_of(make_regular(A, T, C, G, A, T, C, G, A, T, C, G))
    sequence = make_regular(A, T, C, T, A, T, C, G, A, T, C, G)
    add_to_graph(sequence, graph, alignment_bonus=TEST_ALINGMENT_BONUS)
    sequence = make_regular(A, T, C, G, A, T, C, C, A, T, C, G)
    add_to_graph(sequence, graph, alignment_bonus=TEST_ALINGMENT_BONUS)

    sequence = make_regular(A, T, C, T, A, T, C, C, A, T, C, G)
    option = dp_with_graph(sequence, graph, alignment_bonus=TEST_ALINGMENT_BONUS)
    cost, trace = option.cost, option.trace
    assert cost == 0


def test_uncertain_base_scalar_product():
    uncertain_base_scalar_product(just(A), just(A)) == 0
    uncertain_base_scalar_product(just(A), just(C)) == 1
    uncertain_base_scalar_product([0.2, 0.5, 0.2, 0.1], [0, 0.2, 0.4, 0.4]) == 0.22
    uncertain_base_scalar_product([0.5, 0.5, 0, 0], [0.5, 0.5, 0, 0]) == 0.5


def test_align_positional_weight_matrix_reads_single_identical():
    start_graph_node = initial_graph_of(make_uncertain_regular(just(A))).start_nodes[0]
    sequence = make_uncertain_regular(just(A))
    option = wiped_dp_memoized_function(
        start_graph_node, sequence, True, True, alignment_bonus=TEST_ALINGMENT_BONUS
    )
    cost, trace = option.cost, option.trace
    assert cost == 0
    assert len(trace) == 1
    assert trace[0].operation == DpOperation.MATCH


def test_align_positional_weight_matrix_reads_single_non_identical():
    start_graph_node = initial_graph_of(
        make_uncertain_regular([0, 0, 0, 1])
    ).start_nodes[0]
    sequence = make_uncertain_regular([0, 0.1, 0.1, 0.8])
    option = wiped_dp_memoized_function(
        start_graph_node, sequence, True, True, alignment_bonus=TEST_ALINGMENT_BONUS
    )
    cost, trace = option.cost, option.trace
    assert round(cost, 8) == 0.2
    assert len(trace) == 1
    assert trace[0].operation == DpOperation.REPLACE


def test_align_positional_weight_matrix_reads_identical_sequence():
    start_graph_node = initial_graph_of(
        make_uncertain_regular(*just_those(A, T, C))
    ).start_nodes[0]
    sequence = make_uncertain_regular(*just_those(A, T, C))
    option = wiped_dp_memoized_function(
        start_graph_node, sequence, True, True, alignment_bonus=TEST_ALINGMENT_BONUS
    )
    cost, trace = option.cost, option.trace
    assert cost == 0


def test_align_positional_weight_matrix_reads_non_identical_sequence1():
    start_graph_node = initial_graph_of(
        make_uncertain_regular(*just_those(A, T, C, T, T))
    ).start_nodes[0]
    sequence = make_uncertain_regular(*just_those(A, T, G, T, T))
    option = wiped_dp_memoized_function(
        start_graph_node, sequence, True, True, alignment_bonus=TEST_ALINGMENT_BONUS
    )
    cost, trace = option.cost, option.trace
    assert cost == 1


def test_align_positional_weight_matrix_reads_non_identical_sequence2():
    start_graph_node = initial_graph_of(
        make_uncertain_regular(*probably_those(A, T, p=0.7))
    ).start_nodes[0]
    sequence = make_uncertain_regular(*probably_those(A, T, p=0.7))
    option = wiped_dp_memoized_function(
        start_graph_node, sequence, True, True, alignment_bonus=TEST_ALINGMENT_BONUS
    )
    cost, trace = option.cost, option.trace
    assert len(trace) == 2
    assert trace[0].operation == DpOperation.MATCH
    assert trace[1].operation == DpOperation.MATCH
    assert cost == 2 * PositionalWeightMatrixBase(probably(A)).dp_conversion_penalty(
        PositionalWeightMatrixBase(probably(A))
    )


def test_align_positional_weight_matrix_reads_non_identical_sequence3():
    start_graph_node = initial_graph_of(
        make_uncertain_regular(*just_those(A, T, C, T, T))
    ).start_nodes[0]
    sequence = make_uncertain_regular(*just_those(A, T, G, T, T))
    option = wiped_dp_memoized_function(
        start_graph_node, sequence, True, True, alignment_bonus=TEST_ALINGMENT_BONUS
    )
    cost, trace = option.cost, option.trace
    assert cost == 1


def test_align_positional_weight_matrix_reads_non_identical_sequence4():
    start_graph_node = initial_graph_of(
        make_uncertain_regular(mix(A, C, C, C), mix(A, T), just(A))
    ).start_nodes[0]
    sequence = make_uncertain_regular(mix(A, C, C, C), mix(A, G), just(A))
    option = wiped_dp_memoized_function(
        start_graph_node, sequence, True, True, alignment_bonus=TEST_ALINGMENT_BONUS
    )
    cost, trace = option.cost, option.trace

    # assert [t.operation.name for t in trace] == False
    assert cost == 0.5 + 0.75 ** 2 + 0.25 ** 2
    assert len(trace) == 3
    assert (
        trace[0].operation == DpOperation.MATCH
    )  # same uncertain event. Does not mean there is no penalty
    assert trace[1].operation == DpOperation.REPLACE
    assert trace[2].operation == DpOperation.MATCH


def test_align_positional_weight_matrix_reads_detached():
    start_graph_node = initial_graph_of(
        make_uncertain_regular(just(A), mix(A, T, T, T, T), mix(T, G, G, G, G, G))
    ).start_nodes[0]
    sequence = make_uncertain_regular(
        mix(A, A, A, A, G), mix(A, T, T, T), mix(G, G, G, T)
    )
    option = wiped_dp_memoized_function(
        start_graph_node, sequence, True, True, alignment_bonus=TEST_ALINGMENT_BONUS
    )
    cost, trace = option.cost, option.trace

    # assert [t.operation.name for t in trace] == False
    assert len(trace) == 3
    assert (
        trace[0].operation == DpOperation.REPLACE
    )  # same uncertain event. Does not mean there is no penalty
    assert trace[1].operation == DpOperation.REPLACE
    assert trace[2].operation == DpOperation.REPLACE


def test_positional_weigth_matrix_add_to_graph():
    graph = initial_graph_of(
        make_uncertain_regular(mix(A, C, C, C), mix(A, A, A, T), just(A))
    )
    sequence = make_uncertain_regular(mix(A, C, C, C), mix(A, A, A, G), just(A))
    add_to_graph(sequence, graph, alignment_bonus=TEST_ALINGMENT_BONUS)
    start_graph_node = graph.start_nodes[0]

    assert start_graph_node.base == PositionalWeightMatrixBase(mix(A, C, C, C))
    assert len(start_graph_node.read_nodes) == 2
    assert len(start_graph_node.layer.graph_nodes) == 1
    assert len(start_graph_node.successors) == 2
    assert len(start_graph_node.successors[0].read_nodes) == 1
    assert len(start_graph_node.successors[1].read_nodes) == 1
    level_2_successors = set()
    for s in start_graph_node.successors:
        level_2_successors.update(s.successors)
    assert len(level_2_successors) == 1


def test_align_positional_weight_matrix_reads_to_either_existing_path():
    p1 = make_uncertain_regular(just(C), just(A), just(T), mix(A, T))
    p2 = make_uncertain_regular(
        just(C), mix(A, A, A, A, A, G), just(T), mix(A, G, G, G, G, G)
    )
    graph = initial_graph_of(p1)
    add_to_graph(p2, graph, alignment_bonus=TEST_ALINGMENT_BONUS)
    option = dp_with_graph(p1, graph, alignment_bonus=TEST_ALINGMENT_BONUS)
    cost, trace = option.cost, option.trace
    assert cost == 0 + 0 + 0 + 0.5

    option = dp_with_graph(p2, graph, alignment_bonus=TEST_ALINGMENT_BONUS)
    # assert [t.operation.name for t in trace] == False
    # assert round(cost,6) == round(4/9,6)


def test_consent_regular_base():
    RegularBase.consent([RegularBase(A), RegularBase(A)]) == RegularBase(A)
    RegularBase.consent([RegularBase(A), RegularBase(C)]) in [
        RegularBase(A),
        RegularBase(C),
    ]
    RegularBase.consent(
        [RegularBase(A), RegularBase(C), RegularBase(C)]
    ) == RegularBase(C)


def test_consent_positional_base_matrix():
    PositionalWeightMatrixBase.consent(
        [PositionalWeightMatrixBase(just(A)), PositionalWeightMatrixBase(just(A))]
    ) == PositionalWeightMatrixBase(just(A))

    PositionalWeightMatrixBase.consent(
        [PositionalWeightMatrixBase(just(A)), PositionalWeightMatrixBase(just(C))]
    ) == PositionalWeightMatrixBase(mix(A, C))

    PositionalWeightMatrixBase.consent(
        [
            PositionalWeightMatrixBase(just(A)),
            PositionalWeightMatrixBase(just(C)),
            PositionalWeightMatrixBase(just(C)),
        ]
    ) == PositionalWeightMatrixBase(mix(A, C, C))

    PositionalWeightMatrixBase.consent(
        [
            PositionalWeightMatrixBase([0.5, 0.5, 0, 0]),
            PositionalWeightMatrixBase([0.6, 0.4, 0, 0]),
        ]
    ) == PositionalWeightMatrixBase([0.55, 0.45, 0, 0])


def test_consent_of_graph():
    graph = multiple_sequence_alignment(
        [
            make_regular(A, G, C),
            make_regular(A, G, C),
            make_regular(A, T, C),
        ],
        alignment_bonus=TEST_ALINGMENT_BONUS,
    )
    assert consent_of_graph(graph) == make_regular(A, G, C)


def test_consent_positional_weight_matrix():
    assert consent_of_graph(
        multiple_sequence_alignment(
            [
                make_uncertain_regular(
                    just(A), just(A), just(A), just(G), just(A), just(A), just(A)
                ),
                make_uncertain_regular(
                    just(A), just(A), just(A), just(T), just(A), just(A), just(A)
                ),
            ],
            alignment_bonus=TEST_ALINGMENT_BONUS,
        )
    ) == make_uncertain_regular(
        just(A), just(A), just(A), mix(T, G), just(A), just(A), just(A)
    )


def test_consent_positional_weight_matrix_delete():
    assert consent_of_graph(
        multiple_sequence_alignment(
            [
                make_uncertain_regular(
                    just(A),
                    just(A),
                    just(A),
                    just(T),
                    just(A),
                    just(C),
                    just(A),
                    just(A),
                ),
                make_uncertain_regular(
                    just(A),
                    just(A),
                    just(A),
                    just(T),
                    just(A),
                    just(C),
                    just(A),
                    just(A),
                ),
                make_uncertain_regular(
                    just(A), just(A), just(A), just(A), just(C), just(A), just(A)
                ),
            ],
            alignment_bonus=TEST_ALINGMENT_BONUS,
        )
    ) == make_uncertain_regular(
        just(A),
        just(A),
        just(A),
        just(T),
        just(A),
        just(C),
        just(A),
        just(A),
    )


def test_consent_positional_weight_matrix_insert():
    assert (
        consent_of_graph(
            multiple_sequence_alignment(
                [
                    make_uncertain_regular(*just_those(A, T, A, C)),
                    make_uncertain_regular(*just_those(A, T, A, C)),
                    make_uncertain_regular(*just_those(A, T, G, A, C)),
                ],
                alignment_bonus=TEST_ALINGMENT_BONUS,
            )
        )
        == make_uncertain_regular(*just_those(A, T, A, C))
    )


def test_consent_positional_weight_matrix_multiple_insert():
    assert (
        consent_of_graph(
            multiple_sequence_alignment(
                [
                    make_uncertain_regular(*just_those(A, C, T, A, C)),
                    make_uncertain_regular(*just_those(A, T, A, C)),
                    make_uncertain_regular(*just_those(A, T, A, C)),
                    make_uncertain_regular(*just_those(A, T, A, C)),
                    make_uncertain_regular(*just_those(A, T, G, T, A, C)),
                ],
                alignment_bonus=TEST_ALINGMENT_BONUS,
            )
        )
        == make_uncertain_regular(*just_those(A, T, A, C))
    )


def test_consent_positional_weight_matrix_complex():
    consent_of_graph(
        multiple_sequence_alignment(
            [
                make_uncertain_regular(*just_those(A, T, A, G, C, C)),
                make_uncertain_regular(*just_those(A, C, A, C, C)),
                make_uncertain_regular(*just_those(A, T, A, C, C)),
                make_uncertain_regular(*just_those(A, C, G, C, C)),
                make_uncertain_regular(*just_those(A, C, A, C, C)),
            ],
            alignment_bonus=TEST_ALINGMENT_BONUS,
        )
    ) == make_uncertain_regular(
        just(A), mix(T, T, C, C, C), mix(G, A, A, A, A), just(C), just(C)
    )


def test_rigth_consent_from_overritten():
    PositionalWeightMatrixBase.consent(
        make_uncertain_regular(mix(A, C), just(A))
    ) == PositionalWeightMatrixBase(mix(A, A, C))


def test_consent_insert_all_end():
    consent_of_graph(
        multiple_sequence_alignment(
            [
                make_uncertain_regular(*just_those(A, C, A, G)),
                make_uncertain_regular(*just_those(A, C, A, G, C)),
            ],
            alignment_bonus=TEST_ALINGMENT_BONUS,
        )
    ) == make_uncertain_regular(*just_those(A, C, A, G, C))


def test_consent_delete_all_end():
    consent_of_graph(
        multiple_sequence_alignment(
            [
                make_uncertain_regular(*just_those(A, C, A, G, C)),
                make_uncertain_regular(*just_those(A, C, A, G)),
            ],
            alignment_bonus=TEST_ALINGMENT_BONUS,
        )
    ) == make_uncertain_regular(*just_those(A, C, A, G, C))


def test_consent_of_graph1():
    g = multiple_sequence_alignment(
        [
            make_regular(C, T, G, G, A, C),
            make_regular(G, G, A, C, C, T),
        ],
        alignment_bonus=TEST_ALINGMENT_BONUS,
    )
    print(g)
    assert consent_of_graph(g) == make_regular(C, T, G, G, A, C, C, T)


def test_read_consent():
    dna = "AACTGGACCTACGGAT"
    read_length = 6
    reads = [
        generate_read(dna, read_length, start=2),
        generate_read(dna, read_length, start=4),
    ]
    consent = read_consent(reads, alignment_bonus=TEST_ALINGMENT_BONUS)
    assert consent == [C, T, G, G, A, C, C, T]


def test_read_consent_return_read():
    dna = "AACTGGACCTACGGAT"
    read_length = 6
    reads = [
        generate_read(dna, read_length, start=2),
        generate_read(dna, read_length, start=4),
    ]
    consent: Read = read_consent(
        reads, as_read=True, alignment_bonus=TEST_ALINGMENT_BONUS
    )
    # assert consent. == [C, T, G, G, A, C, C, T]


def test_edit_distance():
    assert (
        edit_distance(
            make_regular(A, A, A, C, A, A, A),
            make_regular(A, A, A, T, A, A, A),
            alignment_bonus=TEST_ALINGMENT_BONUS,
        )
        == RegularBase.dp_conversion_penalty(RegularBase(C), RegularBase(T))
    )


"""
def test_alignment1():
    s1 = make_uncertain_regular(*just_those(    G,C,C,G,C,C,C,A,G,A))
    s2 = make_uncertain_regular(*just_those(G,T,G,T,C,G,G,C,C,A))

    option = wiped_dp_memoized_function(initial_graph_of(s1).start_nodes[0], s2, True, True)
    #assert [t.operation.name for t in trace] == False
"""


def test_dp_same_one_base_graph():
    graph = initial_graph_of(make_regular(A))
    option = dp_with_graph(make_regular(A), graph, alignment_bonus=TEST_ALINGMENT_BONUS)
    cost, trace = option.cost, option.trace
    assert cost == 0
    assert len(trace) == 1
    assert trace[0].operation == DpOperation.MATCH
    assert trace[0].graph_node == graph.start_nodes[0]


def test_dp_same_two_bases_graph():
    graph = initial_graph_of(make_regular(C, G))
    option = dp_with_graph(
        make_regular(C, G), graph, alignment_bonus=TEST_ALINGMENT_BONUS
    )
    cost, trace = option.cost, option.trace
    assert cost == 0
    assert len(trace) == 2
    assert trace[0].operation == DpOperation.MATCH
    assert trace[0].graph_node.base == RegularBase(C)
    assert trace[1].operation == DpOperation.MATCH
    assert trace[1].graph_node.base == RegularBase(G)


def test_dp_same_multiple_bases_graph():
    graph = initial_graph_of(make_regular(C, G, A, T))
    option = dp_with_graph(
        make_regular(C, G, A, T), graph, alignment_bonus=TEST_ALINGMENT_BONUS
    )
    cost, trace = option.cost, option.trace
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


def test_new_start_added_correctly():
    graph = initial_graph_of(make_regular(A, T, C, G))
    sequence = make_regular(C, A, T, C, G)
    add_to_graph(sequence, graph, alignment_bonus=TEST_ALINGMENT_BONUS)
    assert len(graph.start_nodes) == 2
    assert graph.start_nodes[1].base == RegularBase(C)
    assert graph.start_nodes[1].successors[0] == graph.start_nodes[0]


def test_new_start_added_correctly2():
    graph = initial_graph_of(make_regular(A, T, C, G, T, T, C))
    sequence = make_regular(C, A, T, C, A, T, T, C)
    add_to_graph(sequence, graph, alignment_bonus=TEST_ALINGMENT_BONUS)
    assert len(graph.start_nodes) == 2


def test_all_starts_considered():
    graph = initial_graph_of(make_regular(A, T, C, G, T, T, C))
    sequence = make_regular(T, A, T, T, T, T)
    add_to_graph(sequence, graph, alignment_bonus=TEST_ALINGMENT_BONUS)
    option = dp_with_graph(
        make_regular(T, A, T, T, T, T), graph, alignment_bonus=TEST_ALINGMENT_BONUS
    )
    cost, trace = option.cost, option.trace
    print(graph.start_nodes)
    print([t.operation.name for t in trace])
    assert cost == 0
    assert sum([t.operation == DpOperation.MATCH for t in trace]) == 6


def test_consent_at_read():
    reads = [
        Read("ATCGTTC", 0, uncertainty_generator=certain_uncertainty_generator),
        Read("ATCATTC", 0, uncertainty_generator=certain_uncertainty_generator),
        Read("ATCATTC", 0, uncertainty_generator=certain_uncertainty_generator),
    ]
    graph = read_multiple_sequence_alignment(
        reads,
        alignment_bonus=TEST_ALINGMENT_BONUS,
        probabilistic=False,
    )

    assert consent_at_read(reads[0]) == make_regular(A, T, C, A, T, T, C)


def test_consent_at_read_probabilistic():
    reads = [
        Read("ATCGTTC", 0, uncertain_text=just_those(A, T, C, G, T, T, C)),
        Read("ATCATTC", 0, uncertain_text=just_those(A, T, C, A, T, T, C)),
        Read("ATCATTC", 0, uncertain_text=just_those(A, T, C, A, T, T, C)),
    ]
    graph = read_multiple_sequence_alignment(
        reads,
        alignment_bonus=TEST_ALINGMENT_BONUS,
        probabilistic=True,
    )
    print(consent_at_read(reads[0]))

    assert consent_at_read(reads[0]) == make_uncertain_regular(
        *[*just_those(A, T, C), mix(G, A, A), *just_those(T, T, C)]
    )


def test_consent_at_read_insert():
    reads = [
        Read("ATCGTTC", 0, uncertainty_generator=certain_uncertainty_generator),
        Read("ATCTTC", 0, uncertainty_generator=certain_uncertainty_generator),
        Read("ATCTTC", 0, uncertainty_generator=certain_uncertainty_generator),
    ]
    graph = read_multiple_sequence_alignment(
        reads,
        alignment_bonus=TEST_ALINGMENT_BONUS,
        probabilistic=False,
    )
    assert consent_at_read(reads[0]) == make_regular(A, T, C, T, T, C)



def test_consent_at_read_delete():
    
    reads = [
        Read("ATCTTC", 0, uncertainty_generator=certain_uncertainty_generator),
        Read("ATCCTTC", 0, uncertainty_generator=certain_uncertainty_generator),
        Read("ATCCTTC", 0, uncertainty_generator=certain_uncertainty_generator),
    ]
    graph = read_multiple_sequence_alignment(
        reads,
        alignment_bonus=TEST_ALINGMENT_BONUS,
        probabilistic=False,
    )
    assert consent_at_read(reads[0]) == make_regular(A, T, C, C, T, T, C)


def test_consent_at_read_delete_end():
    """Semantically: Read only needs to be corrected within its scope.
      So inserts in the front and deletes in the back will be disregarded"""
    reads = [
        Read("ATCCTT", 0, uncertainty_generator=certain_uncertainty_generator),
        Read("ATCCTTC", 0, uncertainty_generator=certain_uncertainty_generator),
        Read("ATCCTTC", 0, uncertainty_generator=certain_uncertainty_generator),
    ]
    graph = read_multiple_sequence_alignment(
        reads,
        alignment_bonus=TEST_ALINGMENT_BONUS,
        probabilistic=False,
    )
    assert consent_at_read(reads[0]) == make_regular(A, T, C, C, T, T)



def test_consent_at_read_insert_end():
    """Inserts at the end cannot be corrected, because this might be the only read going further..."""
    reads = [
        Read("ATCCTTCAGT", 0, uncertainty_generator=certain_uncertainty_generator),
        Read("ATCCTT", 0, uncertainty_generator=certain_uncertainty_generator),
        Read("ATCCTT", 0, uncertainty_generator=certain_uncertainty_generator),
    ]
    graph = read_multiple_sequence_alignment(
        reads,
        alignment_bonus=TEST_ALINGMENT_BONUS,
        probabilistic=False,
    )
    print(consent_at_read(reads[0]))
    assert consent_at_read(reads[0]) == make_regular(A, T, C, C, T, T,C,A,G,T)



def test_consent_at_read_delete_start():
    """Semantically: Read only needs to be corrected within its scope.
      So inserts in the front and deletes in the back will be disregarded"""
    reads = [
        Read("TCCTTC", 0, uncertainty_generator=certain_uncertainty_generator),
        Read("ATCCTTC", 0, uncertainty_generator=certain_uncertainty_generator),
        Read("ATCCTTC", 0, uncertainty_generator=certain_uncertainty_generator),
    ]
    graph = read_multiple_sequence_alignment(
        reads,
        alignment_bonus=TEST_ALINGMENT_BONUS,
        probabilistic=False,
    )
    assert consent_at_read(reads[0]) == make_regular(T, C, C, T, T, C)


def test_consent_at_read_insert_start():
    """Additional bases at the start cannot be corrected, because this might be the only read beginning that early..."""

    reads = [
        Read("ATCCTTC", 0, uncertainty_generator=certain_uncertainty_generator),
        Read("TCCTTC", 0, uncertainty_generator=certain_uncertainty_generator),
        Read("TCCTTC", 0, uncertainty_generator=certain_uncertainty_generator),
    ]
    graph = read_multiple_sequence_alignment(
        reads,
        alignment_bonus=TEST_ALINGMENT_BONUS,
        probabilistic=False,
    )
    assert consent_at_read(reads[0]) == make_regular(A, T, C, C, T, T, C)


def test_consent_corrected_read():
    reads = [
        Read("ATCGTTC", 0, uncertainty_generator=certain_uncertainty_generator),
        Read("ATCATTC", 0, uncertainty_generator=certain_uncertainty_generator),
        Read("ATCATTC", 0, uncertainty_generator=certain_uncertainty_generator),
    ]
    graph = read_multiple_sequence_alignment(
        reads,
        alignment_bonus=TEST_ALINGMENT_BONUS,
        probabilistic=False,
    )
    assert consent_corrected_read(reads[0], probabilistic=False).uncertain_text == \
        list(map(certain_uncertainty_generator,"ATCATTC"))


def test_consent_corrected_read_probabilistic():
    reads = [
        Read("ATCGTTC", 0, uncertain_text=just_those(A, T, C, G, T, T, C)),
        Read("ATCATTC", 0, uncertain_text=just_those(A, T, C, A, T, T, C)),
        Read("ATCATTC", 0, uncertain_text=just_those(A, T, C, A, T, T, C)),
    ]
    graph = read_multiple_sequence_alignment(
        reads,
        alignment_bonus=TEST_ALINGMENT_BONUS,
        probabilistic=True,
    )
    assert consent_corrected_read(reads[0], probabilistic=True).uncertain_text == \
        [*just_those(A,T,C), mix(G,A,A), *just_those(T,T,C)]


def test_sequence_of_read():
    r = Read("ATCGTTC", 0, uncertainty_generator=certain_uncertainty_generator)
    assert sequence_of_read(r, probabilistic=False) == make_regular(A, T, C, G, T, T, C)
    assert sequence_of_read(r, probabilistic=True) == make_uncertain_regular(
        *just_those(A, T, C, G, T, T, C)
    )


def test_correct_reads_with_consent():
    reads = [
        Read("ATCATTC", 0, uncertain_text=make_certain_uncertain("ATCGTTC")),
        Read("ATCATTC", 0, uncertainty_generator=certain_uncertainty_generator),
        Read("ATCATTC", 0, uncertainty_generator=certain_uncertainty_generator),
    ]

    corrected_reads = correct_reads_with_consens(reads, probabilistic=False, alignment_bonus=TEST_ALINGMENT_BONUS)
    for r in corrected_reads:
        assert r.uncertain_text == make_certain_uncertain("ATCATTC")


def test_correct_reads_with_consens():
    reads = [
        Read("ATCATTC", 0, uncertainty_generator=certain_uncertainty_generator,uncertain_text=make_certain_uncertain("ATCGTTC")),
        Read("ATCATTC", 0, uncertainty_generator=certain_uncertainty_generator),
        Read("ATCATTC", 0, uncertainty_generator=certain_uncertainty_generator),
    ]
    corrected_reads = correct_reads_with_consens(reads, probabilistic=False, alignment_bonus=TEST_ALINGMENT_BONUS)
    assert [r.predicted_text for r in corrected_reads] == ["ATCATTC" for _ in range(3)]


def test_create_read_without_measurement():
    r = Read("ATC", 0, uncertainty_generator=certain_uncertainty_generator)
    assert r.uncertain_text == make_certain_uncertain("ATC")


# parametrized test for checking order of adding sequences does not influence result

# machen, dass reads vor dem in probablistic graph tun auf genauigkeit gerundet werden
# machen, dass consent aus graph auch auf grobkörnigkeit gerundet wird

# read consent testen bei inserts und delete