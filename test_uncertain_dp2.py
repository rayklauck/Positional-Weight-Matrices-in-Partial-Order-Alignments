from uncertain_dp2 import *


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
    cost, trace = wiped_dp_memoized_function(start_graph_node, sequence)
    assert cost == 0
    assert len(trace) == 1
    assert trace[0].operation == DpOperation.MATCH
    assert trace[0].graph_node == start_graph_node


def test_dp_same_two_bases():
    start_graph_node = initial_graph_of(make_regular(C, G)).start_nodes[0]
    sequence = make_regular(C, G)
    cost, trace = wiped_dp_memoized_function(start_graph_node, sequence)
    assert cost == 0
    assert len(trace) == 2
    assert trace[0].operation == DpOperation.MATCH
    assert trace[0].graph_node.base == RegularBase(C)
    assert trace[1].operation == DpOperation.MATCH
    assert trace[1].graph_node.base == RegularBase(G)


def test_dp_same_multiple_bases():
    start_graph_node = initial_graph_of(make_regular(C, G, A, T)).start_nodes[0]
    sequence = make_regular(C, G, A, T)
    cost, trace = wiped_dp_memoized_function(start_graph_node, sequence)
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
    cost, trace = wiped_dp_memoized_function(start_graph_node, sequence)
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
    cost, trace = wiped_dp_memoized_function(start_graph_node, sequence)
    assert cost == 0
    assert len(trace) == 1
    assert trace[0].operation == DpOperation.END
    assert trace[0].graph_node == start_graph_node


def test_dp_one_insertion_scenario():
    start_graph_node = initial_graph_of(make_regular(G, T, A, C)).start_nodes[0]
    sequence = make_regular(G, A, C)
    cost, trace = wiped_dp_memoized_function(start_graph_node, sequence)
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
    cost, trace = wiped_dp_memoized_function(start_graph_node, sequence)
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
    start_graph_node = initial_graph_of(make_regular(G, A, C, T, T, G, C)).start_nodes[
        0
    ]
    sequence = make_regular(G, A, T, T, G, A, C)
    cost, trace = wiped_dp_memoized_function(start_graph_node, sequence)
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
    start_graph_node = initial_graph_of(
        make_regular(G, A, C, T, T, G, C, T, A, A)
    ).start_nodes[0]
    sequence = make_regular(G, A, G, T, T, G, T, A, A)
    cost, trace = wiped_dp_memoized_function(start_graph_node, sequence)
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
    cost, trace = wiped_dp_memoized_function(start_graph_node, sequence)
    add_trace_to_graph(sequence, trace)
    assert len(start_graph_node.read_nodes) == 2


def test_add_to_graph_multiple_match():
    start_graph_node = initial_graph_of(make_regular(A, T, G)).start_nodes[0]
    sequence = make_regular(A, T, G)
    cost, trace = wiped_dp_memoized_function(start_graph_node, sequence)
    add_trace_to_graph(sequence, trace)
    assert len(start_graph_node.read_nodes) == 2
    assert len(start_graph_node.layer.graph_nodes) == 1
    assert start_graph_node.layer.graph_nodes[0].base == RegularBase(A)
    assert len(start_graph_node.successors[0].layer.graph_nodes) == 1
    assert len(start_graph_node.successors[0].successors[0].layer.graph_nodes) == 1


def test_add_to_graph_replace():
    start_graph_node = initial_graph_of(make_regular(A, T, G)).start_nodes[0]
    sequence = make_regular(G, T, G)
    cost, trace = wiped_dp_memoized_function(start_graph_node, sequence)
    add_trace_to_graph(sequence, trace)
    assert len(start_graph_node.read_nodes) == 1
    assert len(start_graph_node.layer.graph_nodes) == 2
    bases_here = list(map(lambda x: x.base, start_graph_node.layer.graph_nodes))
    assert RegularBase(G) in bases_here
    assert RegularBase(A) in bases_here


def test_add_to_graph_replace2():
    start_graph_node = initial_graph_of(make_regular(A, T, G)).start_nodes[0]
    sequence = make_regular(A, C, G)
    cost, trace = wiped_dp_memoized_function(start_graph_node, sequence)
    add_trace_to_graph(sequence, trace)
    assert len(start_graph_node.successors) == 2


def test_add_to_graph_delete():
    start_graph_node = initial_graph_of(make_regular(A, T, G)).start_nodes[0]
    sequence = make_regular(A, C, T, G)
    add_to_graph_start_node(sequence, start_graph_node)
    assert len(start_graph_node.layer.graph_nodes) == 1
    assert len(start_graph_node.successors) == 2
    c_node = [s for s in start_graph_node.successors if s.base == RegularBase(C)][0]
    t_node = [s for s in start_graph_node.successors if s.base == RegularBase(T)][0]
    assert c_node.layer != t_node.layer
    assert c_node.successors[0].layer == t_node.layer


def test_add_to_graph_insert():
    start_graph_node = initial_graph_of(make_regular(G, T, A, C)).start_nodes[0]
    sequence = make_regular(G, A, C)
    add_to_graph_start_node(sequence, start_graph_node)
    assert len(start_graph_node.layer.graph_nodes) == 1
    assert len(start_graph_node.successors) == 2
    a_node = [s for s in start_graph_node.successors if s.base == RegularBase(A)][0]
    t_node = [s for s in start_graph_node.successors if s.base == RegularBase(T)][0]
    assert a_node.layer != t_node.layer
    assert t_node.successors[0].layer == a_node.layer


def test_add_to_graph_end():
    start_graph_node = initial_graph_of(make_regular(A, T)).start_nodes[0]
    sequence = make_regular(A)
    add_to_graph_start_node(sequence, start_graph_node)
    assert len(start_graph_node.layer.graph_nodes) == 1
    assert len(start_graph_node.successors) == 1
    assert start_graph_node.successors[0].base == RegularBase(T)


def test_multiple_sequences_alignment():
    start_graph_node = multiple_sequence_alignment([make_regular(A, T), make_regular(A)]).start_nodes[0]
    assert len(start_graph_node.layer.graph_nodes) == 1
    assert len(start_graph_node.successors) == 1
    assert start_graph_node.successors[0].base == RegularBase(T)


def test_add_three_same_sequences():
    start_graph_node = initial_graph_of(make_regular(A, T, C, G)).start_nodes[0]
    sequence = make_regular(A, T, C, G)
    add_to_graph_start_node(sequence, start_graph_node)
    add_to_graph_start_node(sequence, start_graph_node)

    assert len(start_graph_node.layer.graph_nodes) == 1
    assert len(start_graph_node.successors) == 1
    assert start_graph_node.successors[0].base == RegularBase(T)
    assert len(start_graph_node.successors[0].layer.graph_nodes) == 1
    assert len(start_graph_node.successors[0].successors) == 1
    assert start_graph_node.successors[0].successors[0].base == RegularBase(C)
    assert len(start_graph_node.read_nodes) == 3


def test_add_three_different_sequences():
    start_graph_node = initial_graph_of(make_regular(A, T, C, G)).start_nodes[0]
    sequence = make_regular(A, C, C, G)
    add_to_graph_start_node(sequence, start_graph_node)
    sequence = make_regular(A, G, C, G)
    add_to_graph_start_node(sequence, start_graph_node)

    assert len(start_graph_node.successors) == 3
    assert len({s.layer for s in start_graph_node.successors}) == 1


def test_arbitrary_alignment_to_graph():
    start_graph_node = initial_graph_of(make_regular(A, T, G)).start_nodes[0]
    sequence = make_regular(A, C, G)
    add_to_graph_start_node(sequence, start_graph_node)
    sequence = make_regular(A, C, G)
    cost, trace = wiped_dp_memoized_function(start_graph_node, sequence)
    assert cost == 0


def test_arbitrary_alignment_to_graph2():
    start_graph_node = initial_graph_of(make_regular(C, A, T, G, C, G)).start_nodes[0]
    sequence = make_regular(C, A, G, C, G)
    add_to_graph_start_node(sequence, start_graph_node)

    sequence = make_regular(C, A, C, G)
    cost, trace = wiped_dp_memoized_function(start_graph_node, sequence)
    assert cost == 1

    sequence = make_regular(C, A, T, G, G)
    cost, trace = wiped_dp_memoized_function(start_graph_node, sequence)
    assert cost == 1


def test_arbitrary_alignment_to_graph_not_branch_jumping():
    start_graph_node = initial_graph_of(make_regular(C, A, T, G, C, G)).start_nodes[0]
    sequence = make_regular(C, A, A, C, C, G)
    add_to_graph_start_node(sequence, start_graph_node)

    sequence = make_regular(C, A, T, C, C, G)
    cost, trace = wiped_dp_memoized_function(start_graph_node, sequence)
    assert cost == 1  # would be 0 if it could jump branches


def test_arbitrary_alignment_to_graph_take_multiple_branches():
    start_graph_node = initial_graph_of(
        make_regular(A, T, C, G, A, T, C, G, A, T, C, G)
    ).start_nodes[0]
    sequence = make_regular(A, T, C, T, A, T, C, G, A, T, C, G)
    add_to_graph_start_node(sequence, start_graph_node)
    sequence = make_regular(A, T, C, G, A, T, C, C, A, T, C, G)
    add_to_graph_start_node(sequence, start_graph_node)

    sequence = make_regular(A, T, C, T, A, T, C, C, A, T, C, G)
    cost, trace = wiped_dp_memoized_function(start_graph_node, sequence)
    assert cost == 0


def test_uncertain_base_scalar_product():
    uncertain_base_scalar_product(just(A), just(A)) == 0
    uncertain_base_scalar_product(just(A), just(C)) == 1
    uncertain_base_scalar_product([0.2, 0.5, 0.2, 0.1], [0, 0.2, 0.4, 0.4]) == 0.22
    uncertain_base_scalar_product([0.5, 0.5, 0, 0], [0.5, 0.5, 0, 0]) == 0.5


def test_align_positional_weight_matrix_reads_single_identical():
    start_graph_node = initial_graph_of(make_uncertain_regular(just(A))).start_nodes[0]
    sequence = make_uncertain_regular(just(A))
    cost, trace = wiped_dp_memoized_function(start_graph_node, sequence)
    assert cost == 0
    assert len(trace) == 1
    assert trace[0].operation == DpOperation.MATCH


def test_align_positional_weight_matrix_reads_single_non_identical():
    start_graph_node = initial_graph_of(
        make_uncertain_regular([0, 0, 0, 1])
    ).start_nodes[0]
    sequence = make_uncertain_regular([0, 0.3, 0.3, 0.4])
    cost, trace = wiped_dp_memoized_function(start_graph_node, sequence)
    assert cost == 0.6
    assert len(trace) == 1
    assert trace[0].operation == DpOperation.REPLACE


def test_align_positional_weight_matrix_reads_identical_sequence():
    start_graph_node = initial_graph_of(
        make_uncertain_regular(*just_those(A, T, C))
    ).start_nodes[0]
    sequence = make_uncertain_regular(*just_those(A, T, C))
    cost, trace = wiped_dp_memoized_function(start_graph_node, sequence)
    assert cost == 0


def test_align_positional_weight_matrix_reads_non_identical_sequence():
    start_graph_node = initial_graph_of(
        make_uncertain_regular(*just_those(A, T, C, T, T))
    ).start_nodes[0]
    sequence = make_uncertain_regular(*just_those(A, T, G, T, T))
    cost, trace = wiped_dp_memoized_function(start_graph_node, sequence)
    assert cost == 1


def test_align_positional_weight_matrix_reads_non_identical_sequence():
    start_graph_node = initial_graph_of(
        make_uncertain_regular(*probably_those(A, T, p=0.7))
    ).start_nodes[0]
    sequence = make_uncertain_regular(*probably_those(A, T, p=0.7))
    cost, trace = wiped_dp_memoized_function(start_graph_node, sequence)
    assert len(trace) == 2
    assert trace[0].operation == DpOperation.REPLACE
    assert trace[1].operation == DpOperation.REPLACE
    assert cost == 2 * PositionalWeightMatrixBase(probably(A)).dp_conversion_penalty(
        PositionalWeightMatrixBase(probably(A))
    )


def test_align_positional_weight_matrix_reads_non_identical_sequence():
    start_graph_node = initial_graph_of(
        make_uncertain_regular(*just_those(A, T, C, T, T))
    ).start_nodes[0]
    sequence = make_uncertain_regular(*just_those(A, T, G, T, T))
    cost, trace = wiped_dp_memoized_function(start_graph_node, sequence)
    assert cost == 1


def test_align_positional_weight_matrix_reads_non_identical_sequence2():
    start_graph_node = initial_graph_of(
        make_uncertain_regular(mix(A, C), mix(A, T), just(A))
    ).start_nodes[0]
    sequence = make_uncertain_regular(mix(A, C), mix(A, G), just(A))
    cost, trace = wiped_dp_memoized_function(start_graph_node, sequence)

    assert cost == 0.5 + 0.75
    assert len(trace) == 3
    assert (
        trace[0].operation == DpOperation.MATCH
    )  # same uncertain event. Does not mean there is no penalty
    assert trace[1].operation == DpOperation.REPLACE
    assert trace[2].operation == DpOperation.MATCH


def test_align_positional_weight_matrix_reads_detached():
    start_graph_node = initial_graph_of(
        make_uncertain_regular(just(A), just(T), mix(A, T))
    ).start_nodes[0]
    sequence = make_uncertain_regular(mix(A, G), just(C), mix(A, G))
    cost, trace = wiped_dp_memoized_function(start_graph_node, sequence)

    assert cost == 0.5  + 1+ 0.75
    assert len(trace) == 3
    assert (
        trace[0].operation == DpOperation.REPLACE
    )  # same uncertain event. Does not mean there is no penalty
    assert trace[1].operation == DpOperation.REPLACE
    assert trace[2].operation == DpOperation.REPLACE


def test_positional_weigth_matrix_add_to_graph():
    start_graph_node = initial_graph_of(
        make_uncertain_regular(mix(A, C), mix(A, T), just(A))
    ).start_nodes[0]
    sequence = make_uncertain_regular(mix(A, C), mix(A, G), just(A))
    add_to_graph_start_node(sequence, start_graph_node)

    assert start_graph_node.base == PositionalWeightMatrixBase(mix(A, C))
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
    p2 = make_uncertain_regular(just(C), mix(A, G), just(C), mix(A, G))
    start_graph_node = initial_graph_of(p1).start_nodes[0]
    add_to_graph_start_node(p2, start_graph_node)
    cost, trace = wiped_dp_memoized_function(start_graph_node, p1)
    assert cost == 0 + 0 + 0 + 0.5

    cost, trace = wiped_dp_memoized_function(start_graph_node, p2)
    assert cost == 0 + 0.5 + 0 + 0.5


def test_consent_regular_base():
    RegularBase.consent([RegularBase(A), RegularBase(A)]) == RegularBase(A)
    RegularBase.consent([RegularBase(A), RegularBase(C)]) in [RegularBase(A), RegularBase(C)]
    RegularBase.consent([RegularBase(A), RegularBase(C), RegularBase(C)]) == RegularBase(C)


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
        [PositionalWeightMatrixBase([0.5, 0.5, 0, 0]), PositionalWeightMatrixBase([0.6, 0.4, 0, 0])]
    ) == PositionalWeightMatrixBase([0.55, 0.45, 0, 0])


def test_consent_of_graph():
    graph = multiple_sequence_alignment([
          make_regular(A, G, C),
          make_regular(A, G, C),
                    make_regular(A, T, C),
])
    assert consent_of_graph(graph) == make_regular(A, G, C)


def test_consent_positional_weight_matrix():
    assert consent_of_graph(
        multiple_sequence_alignment([
            make_uncertain_regular(just(A), just(G), just(A)),
            make_uncertain_regular(just(A), just(T), just(A)),
        ])
    )==make_uncertain_regular(just(A), mix(T, G), just(A))


def test_consent_positional_weight_matrix_delete():
    assert consent_of_graph(
        multiple_sequence_alignment([
            make_uncertain_regular(just(A), just(T), just(A), just(C)),
            make_uncertain_regular(just(A), just(T), just(A), just(C)),
            make_uncertain_regular(just(A), just(A), just(C)),

        ])
    )==make_uncertain_regular(just(A), just(T), just(A), just(C))



def test_consent_positional_weight_matrix_insert():
    assert consent_of_graph(
        multiple_sequence_alignment([
            make_uncertain_regular(*just_those(A,T,A,C)),
            make_uncertain_regular(*just_those(A,T,A,C)),
            make_uncertain_regular(*just_those(A,T,G,A,C)),

        ])
    )==make_uncertain_regular(*just_those(A,T,A,C))


def test_consent_positional_weight_matrix_multiple_insert():
    assert consent_of_graph(
        multiple_sequence_alignment([
            make_uncertain_regular(*just_those(A,C,T,A,C)),
            make_uncertain_regular(*just_those(A,T,A,C)),
            make_uncertain_regular(*just_those(A,T,A,C)),
            make_uncertain_regular(*just_those(A,T,A,C)),
            make_uncertain_regular(*just_those(A,T,G,T,A,C)),

        ])
    )==make_uncertain_regular(*just_those(A,T,A,C))



def test_consent_positional_weight_matrix_complex():
    consent_of_graph(
        multiple_sequence_alignment([
            make_uncertain_regular(*just_those(A,T,A,G,C,C)),
            make_uncertain_regular(*just_those(A,C,A,C,C)),
            make_uncertain_regular(*just_those(A,T,A,C,C)),
            make_uncertain_regular(*just_those(A,C,G,C,C)),
            make_uncertain_regular(*just_those(A,C,A,C,C)),
        ])
    )==make_uncertain_regular(just(A),mix(T,T,C,C,C),mix(G,A,A,A,A),just(C),just(C))


def test_rigth_consent_from_overritten():
    PositionalWeightMatrixBase.consent(make_uncertain_regular(mix(A, C), just(A))) == PositionalWeightMatrixBase(mix(A,A, C))

# todo: account for flexible start (not necessarily at the start of the graph)
# todo: multiple starting points in general
