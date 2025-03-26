using Random, Plots, DataStructures

# Define a simple Node struct with an id and reward
mutable struct Node
    id::Int
    reward::Float64
end

# Define Partition struct
mutable struct Partition
    nodes_by_depth::Dict{Int, Vector{Node}}  # Stores nodes at each depth
end

# Function to get the top m nodes using sorting (O(k log k))
function get_top_m_nodes_sorting(partition::Partition, depth::Int, m::Int)
    if !haskey(partition.nodes_by_depth, depth)
        return []
    end
    sorted_nodes = sort(partition.nodes_by_depth[depth], by=n -> -n.reward)
    return sorted_nodes[1:min(m, length(sorted_nodes))]
end

# Function to get the top m nodes using a heap (O(k log m))
function get_top_m_nodes_heap(partition::Partition, depth::Int, m::Int)
    if !haskey(partition.nodes_by_depth, depth)
        return []
    end

    heap = BinaryMinHeap{Tuple{Float64, Node}}()  # Min-heap based on reward

    # Process each node
    for node in partition.nodes_by_depth[depth]
        if length(heap) < m
            push!(heap, (node.reward, node))
        elseif node.reward > first(heap)[1]
            pop!(heap)
            push!(heap, (node.reward, node))
        end
    end

    # Extract elements correctly (Fix)
    result = Tuple{Float64, Node}[]  # Create an empty array to store elements
    while !isempty(heap)
        push!(result, pop!(heap))
    end

    return [pair[2] for pair in result]  # Extract the nodes
end

# Benchmark function for both methods
function benchmark_top_m_methods()
    k_values = [10, 10^2, 10^3, 10^4, 10^5]   # Number of nodes
    m_values = [3, 30, 30, 30, 30]     # Number of top nodes to retrieve
    runtimes_sorting = Float64[]
    runtimes_heap = Float64[]

    println("Benchmarking sorting vs heap method:")

    for i in 1:length(k_values)
        k = k_values[i]
        m = m_values[i]

        partition = Partition(Dict())  # Create a new partition
        Random.seed!(42)  # For reproducibility

        # Generate k nodes with random rewards at depth 1
        partition.nodes_by_depth[1] = [Node(j, rand()) for j in 1:k]

        # Measure sorting runtime
        runtime_sort = @elapsed get_top_m_nodes_sorting(partition, 1, m)
        push!(runtimes_sorting, runtime_sort)

        # Measure heap runtime
        runtime_heap = @elapsed get_top_m_nodes_heap(partition, 1, m)
        push!(runtimes_heap, runtime_heap)

        println("k = $k, m = $m --> Sorting: $runtime_sort sec, Heap: $runtime_heap sec")
    end

    # Plot results
    plot(k_values, runtimes_sorting, marker=:o, xlabel="Number of Nodes (k)", 
         ylabel="Runtime (seconds)", title="Sorting vs Heap Runtime",
         label="Sorting (O(k log k))", xscale=:log10, yscale=:log10, lw=2, color=:blue)
    plot!(k_values, runtimes_heap, marker=:s, label="Heap (O(k log m))", lw=2, color=:red)
end

# Run the benchmark
benchmark_top_m_methods()
