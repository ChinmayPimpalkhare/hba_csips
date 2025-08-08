using Convex, Random ,PProf ,Zygote ,BenchmarkTools ,Plots ,JuMP ,CSV ,DataFrames, Dates,Profile ,ProfileView ,MathOptInterface ,Clarabel ,Logging ,ForwardDiff ,PyFormattedStrings, DataStructures 

function f(x)
    return x[1]
end

function g_u_cvx(x_var, y, d_x, d_y)
    dot_1_x = sum(x_var[i] for i in 2:d_x + 1)
    dot_1_y = sum(y[j] for j in 1:d_y)
    f_x_y = -(dot_1_y^3) + (dot_1_x)*(dot_1_y)
    return f_x_y - x_var[1]
end

function compute_g_of_omega(u, initial_guess, max_iters, d_x, d_y)
    n = d_x
    m = d_y
    c = 2.0 + 0.75*d_y*d_y/d_x
    x_var = Variable(n + 1)
    objective = x_var[1] 
    u_parts = [u[(i-1)*m+1:i*m] for i in 1:(d_x + 1)]
    constraints = [g_u_cvx(x_var, part, d_x, d_y) <= -1e-4 for part in u_parts]
    for i in 2:(d_x + 1)
        push!(constraints, x_var[i] <= c)
        push!(constraints, -x_var[i] <= c)
    end 

    x_var.value = initial_guess

    # Solve convex optimization problem and capture output
    problem = minimize(objective, constraints)
    
    optimizer_factory = () -> Clarabel.Optimizer(; 
    max_iter = max_iters,
    verbose = false,
    # tol_gap_abs = 1e-5,
    # tol_gap_rel = 1e-5,
    # tol_feas = 1e-5
    )

    solve!(problem, optimizer_factory)
    
    # Extract results
    optimal_point = vec(evaluate(x_var))
    lagrange_multipliers_temp = [c.dual for c in constraints[1:(d_x + 1)]]
    lagrange_multipliers = repeat(lagrange_multipliers_temp, inner = m)

    # Compute gradient
    grad_constraints = [
        ForwardDiff.gradient(v -> g_u_cvx(optimal_point, v, d_x, d_y), part) for part in u_parts
    ]
    
    final_grad = vcat(grad_constraints...)
    return optimal_point, lagrange_multipliers, final_grad
end

mutable struct Node
    low_domain::Vector{Float64}  #predefined
    up_domain::Vector{Float64}   #predefined
    c_point::Vector{Float64}
    reward::Float64
    gradient::Vector{Float64}
    depth::Int                   #predefined
    visited::Bool             
    id::Int                      #predefined
    index::Int                   #predefined
    parent::Int                  #predefined
    children::Vector{Int}
    is_leaf::Bool
    times_visited::Int
    initial_guess::Vector{Float64}

    function Node(low_domain::Vector{Float64}, up_domain::Vector{Float64}, 
        depth::Int, id::Int, index::Int, parent::Int)

        c_point = (low_domain .+ up_domain) ./ 2
        reward = 0 
        gradient = zeros(length(low_domain))
        visited = false 
        children = Int[]
        is_leaf = true
        times_visited = 0 
        initial_guess = c_point

        new(low_domain, up_domain, c_point, reward, 
        gradient, depth, visited, id, 
        index, parent, children,is_leaf, times_visited, 
        initial_guess)
    end
end

mutable struct Partition
    low_domain::Vector{Float64}
    up_domain::Vector{Float64}
    d_x::Int
    d_y::Int
    root::Node
    nodes_by_depth::Dict{Int, BinaryMaxHeap{Tuple{Float64, Int64, Node}}}
    overall_nodes_heap::BinaryMaxHeap{Tuple{Float64, Int64, Node}}
    height_tree::Int
    current_node::Node 
    global_node_id::Int

    function Partition(low_domain::Vector{Float64}, up_domain::Vector{Float64}, d_x::Int, d_y::Int)
        root = Node(low_domain, up_domain, 0, 1, 1, -1)
        heap0 = BinaryMaxHeap{Tuple{Float64, Int64, Node}}()
        overall_nodes_heap = BinaryMaxHeap{Tuple{Float64, Int64, Node}}()
        push!(heap0, (root.reward, root.id, root))
        push!(overall_nodes_heap, (root.reward, root.id, root))
        new(low_domain, up_domain, d_x, d_y, root, Dict(0 => heap0),overall_nodes_heap, 0, root, 2)
    end
end

function make_children!(partition::Partition, parent::Node, 
    add_layer::Bool, split_dim::Int, K_m_h::Int )
    
    
    low1, up1 = copy(parent.low_domain), copy(parent.up_domain)
    low2, up2 = copy(parent.low_domain), copy(parent.up_domain)
    up1[split_dim] = (parent.low_domain[split_dim] + parent.up_domain[split_dim]) / 2
    low2[split_dim] = (parent.low_domain[split_dim] + parent.up_domain[split_dim]) / 2

    child1_id, child2_id = partition.global_node_id, partition.global_node_id + 1
    partition.global_node_id += 2

    child1 = Node(low1, up1, parent.depth + 1, child1_id, 2 * parent.index - 1, parent.id)
    child2 = Node(low2, up2, parent.depth + 1, child2_id, 2 * parent.index, parent.id)
    
    #child 1 
    x_opt1, lag_mul1, fin_grad1 = compute_g_of_omega(child1.c_point, child1.initial_guess, K_m_h, partition.d_x, partition.d_y)
    child1.reward = f(x_opt1)
    child1.gradient = lag_mul1 .* fin_grad1
    child1.initial_guess = x_opt1
    child1.times_visited += K_m_h   
    
    #child 2 
    x_opt2, lag_mul2, fin_grad2 = compute_g_of_omega(child2.c_point, child2.initial_guess, K_m_h, partition.d_x, partition.d_y)
    child2.reward = f(x_opt2)
    child2.gradient = lag_mul2 .* fin_grad2
    child2.initial_guess = x_opt2
    child2.times_visited += K_m_h   

    parent.children, parent.is_leaf, parent.visited = [child1_id, child2_id], false, true 

    if !haskey(partition.nodes_by_depth, parent.depth + 1)
        partition.nodes_by_depth[parent.depth + 1] = BinaryMaxHeap{Tuple{Float64, Int64, Node}}()
    end

    push!(partition.nodes_by_depth[parent.depth + 1], (child1.reward, child1.id, child1))
    push!(partition.nodes_by_depth[parent.depth + 1], (child2.reward, child2.id, child2))
    push!(partition.overall_nodes_heap, (child1.reward, child1.id, child1))
    push!(partition.overall_nodes_heap, (child2.reward, child2.id, child2))


    if add_layer
        partition.height_tree += 1
    end

end

#FIXME Play around with the thresholds
function compute_E_m_h(h_max, h, m_max, m)
    E_m_h = floor(Int, h_max / (h * m))
    return E_m_h
end

function compute_K_m_h(h_max, h, m_max, m)
    K_m_h = floor(Int, h_max / (h * m))
    return K_m_h
end

function compute_N_p(p::Int64)
    N_p = 2^p
    return N_p
end

function compute_F_p(p::Int64, h_max)
    F_p = floor(Int, h_max/2)
    return F_p
end

function expand_top_m_stroquool!(
    partition::Partition, h::Int, result::Dict{Int, Node}, 
    iter_count::Int, h_max::Int, m_max::Int)
    original_heap = partition.nodes_by_depth[h]
    temp_heap = deepcopy(original_heap)  # avoid mutating original 
    for m in 1:m_max
        E_m_h = compute_E_m_h(h_max, h, m_max, m)
        K_m_h = compute_K_m_h(h_max, h, m_max, m)
        skipped_nodes = []

        # Try to find the best node satisfying the visit E_m_h
        while !isempty(temp_heap)
            current_node = pop!(temp_heap)[3]

            if current_node.times_visited >= E_m_h
                ##[x] Randomization works better
                split_dim = rand(1:length((current_node.gradient)))
                make_children!(partition, current_node, true, split_dim, K_m_h)

                result[iter_count] = current_node
                iter_count += 1
                break  # Only one expansion per (h, m) pair
            else
                push!(skipped_nodes, current_node)
            end
        end

        # Reinsert skipped nodes back into heap for future m's
        for node in skipped_nodes
            push!(temp_heap, (node.reward, node.id, node))  # Use random tiebreaker if needed
        end
    end

    return iter_count
end

function validation_loop_stroquool!(
    partition::Partition, p_max::Int, h_max::Int)
    original_heap = partition.overall_nodes_heap
    temp_heap = deepcopy(original_heap)  # avoid mutating original
    best_nodes = Dict{Int64, Node}()

    for p in 1:p_max
        N_p = compute_N_p(p)
        F_p = compute_F_p(p, h_max)
        skipped_nodes = []

        # Try to find the best node satisfying the visit threshold
        while !isempty(temp_heap)
            current_node = pop!(temp_heap)[3]

            if current_node.times_visited >= N_p
                x_opt, _, _ = compute_g_of_omega(current_node.c_point, current_node.initial_guess, current_node.times_visited + F_p, partition.d_x, partition.d_y)
                current_node.reward = f(x_opt)
                current_node.times_visited += F_p   

                best_nodes[p] = current_node
                break  # Only one expansion per (h, m) pair
            else
                push!(skipped_nodes, current_node)
            end
        end

        # Reinsert skipped nodes back into heap for future m's
        for node in skipped_nodes
            push!(temp_heap, (node.reward, node.id, node))  # Use random tiebreaker if needed
        end
    end
    return best_nodes
end

function run_algorithm_StroquOOL!(partition::Partition, T::Int64)
    h_max = T
    p_max = floor(Int, log2(h_max))

    # STORE THE RESULT 
    result = Dict{Int64, Node}()    
    iter_count = 1
    
    # PERFORM THE PROCESSES RELEVANT TO THE ROOT NODE 
    root_node = partition.root
    split_dim = argmax(abs.(root_node.gradient))
    root_K_m_h = compute_K_m_h(h_max, 1, h_max, 1)
    make_children!(partition, root_node, true, split_dim, root_K_m_h)

    # EXPLORATION LOOP 
    for h in 1:h_max
        m_max = floor(Int, h_max/h)  
        iter_count = expand_top_m_stroquool!(partition, h, result, iter_count, h_max, m_max)
    end

    # CROSS-VALIDATION LOOP
    best_nodes = validation_loop_stroquool!(partition, p_max, h_max)  
      #// BUG ---> Incorrect application of best_nodes 
    result = best_nodes
    return result
end

function plot_rewards(node_stack_expand)
    rewards = []
    max_rewards = []
    node_ids = []
    max_reward = -Inf
    best_node = nothing

    # Iterate over the node stack and collect rewards and node IDs
    for (iter, node) in node_stack_expand
        push!(rewards, node.reward)
        push!(node_ids, node.id)
        
        # Update the node with the maximum reward
        if node.reward > max_reward
            max_reward = node.reward
            best_node = node
        end
        push!(max_rewards, max_reward)
    end
    Plots.default(overwrite_figure = false)
    plot_obj1 = plot(1:length(rewards), [rewards, max_rewards], xlabel="Iteration", ylabel="Reward", 
                    title="Reward vs Iteration", legend=false, lw=2)
    display(plot_obj1) 

    return best_node
end

function log_to_csv(partition::Partition)
    println("Global Id", partition.global_node_id)
    ids, low_domains, up_domains, c_points, rewards, gradients, depths, visiteds, indexs, parents, 
    is_leafs, times_visiteds, initial_guesss, correct_rewards, correct_xs = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    
    for h_i in keys(partition.nodes_by_depth)
        temp_heap = deepcopy(partition.nodes_by_depth[h_i])
        while !isempty(temp_heap)
            current_node = pop!(temp_heap)[3]
            x_opt, _ , _ = compute_g_of_omega(current_node.c_point, current_node.initial_guess, 100, partition.d_x, partition.d_y)
            push!(ids, current_node.id)
            push!(low_domains, isempty(current_node.low_domain) ? "[]" : string("[", join(current_node.low_domain, ","), "]"))
            push!(up_domains, isempty(current_node.up_domain) ? "[]" : string("[", join(current_node.up_domain, ","), "]"))
            push!(c_points, isempty(current_node.c_point) ? "[]" : string("[", join(current_node.c_point, ","), "]"))
            push!(rewards, current_node.reward)
            push!(gradients, isempty(current_node.gradient) ? "[]" : string("[", join(current_node.gradient, ","), "]"))
            push!(depths, current_node.depth)
            push!(visiteds, current_node.visited)
            push!(indexs, current_node.index)
            push!(parents, current_node.parent)
            push!(is_leafs, current_node.is_leaf)
            push!(times_visiteds, current_node.times_visited)
            push!(initial_guesss, isempty(current_node.initial_guess) ? "[]" : string("[", join(current_node.initial_guess, ","), "]"))
            push!(correct_rewards, f(x_opt))
            push!(correct_xs, x_opt)
        end 
    end 
    
    df = DataFrame((
        id = ids,
        #low_domain = low_domains,
        #up_domain = up_domains,
        c_point = c_points,
        reward_approx = rewards,
        #gradient = gradients,
        depth = depths,
        #visited = visiteds,
        #index = indexs,
        #parent = parents,
        #is_leaf = is_leafs,
        times_visited = times_visiteds,
        initial_guess = initial_guesss, 
        reward_true = correct_rewards, 
        x_true = correct_xs 
    ))
    
    CSV.write("toy_example_highdim_dotprod_with_cpoint.csv", df)
end


#// [x] Change these for every problem
global d_x = 10
global d_y = 3 
global low_domain = [-1.0 for _ in 1:(d_x + 1)*(d_y)]
global up_domain  = [1.0 for _ in 1:(d_x + 1)*(d_y)]
global partition = Partition(low_domain, up_domain, d_x, d_y)
global T = 400
node_stack_expand =  @time run_algorithm_StroquOOL!(partition, T)
best_node = plot_rewards(node_stack_expand)
log_to_csv(partition)

sorted_keys = sort(collect(keys(node_stack_expand)))

# Extract nodes in sorted order
sorted_nodes = [node_stack_expand[k] for k in sorted_keys]

for node in sorted_nodes
    println(node.c_point)
    println(node.reward)
    println(node.depth)
    println(" ")
end
