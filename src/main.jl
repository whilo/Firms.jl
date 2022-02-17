using Agents, Random, Distributions


# ========================================
# Constants

const Effort = Float64
const Utility = Float64
const Wage = Float64
const Step = Int64
const FirmID = Int64
const WorkerID = Int64


# ========================================
# Structs

mutable struct Worker <: AbstractAgent
    id::WorkerID
    Theta::Float64
    employer::Any # Ideally, but circular: Union{Firm, Nothing}
    effort::Effort
    utility::Utility
    friends::Array{Worker, 1}
end

mutable struct Firm
    id::FirmID
    book::Set{Worker} # Dict # {Worker, Int64}
    creationStep::Step
    deletionStep::Step
end

function Firm(id::FirmID, book::Set{Worker})
    return Firm(id, book, 0, -1)
end

Base.show(io::IO, f::Firm) = begin
    print(io, "Firm: ", f.id, ", born: ", f.creationStep);
    if (f.deletionStep != -1)
        print(io, " died: ", f.deletionStep)
    end
end

Base.copy(w::Worker) = Worker(w.id, w.Theta, copy(w.employer), w.effort, w.utility, w.friends)

Base.copy(f::Firm) = Firm(f.id, copy(f.book), f.creationStep, f.deletionStep)


# ========================================
# Worker

function my_random_agent(model::AgentBasedModel)::Worker
    model.agents[rand(model.rng,
                      DiscreteUniform(1, length(model.agents)))]::Worker
end

function get_size(firm::Nothing)::Int64
    return 0
end

function get_size(firm::Firm)::Int64
    return length(firm.book)
end

function get_efforts(firm::Nothing)::Effort
    return 0.0 # TODO check
end

# CACHE_THRESHOLD = 10
# firm_cache = Dict()

# function get_efforts(firm::Firm)::Effort
#     # if rand() < 0.01
#     #     global firm_cache = Dict()
#     # end

#     if length(firm.book) > CACHE_THRESHOLD
#         if haskey(firm_cache, firm)
#             return firm_cache[firm]
#         else
#             effort = get_efforts_real(firm)
#             firm_cache[firm] = effort
#             return effort
#         end
#     else
#         return get_efforts_real(firm)
#     end
# end

function get_efforts(firm::Firm)::Effort
    s = 0.0
    if isempty(firm.book)
        return s
    else
        for worker in firm.book
            s += worker.effort
        end
        return s # sum([worker.effort for worker in firm.book])
    end
end

function get_output(firm::Nothing)::Float64
    return 0.0
end

function get_output(firm::Firm)::Float64
    Es = get_efforts(firm)
    return Es + Es^2
end

function get_neighbor_firms(worker::Worker, model::AgentBasedModel)::Array{Firm, 1}
    if rand(model.rng) > 0.01
        return [friend.employer::Firm for friend::Worker in worker.friends
                    # if friend.employer != worker.employer
                        ] # unique()
    else # generates Zipf law:
        return unique([friend.employer::Firm for friend::Worker in my_random_agent(model).friends
                               if friend.employer != worker.employer])
    end
end

function update_effort(worker::Worker)
    worker.effort = optimal_effort(worker.Theta, get_efforts(worker.employer) - worker.effort)
    outputs = get_output(worker.employer)
    sizes = get_size(worker.employer)
    worker.utility = compute_utility(worker.Theta, outputs/sizes, worker.effort)
end

function separation(worker::Worker, firm::Firm, step::Step)
    worker.employer = nothing
    separate(firm, worker, step)
end

function hiring(worker::Worker, firm::Firm)
    worker.employer = firm
    hire(firm, worker)
end

function migration(worker::Worker, new_firm::Firm, step::Step)
    separation(worker, worker.employer, step)
    hiring(worker, new_firm)
end

function get_best_firm(worker::Worker, startup::Firm, model::AgentBasedModel)::Tuple{Firm, Effort, Utility}
    neighboring_firms = get_neighbor_firms(worker, model)
    new_firms_ = push!(neighboring_firms, startup)
    # operate on copies from now on
    new_firms = [copy(f) for f in new_firms_] # deepcopy(new_firms_)
    worker = copy(worker)
    worker.employer = copy(worker.employer)
    for f in new_firms
        # speculatively hire the worker
        hiring(worker, f)
    end
    efforts = [compute_effort(worker.Theta, firm) for firm in new_firms]
    sizes = [get_size(firm) + 1 for firm in new_firms]
    outputs = [get_output(firm) for firm in new_firms]
    utilities = [compute_utility(worker.Theta,
                                 outputs[i]/sizes[i],
                                 efforts[i])
                 for i=1:length(new_firms)]
    best_index = argmax(utilities)
    # return reference to actual firm
    return new_firms_[best_index], efforts[best_index], utilities[best_index]
end

function choose_firm(worker::Worker, new_firm_id::Int64, model::AgentBasedModel)::Firm
    startup = Firm(new_firm_id, Set{Worker}(), model.step, -1)
    new_firm, new_effort, new_utility = get_best_firm(worker, startup, model)
    if new_firm != worker.employer
        migration(worker, new_firm, model.step)
        worker.effort = new_effort
        worker.utility = new_utility
    end
    return new_firm
end


# ========================================
# Firm

function hire(firm::Firm, worker::Worker)
    push!(firm.book, worker)
end

function separate(firm::Firm, worker::Worker, step::Step)
    # deleteat!(firm.book, findall(x->x==worker, firm.book))
    delete!(firm.book, worker)
    if length(firm.book) == 0
        firm.deletionStep = step
    end
end

function compute_effort(Theta::Float64, firm::Firm)::Effort
    E = get_efforts(firm)
    return optimal_effort(Theta, E)
end

function update_efforts(firm::Nothing)
end

function update_efforts(firm::Firm)
    for worker in firm.book
        update_effort(worker)
    end
end


# ========================================
# Formulas

function compute_utility(Theta::Float64, wage::Wage, effort::Effort)::Utility
    W = wage
    E = effort
    Θ = Theta
    return W^Θ * (1-E)^(1-Θ)
end

function optimal_effort(Theta::Float64, team_effort::Effort)::Effort
    # See Formula 2 from "Endogenous Firms and Their Dynamics (Axtell, 2013)"
    E = team_effort
    Θ = Theta
    e_star = (-1 - 2*(E - Θ) + (1 + 4*Θ^2*(1+E)*(2+E))^(1/2)) / (2 * (1 + Θ))
    return max(0, min(1, e_star))
end


# ========================================
# Simulation

function worker_step!(worker::Worker, model::AgentBasedModel)
    if rand(model.rng) < model.active_workers::Float64
        # update state
        update_efforts(worker.employer::Firm)
        update_effort(worker)
        model.step += 1

        # consider creation of new companies
        old_max_firm_id = model.max_firm_id::FirmID
        new_firm = choose_firm(worker, old_max_firm_id + 1, model)
        if new_firm.id > old_max_firm_id
            model.max_firm_id = new_firm.id
            # println("created new startup with id: ", new_firm.id)
            push!(model.firms, new_firm)
        end
    end
end

function firms(;
    num_workers = 10,
    active_workers = 0.04,
    num_friends = [2, 6],
    seed = 42
)
    space = nothing
    num_firms = num_workers
    properties = Dict{Symbol, Any}(
        :num_workers => num_workers,
        :active_workers => active_workers,
        :num_friends => num_friends,
        :firms => Firm[],
        :max_firm_id => num_firms,
        :step => 0
    )
    model = AgentBasedModel(
        Worker,
        space,
        scheduler = Schedulers.randomly, # TODO schedule subset
        properties = properties,
        rng = MersenneTwister(seed)
    )
    workers = Worker[]
    for wid in 1:num_workers
        Theta = rand(model.rng)
        effort = optimal_effort(Theta, 0.0)
        employer = nothing
        friends = Worker[]
        worker = Worker(wid,
                        Theta,
                        employer,
                        effort,
                        compute_utility(Theta, effort, effort),
                        friends)
        firm = Firm(wid, Set([worker]))
        worker.employer = firm
        add_agent!(worker, model)
        # add one company for each worker
        push!(model.firms::Array{Firm, 1}, firm)
    end

    # social network
    for wid in 1:num_workers
        w = model.agents[wid]::Worker

        num_friends_max = rand(model.rng,
                               DiscreteUniform(num_friends[1], num_friends[2]))
        num_friends_w = 0
        while num_friends_w < num_friends_max
            f = my_random_agent(model)

            #f = random_agent(model)::Worker
            if f != w
                push!(w.friends, f)
                num_friends_w += 1
            end
        end
    end

    return model
end

# ========================================
# Evaluation

using Plots;
using ProfileView;

function get_avg_efforts(model)
    efforts = 0
    for (id, worker) in model.agents
        efforts += worker.effort
    end
    return efforts / length(model.agents)
end

function get_avg_utilities(model)
    utilities = 0
    for (id, worker) in model.agents
        utilities += worker.utility
    end
    return utilities / length(model.agents)
end

function playground()
    model = @time firms(num_workers=10000, seed=rand(1:1001));


    @profview step!(model, worker_step!, 1);

    @code_warntype worker_step!(model.agents[1], model)

    @code_warntype choose_firm(model.agents[1], 1, model)

    @code_warntype get_best_firm(model.agents[1], Firm(1, Worker[]), model)

    sum([length(f.book) for f in model.firms])


    worker_counts = []
    avg_efforts = []
    avg_utilities = []
    max_steps = 2000
    for i in 1:max_steps
        @time step!(model, worker_step!, 1);
        push!(worker_counts, [length(f.book) for f in model.firms])
        push!(avg_efforts, get_avg_efforts(model))
        push!(avg_utilities, get_avg_utilities(model))
        println("Iteration: ", i)
    end

    plot(avg_efforts, title="Avg Effort")  # Axtell AAMAS Figure 3
    plot(avg_utilities, title="Avg Utility") #            Figure 4

    non_zero_worker_counts = [sort(filter(x -> x!=0, worker_count))
                              for worker_count in worker_counts]

    histogram(non_zero_worker_counts[end], yscale=:log10) # Figure 6
    sum(worker_count)
    length(worker_count)

    model.step

end
