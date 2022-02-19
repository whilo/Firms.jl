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
    employer::FirmID
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

VOID_FIRM_ID = 0

Base.show(io::IO, f::Firm) = begin
    print(io, "Firm: ", f.id, ", born: ", f.creationStep);
    if (f.deletionStep != -1)
        print(io, " died: ", f.deletionStep)
    end
end

Base.copy(w::Worker) = Worker(w.id, w.Theta, w.employer, w.effort, w.utility, w.friends)

Base.copy(f::Firm) = Firm(f.id, copy(f.book), f.creationStep, f.deletionStep)


# ========================================
# Worker

function my_random_agent(model::AgentBasedModel)::Worker
    model.agents[rand(model.rng,
                      DiscreteUniform(1, length(model.agents)))]::Worker
end

function get_size(firm::Firm)::Int64
    return length(firm.book)
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
    else # sum up
        for worker in firm.book
            s += worker.effort
        end
        return s
    end
end

function get_output(firm::Firm)::Float64
    if firm.id == VOID_FIRM_ID
        return 0.0
    end

    Es = get_efforts(firm)
    return Es + Es^2
end

function get_neighbor_firms(worker::Worker, model::AgentBasedModel)::Array{Firm, 1}
    firms = model.firms
    if rand(model.rng) > 0.01
        return [firms[friend.employer]::Firm for friend::Worker in worker.friends
                    # if friend.employer != worker.employer
                        ] # unique()
    else # generates Zipf law:
        return unique([firms[friend.employer]::Firm for friend::Worker in my_random_agent(model).friends
                               if friend.employer != worker.employer])
    end
end

function update_effort(model::AgentBasedModel, worker::Worker)
    firms = model.firms
    employer = firms[worker.employer]::Firm
    worker.effort = optimal_effort(worker.Theta, get_efforts(employer) - worker.effort)
    outputs = get_output(employer)
    sizes = get_size(employer)
    worker.utility = compute_utility(worker.Theta, outputs/sizes, worker.effort)
end

function separation(model::AgentBasedModel, worker::Worker, firm::Firm)
    step = model.scheduler.n::Step
    worker.employer = VOID_FIRM_ID

    # firm
    delete!(firm.book, worker)
    if length(firm.book) == 0
        firm.deletionStep = step
    end
end

function hiring(worker::Worker, firm::Firm)
    worker.employer = firm.id

    # firm
    @assert firm.id != VOID_FIRM_ID
    push!(firm.book, worker)
end

function migration(model::AgentBasedModel, worker::Worker, new_firm::Firm)
    firms = model.firms
    separation(model, worker, firms[worker.employer])
    hiring(worker, new_firm)
end

function get_best_firm(worker::Worker, startup::Firm, model::AgentBasedModel)::Tuple{Firm, Effort, Utility}
    neighboring_firms = get_neighbor_firms(worker, model)
    new_firms_ = push!(neighboring_firms, startup)
    # operate on copies from now on
    new_firms = [copy(f) for f in new_firms_] # deepcopy(new_firms_)
    worker = copy(worker)
    # worker.employer = copy(worker.employer)
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
    startup = Firm(new_firm_id, Set{Worker}(), model.scheduler.n, -1)
    new_firm, new_effort, new_utility = get_best_firm(worker, startup, model)
    firms = model.firms
    if new_firm != firms[worker.employer]
        migration(model, worker, new_firm)
        worker.effort = new_effort
        worker.utility = new_utility
    end
    return new_firm
end


# ========================================
# Firm

function compute_effort(Theta::Float64, firm::Firm)::Effort
    E = get_efforts(firm)
    return optimal_effort(Theta, E)
end

function update_efforts(model::AgentBasedModel, firmID::FirmID)
    firms = model.firms
    firm = firms[firmID]::Firm
    @assert firm.id != VOID_FIRM_ID

    for worker in firm.book
        update_effort(model, worker)
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

mutable struct WorkerScheduler
    n::Int # step number
    p::Float64
end

function (ws::WorkerScheduler)(model::ABM)
    ws.n += 1 # increment internal counter by 1 each time its called

    ids = collect(keys(model.agents))
    subset = randsubseq(model.rng, ids, ws.p)
    return subset

    # be careful to use a *new* instance of this scheduler when plotting!
    # if ms.n < 10
    #     return allids(model) # order doesn't matter in this case
    # else
    #     ids = collect(allids(model))
    #     # filter all ids whose agents have `w` less than some amount
    #     filter!(id -> model[id].w < ms.w, ids)
    #     return ids
    # end
end

function worker_step!(worker::Worker, model::AgentBasedModel)
    # update state
    update_efforts(model, worker.employer)
    update_effort(model, worker)

    # consider creation of new companies
    old_max_firm_id = model.max_firm_id::FirmID
    new_firm = choose_firm(worker, old_max_firm_id + 1, model)
    if new_firm.id > old_max_firm_id
        model.max_firm_id = new_firm.id
        # println("created new startup with id: ", new_firm.id)
        push!(model.firms, new_firm)
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
    ws = WorkerScheduler(0, active_workers)
    model = AgentBasedModel(
        Worker,
        space,
        scheduler = ws,
        properties = properties,
        rng = MersenneTwister(seed)
    )
    workers = Worker[]
    for wid in 1:num_firms
        Theta = rand(model.rng)
        effort = optimal_effort(Theta, 0.0)
        employer = wid
        friends = Worker[]
        worker = Worker(wid,
                        Theta,
                        employer,
                        effort,
                        compute_utility(Theta, effort, effort),
                        friends)
        firm = Firm(wid, Set([worker]))
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

function get_avg_efforts(model::AgentBasedModel)
    efforts = 0
    for (id, worker) in model.agents
        efforts += worker.effort
    end
    return efforts / length(model.agents)
end

function get_avg_utilities(model::AgentBasedModel)
    utilities = 0
    for (id, worker) in model.agents
        utilities += worker.utility
    end

    return utilities / length(model.agents)
end

function test1()
    num_workers = 10000
    model = firms(num_workers=num_workers, seed=rand(1:1001));
    for i in 1:20
        step!(model, worker_step!, 1);
    end
    @assert sum([length(f.book) for f in model.firms]) == num_workers
end

function diagnosis_plots(models::Array{AgentBasedModel, 1})
    avg_efforts = [get_avg_efforts(model) for model in models];
    avg_utilities = [get_avg_utilities(model) for model in models];
    worker_counts = [[length(f.book) for f in model.firms] for model in models];

    plot(avg_efforts, title="Avg Effort")  # Axtell AAMAS Figure 3

    plot(avg_utilities, title="Avg Utility") #            Figure 4

    non_zero_worker_counts = [sort(filter(x -> x!=0, worker_count))
                              for worker_count in worker_counts]

    histogram(non_zero_worker_counts[end], yscale=:log10) # Figure 6
end

function test_simulation(num_workers::Int64, num_steps::Int64, seed::Int64, path::String)
    model = firms(num_workers=num_workers, seed=seed);

    models = [deepcopy(model)];
    for i in 1:num_steps
        println("iteration: ", i)
        @time step!(model, worker_step!, 1);
        println("copying model: ")
        @time push!(models, deepcopy(model))
    end

    diagnosis_plots(models)
    # TODO save models
end

function playground()
    model = @time firms(num_workers=10000, seed=rand(1:1001));

    for i in 1:20
        @time step!(model, worker_step!, 1);
    end

    @profview step!(model, worker_step!, 1);

    @code_warntype worker_step!(model.agents[1], model)

    @code_warntype choose_firm(model.agents[1], 1, model)

    @code_warntype get_best_firm(model.agents[1], Firm(1, Worker[]), model)

    sum([length(f.book) for f in model.firms])


    models = [deepcopy(model)];
    max_steps = 10
    for i in 1:max_steps
        println("iteration: ", i)
        @time step!(model, worker_step!, 1);
        println("copying model: ")
        @time push!(models, deepcopy(model))
    end




    avg_efforts = [get_avg_efforts(model) for model in models];
    avg_utilities = [get_avg_utilities(model) for model in models];
    worker_counts = [[length(f.book) for f in model.firms] for model in models];

    plot(avg_efforts, title="Avg Effort")  # Axtell AAMAS Figure 3

    plot(avg_utilities, title="Avg Utility") #            Figure 4

    non_zero_worker_counts = [sort(filter(x -> x!=0, worker_count))
                              for worker_count in worker_counts]

    histogram(non_zero_worker_counts[end], yscale=:log10) # Figure 6


end
