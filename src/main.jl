using Agents, Random


const Period = Int64 # step of simulation
const Effort = Float64
const Wage = Float64

mutable struct Worker <: AbstractAgent
    id::Int64
    Theta::Float64
    employer::Any # {Firm, Nothing}
    effort::Float64
    utility::Float64
    friends::Array{Worker, 1}
end

mutable struct Firm
    id::Int64
    book::Array{Worker, 1} # Dict # {Worker, Int64}
end

function get_neighbor_firms(worker::Worker)
    return unique([friend.employer for friend in worker.friends if friend.employer != worker.employer])
end

function get_size(firm::Firm)
    return length(firm.book)
end

function get_efforts(firm::Firm)
    return sum([worker.effort for worker in keys(firm.book)])
end

function get_output(firm::Firm)
    Es = get_efforts(firm)
    return Es + Es^2
end

function compute_effort(Theta::Float64, firm::Firm)
    E = get_efforts(firm)
    return optimal_effort(Theta, E)
end

function get_best_firm(worker::Worker, startup::Firm)
    # TODO model social network
    neighboring_firms = get_neighbor_firms(worker)
    new_firms = push!(neighboring_firms, startup)
    efforts = [compute_effort(worker.Theta, firm) for firm in new_firms]
    sizes = [get_size(firm) + 1 for firm in new_firms]
    outputs = [get_output(new_firms[i]) for i=1:length(new_firms)]
    utilities = [compute_utility(outputs[i]/sizes[i], efforts[i]) for i=1:length(new_firms)]
    best_index = argmax(utilities)
    return new_firms[best_index], efforts[best_index], utilities[best_index]
end

function update_efforts(firm::Firm)
end

function get_efforts(firm::Firm)
end

function compute_utility(Theta::Float64, wage::Wage, effort::Effort)
    W = wage
    E = effort
    Θ = Theta
    return W^Θ * (1-E)^(1-Θ)
end

function optimal_effort(Theta::Float64, effort::Effort)
    E = effort
    Θ = Theta
    e_star = (-1 - 2*(E - Θ) + (1 + 4*Θ^2*(1+E)*(2+E))^(1/2)) / (2 * (1 + Θ))
    return max(0, min(1, e_star))
end

function update_effort(worker::Worker)
    worker.effort = optimal_effort(worker.Theta, get_efforts(worker.employer) - worker.effort)
    worker.utility = compute_utility(worker.Theta, get_output(worker.employer)/get_size(worker.employer), worker.effort)
end

function migration(new_firm)
end

function choose_firm(worker::Worker, max_firm_id::Int64)
    startup = Firm(max_firm_id, Worker[])
    new_firm, new_effort, new_utility = get_best_firm(worker, startup)
    update_efforts(worker.employer)
    update_effort(worker)
    if new_utility > worker.utility
        migration(new_firm)
        worker.effort = new_effort
        worker.utility = new_utility
    end
    return new_firm
end

function worker_step!(worker::Worker, model)
    # TODO attach market to model
    if rand() < model.active_workers
        new_firm = choose_firm(worker, model.max_firm_id + 1)
        model.max_firm_id = max(model.max_firm_id, new_firm.id)
    end
end

function firms(;
    num_workers = 10,
    active_workers = 0.4,
    num_friends = 4,
    seed = 42
)
    space = nothing # GridSpace(dims)
    properties = Dict{Symbol, Any}(
        :num_workers => num_workers,
        :active_workers => active_workers,
        :num_friends => num_friends,
        :firms => Firm[],
        :max_firm_id => 0
    )
    model = AgentBasedModel(
        Worker,
        space,
        scheduler = Schedulers.randomly, # TODO schedule subset
        properties = properties,
        rng = MersenneTwister(seed)
    )
    for ag in 1:num_workers
        Theta = rand(model.rng)
        effort = optimal_effort(Theta, 0.0)
        employer = nothing
        friends = Worker[]
        add_agent!(
            Worker(ag,
                   Theta,
                   employer,
                   effort,
                   compute_utility(Theta, effort, effort),
                   friends
                   ),
            model
        )
        # TODO add friends
        # neigh for neigh in rd.sample(workers, number_neighbors) if neigh != worker
    end
    return model
end

model = firms()

step!(model, worker_step!, 10)


