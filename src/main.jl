using Agents, Random


const Period = Int64 # step of simulation
const Effort = Float64
const Wage = Float64


# ========================================
# Structs

mutable struct Worker <: AbstractAgent
    id::Int64
    Theta::Float64
    employer::Any # Ideally, but circular: Union{Firm, Nothing}
    effort::Float64
    utility::Float64
    friends::Array{Worker, 1}
end

mutable struct Firm
    id::Int64
    book::Array{Worker, 1} # Dict # {Worker, Int64}
end

# ========================================
# Worker

function get_size(firm::Nothing)
    return 0
end

function get_size(firm::Firm)
    return length(firm.book)
end

function get_efforts(firm::Nothing)
    return 0.0 # TODO check
end

function get_efforts(firm::Firm)
    if isempty(firm.book)
        return 0.0
    else
        return sum([worker.effort for worker in keys(firm.book)])
    end
end

function get_output(firm::Nothing)
    return 0.0
end

function get_output(firm::Firm)
    Es = get_efforts(firm)
    return Es + Es^2
end

function get_neighbor_firms(worker::Worker, model::AgentBasedModel)
    if rand(model.rng) > 0.01
        return unique([friend.employer for friend in worker.friends
                           if friend.employer != worker.employer])
    else # generates Zipf law:
        return unique([friend.employer for friend in random_agent(model).friends
                           if friend.employer != worker.employer])
    end
end

# ========================================
# Firm

function hire(firm::Firm, worker::Worker)
    push!(firm.book, worker)
end

function separate(firm::Firm, worker::Worker)
    deleteat!(firm.book, findall(x->x==worker, firm.book))
end

function compute_effort(Theta::Float64, firm::Firm)
    E = get_efforts(firm)
    return optimal_effort(Theta, E)
end

function get_best_firm(worker::Worker, startup::Firm, model::AgentBasedModel)
    # TODO model social network
    neighboring_firms = get_neighbor_firms(worker, model)
    new_firms = push!(neighboring_firms, startup)
    efforts = [compute_effort(worker.Theta, firm) for firm in new_firms]
    sizes = [get_size(firm) + 1 for firm in new_firms]
    outputs = [get_output(new_firms[i]) for i=1:length(new_firms)]
    utilities = [compute_utility(worker.Theta,
                                 outputs[i]==0.0 && sizes[i]==0.0 ? 1.0 : outputs[i]/sizes[i],
                                 efforts[i])
                 for i=1:length(new_firms)]
    best_index = argmax(utilities)
    return new_firms[best_index], efforts[best_index], utilities[best_index]
end

function update_efforts(firm::Nothing)
end

function update_efforts(firm::Firm)
    for worker in keys(firm.book)
        update_effort(worker)
    end
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
    o = get_output(worker.employer)
    s = get_size(worker.employer)
    worker.utility = compute_utility(worker.Theta,
                                     o==0.0 && s==0.0 ? 1.0 : o/s,
                                     worker.effort)
end

function separation(worker::Worker, firm::Firm)
    worker.employer = nothing
    separate(firm, worker)
end

function hiring(worker::Worker, firm::Firm)
    worker.employer = firm
    hire(firm, worker)
end

function migration(worker::Worker, new_firm::Firm)
    separation(worker, worker.employer)
    hiring(worker, new_firm)
end

function choose_firm(worker::Worker, max_firm_id::Int64, model::AgentBasedModel)
    startup = Firm(max_firm_id, Worker[])
    new_firm, new_effort, new_utility = get_best_firm(worker, startup, model)
    update_efforts(worker.employer)
    update_effort(worker)
    if new_utility > worker.utility
        migration(worker, new_firm)
        worker.effort = new_effort
        worker.utility = new_utility
    end
    return new_firm
end

function worker_step!(worker::Worker, model::AgentBasedModel)
    if rand() < model.active_workers
        new_firm = choose_firm(worker, model.max_firm_id + 1, model)
        model.max_firm_id = max(model.max_firm_id, new_firm.id)
        if new_firm.id == model.max_firm_id
            push!(model.firms, new_firm)
        end
    end
end

function firms(;
    num_workers = 10,
    active_workers = 0.4,
    num_friends = 4,
    seed = 42
)
    space = nothing
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
    for id in 1:num_workers
        Theta = rand(model.rng)
        effort = optimal_effort(Theta, 0.0)
        employer = nothing
        friends = Worker[]
        add_agent!(
            Worker(id,
                   Theta,
                   employer,
                   effort,
                   compute_utility(Theta, effort, effort),
                   friends
                   ),
            model
        )
    end

    for id in 1:num_workers
        w = model.agents[id]

        num_friends_w = 0
        while num_friends_w < num_friends
            f = random_agent(model)
            if f != w
                push!(w.friends, f)
                num_friends_w += 1
            end
        end
    end
    return model
end

model = firms()

step!(model, worker_step!, 10)

