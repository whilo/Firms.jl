using Agents, Random

mutable struct SugarSeeker <: AbstractAgent
    id::Int
    pos::Dims{2}
    vision::Int
    metabolic_rate::Int
    age::Int
    max_age::Int
    wealth::Int
end


# Functions `distances` and `sugar_caps` produce a matrix for the distribution of sugar capacities."

function distances(pos, sugar_peaks, max_sugar)
    all_dists = Array{Int,1}(undef, length(sugar_peaks))
    for (ind, peak) in enumerate(sugar_peaks)
        d = round(Int, sqrt(sum((pos .- peak) .^ 2)))
        all_dists[ind] = d
    end
    return minimum(all_dists)
end

function sugar_caps(dims, sugar_peaks, max_sugar, dia = 4)
    sugar_capacities = zeros(Int, dims)
    for i in 1:dims[1], j in 1:dims[2]
        sugar_capacities[i, j] = distances((i, j), sugar_peaks, max_sugar)
    end
    for i in 1:dims[1]
        for j in 1:dims[2]
            sugar_capacities[i, j] = max(0, max_sugar - (sugar_capacities[i, j] ÷ dia))
        end
    end
    return sugar_capacities
end

"Create a sugarscape ABM"
function sugarscape(;
    dims = (50, 50),
    sugar_peaks = ((10, 40), (40, 10)),
    growth_rate = 1,
    N = 250,
    w0_dist = (5, 25),
    metabolic_rate_dist = (1, 4),
    vision_dist = (1, 6),
    max_age_dist = (60, 100),
    max_sugar = 4,
    seed = 42
)
    sugar_capacities = sugar_caps(dims, sugar_peaks, max_sugar, 6)
    sugar_values = deepcopy(sugar_capacities)
    space = GridSpace(dims)
    properties = Dict(
        :growth_rate => growth_rate,
        :N => N,
        :w0_dist => w0_dist,
        :metabolic_rate_dist => metabolic_rate_dist,
        :vision_dist => vision_dist,
        :max_age_dist => max_age_dist,
        :sugar_values => sugar_values,
        :sugar_capacities => sugar_capacities,
    )
    model = AgentBasedModel(
        SugarSeeker,
        space,
        scheduler = Schedulers.randomly,
        properties = properties,
        rng = MersenneTwister(seed)
    )
    for ag in 1:N
        add_agent_single!(
            model,
            rand(model.rng, vision_dist[1]:vision_dist[2]),
            rand(model.rng, metabolic_rate_dist[1]:metabolic_rate_dist[2]),
            0,
            rand(model.rng, max_age_dist[1]:max_age_dist[2]),
            rand(model.rng, w0_dist[1]:w0_dist[2]),
        )
    end
    return model
end

model = sugarscape()

# Let's plot the spatial distribution of sugar capacities in the Sugarscape.
using CairoMakie
CairoMakie.activate!() # hide

fig = Figure(resolution = (600, 600))
ax, hm = heatmap(fig[1,1], model.sugar_capacities; colormap=cgrad(:thermal))
Colorbar(fig[1, 2], hm, width = 20)
fig

# ## Defining stepping functions

# Now we define the stepping functions that handle the time evolution of the model

function model_step!(model)
    ## At each position, sugar grows back at a rate of $\alpha$ units
    ## per time-step up to the cell's capacity c.
    togrow = findall(
        x -> model.sugar_values[x] < model.sugar_capacities[x],
        1:length(positions(model)),
    )
    model.sugar_values[togrow] .+= model.growth_rate
end

function movement!(agent, model)
    newsite = agent.pos
    ## find all unoccupied position within vision
    neighbors = nearby_positions(agent.pos, model, agent.vision)
    empty = collect(empty_positions(model))
    if length(empty) > 0
        ## identify the one(s) with greatest amount of sugar
        available_sugar = (model.sugar_values[x,y] for (x, y) in empty)
        maxsugar = maximum(available_sugar)
        if maxsugar > 0
            sugary_sites_inds = findall(x -> x == maxsugar, collect(available_sugar))
            sugary_sites = empty[sugary_sites_inds]
            ## select the nearest one (randomly if more than one)
            for dia in 1:(agent.vision)
                np = nearby_positions(agent.pos, model, dia)
                suitable = intersect(np, sugary_sites)
                if length(suitable) > 0
                    newsite = rand(model.rng, suitable)
                    break
                end
            end
            ## move there and collect all the sugar in it
            newsite != agent.pos && move_agent!(agent, newsite, model)
        end
    end
    ## update wealth (collected - consumed)
    agent.wealth += (model.sugar_values[newsite...] - agent.metabolic_rate)
    model.sugar_values[newsite...] = 0
    ## age
    agent.age += 1
end

function replacement!(agent, model)
    ## If the agent's sugar wealth become zero or less, it dies
    if agent.wealth <= 0 || agent.age >= agent.max_age
        kill_agent!(agent, model)
        ## Whenever an agent dies, a young one is added to a random pos.
        ## New agent has random attributes
        add_agent_single!(
            model,
            rand(model.rng, model.vision_dist[1]:model.vision_dist[2]),
            rand(model.rng, model.metabolic_rate_dist[1]:model.metabolic_rate_dist[2]),
            0,
            rand(model.rng, model.max_age_dist[1]:model.max_age_dist[2]),
            rand(model.rng, model.w0_dist[1]:model.w0_dist[2]),
        )
    end
end

function agent_step!(agent, model)
    movement!(agent, model)
    replacement!(agent, model)
end

# ## Plotting & Animating

# We can plot the ABM and the sugar distribution side by side using [`abm_plot`](@ref)
# and standard Makie.jl commands like so
using InteractiveDynamics

model = sugarscape()
fig, abmstepper = abm_plot(model; resolution = (800, 600))
ax, hm = heatmap(fig[1,2], model.sugar_values; colormap=cgrad(:thermal), colorrange=(0,4))
ax.aspect = AxisAspect(1) # equal aspect ratio for heatmap
Colorbar(fig[1, 3], hm, width = 15, tellheight=false)
rowsize!(fig.layout, 1, ax.scene.px_area[].widths[2]) # Colorbar height = axis height
fig


# To animate them both however, we will use the approach Makie.jl suggests for animations,
# which is based on `Observables`. We start similarly with a call to `abm_plot`,
# but now make the plotted heatmap an obsrvable
fig, abmstepper = abm_plot(model; resolution = (800, 600))
obs_heat = Observable(model.sugar_values)
ax, hm = heatmap(fig[1,2], obs_heat; colormap=cgrad(:thermal), colorrange=(0,4))
ax.aspect = AxisAspect(1) # equal aspect ratio for heatmap
Colorbar(fig[1, 3], hm, width = 15, tellheight=false)
rowsize!(fig.layout, 1, ax.scene.px_area[].widths[2]) # Colorbar height = axis height

# and also add a title for good measure
s = Observable(0) # counter of current step, also observable
t = lift(x -> "Sugarscape, step = $x", s)
supertitle = Label(fig[0, :], t, textsize = 24, halign = :left)
fig

# We animate the evolution of both the ABM and the sugar distribution using the following
# simple loop involving the abm stepper
record(fig, "sugarvis.mp4"; framerate = 3) do io
    for j in 0:50 # = total number of frames
        recordframe!(io) # save current state
        ## This updates the abm plot:
        Agents.step!(abmstepper, model, agent_step!, model_step!, 1)
        ## This updates the heatmap:
        obs_heat[] = model.sugar_values
        ## This updates the title:
        s[] = s[] + 1
    end
end
