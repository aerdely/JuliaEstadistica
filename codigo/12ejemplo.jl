### Regresión Bernoulli
### Autor: Dr. Arturo Erdely
### Versión: 2024-11-17


## Cargar paquetes necesarios:

begin
    using Distributions, Plots # previamente instalados
    include("06EDA.jl")
end


## Simular muestra condicionada a partir de un modelo

begin
    θ18h = 0.68
    θ18m = 0.75
    θ55h = 0.85
    θ55m = 0.87
    b = zeros(4)
    b[2] = (log(1 - θ18h) - log(1 - θ55h))/37
    b[1] = -log(1 - θ18h) - 18b[2]
    b[4] = (log(1 - θ18m) - log(1 - θ55m))/37
    b[3] = -log(1 - θ18m) - 18b[4]
    gh(x) = 1 - exp(-b[1] - b[2]*x)
    gm(x) = 1 - exp(-b[3] - b[4]*x)
    g(x1,x2) = (1-x2)*gh(x1) + x2*gm(x1)
    edad = collect(18:55)
    scatter(edad, gm.(edad), xticks = 18:2:55, label = "mujeres", color = :pink, xlabel = "Edad", ylabel = "Probabilidad de cumplimiento", legend = :right)
    scatter!(edad, gh.(edad), label = "hombres", color = :cyan)
    hline!([0], lw = 0.1, color = :lightgray, label = "")
    hline!([1], lw = 0.1, color = :lightgray, label = "")
end

begin
    nsim = 3_000 # número de simulaciones 
    x1 = 17 .+ rand(Categorical(55-17), nsim) # edad simulada 
    x2 = rand(Bernoulli(0.5), nsim) # sexo simulado 
    y = zeros(Int, nsim) # inicializar vector de cumplimiento
    for i ∈ 1:nsim
        y[i] = rand(Bernoulli(g(x1[i], x2[i])), 1)[1]
    end
    println("Edad | Sexo | Cumplimiento")
    hcat(x1, x2, y)
end



# Estimación puntual de parámetros por mínimos cuadrados ordinarios

begin
    ϕ(x1,x2,β) = (1-x2)*(1-exp(-β[1]-β[2]*x1)) + x2*(1-exp(-β[3]-β[4]*x1))
    n = length(y)
    function h(β)
        if min(β[2], β[4]) ≤ 0
            return Inf
        end
        s = 0.0
        for i ∈ 1:n 
            s += (y[i] - ϕ(x1[i], x2[i], β))^2
        end
        return s 
    end
    est = EDA(h, [0,0.001,0,0.001], [2,0.1,2,0.1])
    βest = est.x
    plot!(edad, 1 .- exp.(-βest[3] .- βest[4].*edad), color = :violet, lw = 2, label = "mujeres estim")
    plot!(edad, 1 .- exp.(-βest[1] .- βest[2].*edad), color = :blue, lw = 2, label = "hombres estim")
end

begin
    println("  β    teórica    estimada")
    println(" -------------------------")
    bb = "β" .*  string.(collect(1:4))
    hcat(bb, b, βest)
end
