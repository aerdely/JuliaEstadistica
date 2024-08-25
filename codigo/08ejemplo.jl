### Ejemplo: función de densidad triangular
### Autor: Dr. Arturo Erdely
### Versión: 2024-08-25


## Cargar paquetes y código externo necesarios:
begin
    using Distributions, Plots
    include("06EDA.jl") 
end

@doc TriangularDist


## Simulamos muestra aleatoria

a, b, c = -1.0, 3.0, 1.0
X = TriangularDist(a,b,c)
n = 1_000 # tamaño de muestra
muestra = rand(X, n)
# histograma:
begin
    histogram(muestra, normalize = true, color = :yellow, label = "muestral")
    xaxis!("x"); yaxis!("densidad")
end
x = range(a, b, length = 300)
plot!(x, pdf(X, x), lw = 2, color = :red, label = "teórica")


## Estimación puntual inicial heurística

a0, b0 = extrema(muestra)
c0 = 3*mean(muestra) - (a0+b0)


## Estimación puntual por máxima verosimilitud

function mlogV(θ) # menos log-verosimilitud
    if θ[1] < θ[3] < θ[2]
        return -sum(log.(pdf(TriangularDist(θ[1], θ[2], θ[3]), muestra)))
    else
        return Inf
    end
end

emv = EDA(mlogV, [2*a0-c0,c0,a0], [c0,2*b0-c0,b0])

emvX = TriangularDist(emv.x[1], emv.x[2], emv.x[3])
xx = range(emv.x[1], emv.x[2], length = 300)
plot!(xx, pdf(emvX, xx), lw = 2, color = :green, label = "estimada")


### Prueba de hipótesis
### Ho: c = (a+b)/2  ≡  Ho: simétrica

## Simulamos una muestra 

n = 100 # tamaño de muestra 
muestra = rand(TriangularDist(-1,3,2.0), n)
# histograma:
begin
    histogram(muestra, normalize = true, color = :yellow, label = "muestral")
    xaxis!("x"); yaxis!("densidad")
end
# estimación puntual por máxima verosimilitud
begin
    a0, b0 = extrema(muestra)
    c0 = 3*mean(muestra) - (a0+b0)
    emv = EDA(mlogV, [2*a0-b0,a0,a0], [b0,2*b0-a0,b0], maxiter = 3_000)
end
# valor observado del estadístico de la prueba:
Tobs = abs(emv.x[3] - (emv.x[2] + emv.x[1])/2)


## Menos log-verosimilitud bajo Ho a minimizar

function mlogV2(θ) 
    if θ[1] < θ[2]
        return -sum(log.(pdf(TriangularDist(θ[1], θ[2]), muestra)))
    else
        return Inf
    end
end

emv2 = EDA(mlogV2, [2*a0-b0,a0], [b0,2*b0-a0])

XHo = TriangularDist(emv2.x[1], emv2.x[2])


## Simular el estadístico de prueba bajo Ho

function simulaTbajoHo(m)
    simTHo = zeros(m)
    éxito = 0
    while éxito < m
        global muestra = rand(XHo, n)
        a0, b0 = extrema(muestra)
        c0 = 3*mean(muestra) - (a0+b0)
        emv = EDA(mlogV, [2*a0-b0,a0,a0], [b0,2*b0-a0,b0], maxiter = 3_000)
        if emv.iter < 3_000
            éxito += 1
            @show éxito
            simTHo[éxito] = abs(emv.x[3] - (emv.x[1] + emv.x[2])/2)
        end
    end
    return simTHo
end

# primero estimar tiempo con pocas simulaciones
@time simTHo = simulaTbajoHo(10)

# Simular estadístico de la prueba bajo Ho
@time simTHo = simulaTbajoHo(1_000)

histogram(simTHo, label = "simulada", xlabel = "estadístico", ylabel = "densidad", color = :cyan)
vline!([Tobs], lw = 3, color = :red, label = "observado")

# p-value de rechazar Ho
mean(simTHo .≥ Tobs)
