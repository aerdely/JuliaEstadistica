### Ejemplo de inferencia bayesiana
### Autor: Dr. Arturo Erdely
### Versión: 2024-10-13


# Cargar paquetes (previamente instalados)
using Distributions, Plots, LaTeXStrings


# Simulamos una muestra aleatoria
begin
    r, θ = 17, 0.35 # parámetros
    X = Binomial(r, θ) # variable aleatoria Binomial
    n = 100 # tamaño de muestra a simular
    muestra = rand(X, n)
    println("Muestra aleatoria observada de Binomial(r=$r,θ=$θ) de tamaño n=$n\n")
    println(muestra)
end;


# Distribución a posteriori con a priori θ ~ Uniforme(0,1)
# y estimación puntual a posteriori de θ
begin
    α, β = 1, 1
    s = sum(muestra)
    post = Beta(α + s, β + n*r - s)
    t = range(0, 1, length = 1_000)
    plot(t, pdf(post, t), lw = 3, legend = false)
    xaxis!(L"\theta"); yaxis!(L"p(\theta\,|\mathbf{x})")
end
begin
    t = range(0.25, 0.5, length = 1_000)
    plot(t, pdf(post, t), lw = 3, label = "")
    xaxis!(L"\theta"); yaxis!(L"p(\theta\,|\mathbf{x})")
    vline!([θ], label = "θ teórica", lw = 3)
    estimθ = round(mean(post), digits = 3)
    vline!([estimθ], label = "estimación puntual = $estimθ", lw = 2)
end

# Estimación por intervalo de probabilidad 95% para θ
γ = 0.95
a, b = quantile(post, [(1-γ)/2, (1+γ)/2]) # extremos del intervalo
cdf(post, b) - cdf(post, a) # verificando que encierran probabilidad γ
begin # agregando intervalo a la gráfica
    aa = round(a, digits = 3)
    bb = round(b, digits = 3)
    plot!([a, b], [0, 0], lw = 5, color = :red, 
          label = "intervalo $(100*γ)% = $([aa, bb])"
    )
end

# Probabilidad de la hipótesis H: θ > 1/3
begin
    probH = round(1 - cdf(post, 1/3), digits = 3)
    vline!([1/3], label = "P(θ>1/3) = $probH", color = :black)
end
