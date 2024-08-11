#=

    Instala algunos paquetes externos necesarios para el curso de 
    Estadística II mediante un ciclo. También podrían instalarse
    uno por uno desde el modo paquete en la terminal mediante
    la instrucción `add` seguida del nombre del paquete.

=#

using Pkg # cargando el instalador de paquetes externos

begin
    paquete = ["QuadGK", "HCubature", "Optim",
               "SpecialFunctions", "LaTeXStrings", 
               "Distributions", "StatsBase",
               "Plots", "StatsPlots"
    ]
    for p in paquete
        println("*** Instalando paquete: ", p)
        Pkg.add(p)
    end
    println("*** Fin de la lista de paquetes.")
end
