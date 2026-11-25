# Lévy flights: trabajo futuro / future work

[Español](#español) · [English](#english)

## Español

Este cuaderno prueba un modelo distinto a los de la memoria y no forma parte de los resultados reportados, aunque ofrece ajustes prometedores. Se incluyen comentarios sobre lo observado como punto de partida para un potencial trabajo futuro.

### Vuelos de Lévy

- Los ángulos entre pasos de las trayectorias 2D no siempre están decorrelacionados y no parecen tener distribución uniforme.
- Las longitudes entre puntos de la trayectoria parecen seguir leyes de potencia, aunque en la trayectoria en espiral no tanto. Al comparar la ley de potencia con distribuciones similares (exponencial, lognormal, Weibull), ninguna es preferida frente a la ley de potencia, pero la ley de potencia tampoco es preferida frente a ellas, al menos con el nivel de confianza clásico del 95 %.
- Al ajustar la ley de potencia, el exponente resulta muy alto para lo que se espera de un vuelo de Lévy (0 < α < 2), y el algoritmo de simulación está diseñado solo para ese rango. Sin embargo, al ajustar una ley de potencia truncada y usar los ángulos observados, las simulaciones se ven razonables.
- Con x_min ≈ x̂_min / 2, la escala de las trayectorias simuladas se parece aún más a la observada, pero eso ya es un ajuste subjetivo.

Queda abierta la pregunta de si las estimaciones de x_min y α muestran diferencias según la salinidad.

### Calibración del modelo CIR

Otra opción es mirar el cociente μ/α en lugar de cada parámetro por separado. Así no se descartan estimaciones solo porque α o μ sean muy pequeños o muy grandes, y los valores atípicos se pueden identificar según el cociente. Este cociente tiene interpretación, porque es un factor que aparece en la varianza de largo plazo del CIR: puede que μ̂ y α̂ sean atípicos por separado, pero no en conjunto.

### Para ejecutar el cuaderno

Requiere el paquete `powerlaw` (`pip install powerlaw`) y un archivo `df.csv` en esta carpeta con las columnas `id`, `t`, `x`, `y` y `salinidad`, que se puede exportar desde la tabla `datos_trackeo` de la base de datos.

## English

This notebook tests a model different from those in the thesis and is not part of the reported results, although it yields promising fits. Comments on the observations are included as a starting point for potential future work.

### Lévy flights

- Turning angles between steps of the 2D trajectories are not always uncorrelated and do not appear to be uniformly distributed.
- Step lengths along the trajectories appear to follow power laws, although less so for the spiral-shaped trajectory. When comparing the power law against similar distributions (exponential, lognormal, Weibull), none of them is preferred over the power law, but the power law is not preferred over them either, at least at the usual 95% confidence level.
- The fitted power-law exponent is too high for a Lévy flight (0 < α < 2), and the simulation algorithm is designed only for that range. However, fitting a truncated power law and using the observed angles produces reasonable-looking simulations.
- Using x_min ≈ x̂_min / 2, the scale of the simulated trajectories matches the observed ones even better, but this is already a subjective adjustment.

An open question is whether the estimates of x_min and α differ across salinity levels.

### CIR model calibration

Another option is to look at the ratio μ/α instead of each parameter separately. This avoids discarding estimates just because α or μ alone is very small or very large, and outliers can be identified based on the ratio. The ratio is interpretable, since it appears as a factor in the long-run variance of the CIR process: μ̂ and α̂ may each look atypical on their own but not jointly.

### Running the notebook

It requires the `powerlaw` package (`pip install powerlaw`) and a `df.csv` file in this folder with columns `id`, `t`, `x`, `y` and `salinidad`, which can be exported from the `datos_trackeo` table of the database.
