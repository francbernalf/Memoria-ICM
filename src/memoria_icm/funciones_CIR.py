import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import ive
from scipy.stats import ncx2
from scipy.stats import gaussian_kde

def filtrar_grupo_salinidad(df: pd.DataFrame, especie: str, tratamiento: str, salinidad: str):
    df_filtrado = df[
        (df["especie"] == especie) & 
        (df["tratamiento"] == tratamiento) & 
        (df["salinidad"] == salinidad)
    ]
    return df_filtrado

def filtrar_grupo_nivel_salinidad(df: pd.DataFrame, especie: str, tratamiento: str, nivel_salinidad: str):
    df_filtrado = df[
        (df["especie"] == especie) & 
        (df["tratamiento"] == tratamiento) & 
        (df["nivel_salinidad"] == nivel_salinidad)
    ]
    return df_filtrado

def simular_CIR(N, dt, alpha, r_infinito, sigma, r0):
    alpha = float(alpha)
    sigma = float(sigma)
    r_infinito = float(r_infinito)
    r0 = float(r0)

    if alpha > 0.0 and sigma > 0.0 and r_infinito >= 0.0 and r0 >= 0.0:
        r_sim = np.zeros(N + 1)
        r_sim[0] = r0
        for k in range(N):
            inc = (alpha * (r_infinito - r_sim[k]) * dt + 
                sigma * np.sqrt(max(r_sim[k], 0)) * np.random.normal(0, np.sqrt(dt)))
            r_sim[k + 1] = max(r_sim[k] + inc, 0)
        return r_sim
    else:
        print("Parámetros inválidos. Omitiendo simulación.")
        return

def extraer_trayectorias(df_filtrado, variable, tiempo="t"):
    trayectorias = []
    ids = []
    for id_larva, df_larva in df_filtrado.groupby("id"):
        x = df_larva[variable].to_numpy()
        t = df_larva[tiempo].to_numpy()
        if len(x) > 5 and np.all(np.diff(t) > 0):
            dt_real = float(np.mean(np.diff(t)))
            trayectorias.append((x, dt_real))
            ids.append(id_larva)
    return trayectorias, ids

def estimar_sigma_CIR(df, col_r, col_t='t', col_id='id'): #estimación de sigma mediante variación cuadrática
    df = df.sort_values(by=[col_id, col_t]).copy()

    def calcular_sigma(grupo):
        grupo = grupo[grupo[col_r] > 0.0]
        r = grupo[col_r].values
        t = grupo[col_t].values
        dr = np.diff(r)
        dt = np.diff(t)

        numerador = np.sum(dr**2)
        denominador = np.sum(r[:-1] * dt)
        sigma2_hat = numerador / denominador if denominador > 0 else np.nan
        sigma_hat = np.sqrt(sigma2_hat) if sigma2_hat >= 0 else np.nan

        grupo['sigma2_hat'] = sigma2_hat
        grupo['sigma_hat'] = sigma_hat
        return grupo

    df = df.groupby(col_id, group_keys=False).apply(calcular_sigma)
    return df

def estimar_alpha_mu_CIR(df, col_r, col_t = 't', col_id = 'id'): #formula del paper para estimar alpha y mu
    df = df.sort_values(by=[col_id, col_t]).copy()
    
    def calcular_alpha_mu(grupo):
        grupo = grupo[grupo[col_r] > 0.0]
        r = np.asarray(grupo[col_r])
        t = np.asarray(grupo[col_t])
        N = len(r)
        
        grupo['alpha_hat'] = np.nan
        grupo['mu_hat'] = np.nan
        
        r = np.maximum(r, 1e-10)
        dts = np.diff(t)
        dt_mean = np.mean(dts)

        r_i = r[:-1]
        r_ip1 = r[1:]

        sum_ri = np.sum(r_i)
        sum_rip1 = np.sum(r_ip1)
        sum_inv_ri = np.sum(1.0 / r_i)
        sum_rip1_div_ri = np.sum(r_ip1 / r_i)

        num_alpha = (
            (N - 1)**2 +
            sum_rip1 * sum_inv_ri -
            sum_ri * sum_inv_ri -
            (N - 1) * sum_rip1_div_ri
        )

        den_alpha = (
            (N - 1)**2 -
            sum_ri * sum_inv_ri
        ) * dt_mean

        alpha_hat = num_alpha / (den_alpha) if den_alpha != 0.0 else np.nan

        num_mu = (
            (N - 1) * sum_rip1 -
            sum_rip1_div_ri * sum_ri
        )

        den_mu = (
            (N - 1)**2 +
            sum_rip1 * sum_inv_ri -
            sum_ri * sum_inv_ri -
            (N - 1) * sum_rip1_div_ri
        )

        mu_hat = num_mu / den_mu if den_mu != 0.0 else np.nan
        
        grupo['alpha_hat'] = alpha_hat
        grupo['mu_hat'] = mu_hat
        
        return grupo
    
    df_salida = df.groupby(col_id, group_keys = False).apply(calcular_alpha_mu)
    return df_salida

def logverosimilitud_CIR_Bessel(params, data, dt, sigma):
    alpha, r_infinito = params
    
    penalizacion = 0.0
    
    if alpha <= 0 or r_infinito <= 0:
        penalizacion = 10e8

    data_f = data[1:]
    data_l = data[:-1]

    try:
        c = 2 * alpha / (sigma**2 * (1 - np.exp(-alpha * dt)))
        q = 2 * alpha * r_infinito / sigma**2 - 1
        u = c * np.exp(-alpha * dt) * data_l
        v = c * data_f

        z = 2 * np.sqrt(u * v)
        z = np.where(z < 1e-8, 1e-8, z)

        # ive en lugar de iv * exp(-z) por estabilidad numérica
        bf = ive(q, z)
        bf = np.where(bf <= 0, 1e-12, bf)

        lnL = -(len(data) - 1) * np.log(c) + np.sum(
            u + v - 0.5 * q * np.log(np.maximum(v / u, 1e-12)) - np.log(bf) - z
        )

        return lnL + penalizacion

    except Exception:
        return np.inf  # en caso de errores numéricos extremos
    
def logverosimilitud_CIR_ncx2(params, data, dt, sigma):
    alpha, mu = params

    if sigma <= 0 or alpha <= 0 or np.any(data <= 0):
        return np.inf

    data_l = data[:-1]
    data_f = data[1:]

    try:
        c = 2 * alpha / (sigma**2 * (1 - np.exp(-alpha * dt)))
        q = 2 * alpha * mu / sigma**2 - 1
        u = c * np.exp(-alpha * dt) * data_l
        s = 2 * c * data_f
        df = 2 * q + 2
        nc = 2 * u

        pdf_vals = ncx2.pdf(s, df, nc)
        pdf_vals = np.where(pdf_vals <= 1e-12, 1e-12, pdf_vals)

        logL = -np.sum(np.log(2 * c * pdf_vals))
        if np.isnan(logL) or np.isinf(logL):
            return np.inf
        return logL

    except Exception:
        return np.inf

def evaluar_estimadores_CIR_paper_una_trayectoria(simular_CIR, trayectoria, dt, alpha, r_infinito, sigma, variable,
                                                  especie, tratamiento, salinidad, M=1000):
    r_vals, dt_tray = trayectoria
    N_rep = len(r_vals) - 1
    r0 = r_vals[0] #acá el valor inicial es el primer término de la trayectoria simplemente

    sims = np.stack([simular_CIR(N_rep, dt, alpha, r_infinito, sigma, r0) for _ in range(M)])
    percentiles = [np.percentile(sims, q, axis=0) for q in (0, 5, 50, 95, 100)]
    p0, p5, p50, p95, p100 = percentiles
    times = np.arange(N_rep + 1) * dt

    # Gráfico
    plt.figure(figsize=(10, 6))
    plt.fill_between(times, p0, p100, color='lightgray', alpha=0.4, label='0–100% simuladas')
    plt.fill_between(times, p5, p95, color='lightblue', alpha=0.4, label='Banda 90% simulada')
    plt.plot(times, p50, 'b-', linewidth=2, label='Mediana simulada')
    plt.plot(times, p5, '--', color='red', lw=1.5, label='5º percentil')
    plt.plot(times, p95, '--', color='blue', lw=1.5, label='95º percentil')

    r_trunc = r_vals[:N_rep + 1]
    t_tray = np.arange(len(r_trunc)) * dt_tray
    plt.plot(t_tray, r_trunc,
             color='black',
             linewidth=1.5,
             alpha=0.9,
             label=f"Trayectoria real")

    plt.legend(loc='upper right')
    plt.xlabel("Tiempo [s]")
    plt.ylabel(variable)
    plt.title(f"Grupo {especie}-{tratamiento}-{salinidad.upper()}")
    plt.grid(True, alpha=0.3)
    plt.show()
    
# def evaluar_estimadores_CIR_paper(simular_CIR, trayectorias, dt, alpha, mu, sigma, variable,
#                                    df_grp, especie, tratamiento, salinidad, M=1000):

#     N_rep = min(len(r) for r, _ in trayectorias) - 1
#     r0 = 0

#     sims = np.stack([simular_CIR(N_rep, dt, alpha, mu, sigma, r0) for _ in range(M)])
#     percentiles = [np.percentile(sims, q, axis=0) for q in (0, 5, 50, 95, 100)]
#     p0, p5, p50, p95, p100 = percentiles
#     times = np.arange(N_rep + 1) * dt

#     # Mapeo colores por salinidad
#     sal_map = df_grp.groupby('id')['salinidad'].first().to_dict()
#     unique_sals = sorted(set(sal_map.values()))
#     cmap = plt.get_cmap('tab10')
#     color_map = {sal: cmap(i) for i, sal in enumerate(unique_sals)}

#     # Gráfico
#     plt.figure(figsize=(10, 6))
#     plt.fill_between(times, p0, p100, color='lightgray', alpha=0.4, label='0–100% simuladas')
#     plt.fill_between(times, p5, p95, color='lightblue', alpha=0.4, label='Banda 90% simulada')
#     plt.plot(times, p50, 'b-', linewidth=2, label='Mediana simulada')
#     plt.plot(times, p5, '--', color='red', lw=1.5, label='5º percentil')
#     plt.plot(times, p95, '--', color='blue', lw=1.5, label='95º percentil')

#     ids = list(sal_map.keys())
#     for (r_vals, dt_tray), larva_id in zip(trayectorias, ids):
#         r_trunc = r_vals[:N_rep + 1]
#         t_tray = np.arange(len(r_trunc)) * dt_tray
#         sal = sal_map[larva_id]
#         plt.plot(t_tray, r_trunc,
#                  color=color_map[sal],
#                  linewidth=0.8,
#                  alpha=0.7,
#                  label=f"{sal}")

#     handles, labels = plt.gca().get_legend_handles_labels()
#     by_label = dict(zip(labels, handles))
#     plt.legend(by_label.values(), by_label.keys(), loc='upper right')
#     plt.xlabel("Tiempo [s]")
#     plt.ylabel(variable)
#     plt.title(f"Grupo {especie}-{tratamiento}-{salinidad.upper()}")
#     plt.grid(True, alpha=0.3)
#     plt.show()
