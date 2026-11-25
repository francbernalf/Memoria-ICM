import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde
from math import exp

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

def simular_OU(N, dt, alpha, omega_infinito, sigma, omega0):
    alpha = float(alpha)
    sigma = float(sigma)
    omega_infinito = float(omega_infinito) 
    omega0 = float(omega0)
    
    if alpha > 0.0 and sigma > 0.0:
        omega_sim = np.zeros(N + 1)
        omega_sim[0] = omega0
        for k in range(N):
            inc = alpha * (omega_infinito - omega_sim[k]) * dt + sigma * np.random.normal(0, np.sqrt(dt))
            omega_sim[k + 1] = omega_sim[k] + inc
        return omega_sim
    else:
        print("Parámetros inválidos. Omitiendo simulación")
        return

def estimar_sigma_OU(df, col_x, col_t='t', col_id='id'):
    df = df.sort_values(by=[col_id, col_t]).copy()

    def calcular_sigma(grupo):
        x = grupo[col_x].values
        t = grupo[col_t].values
        dx = np.diff(x)
        dt = np.diff(t)

        numerador = np.sum(dx**2)
        denominador = np.sum(dt)
        sigma2_hat = numerador / denominador if denominador > 0 else np.nan
        sigma_hat = np.sqrt(sigma2_hat) if sigma2_hat >= 0 else np.nan

        grupo['sigma2_hat'] = sigma2_hat
        grupo['sigma_hat'] = sigma_hat
        return grupo

    df = df.groupby(col_id, group_keys=False).apply(calcular_sigma)
    return df

def estimar_alpha_mu_OU(df, col_x, col_t='t', col_id='id'):
    df = df.sort_values(by=[col_id, col_t]).copy()

    def calcular_alpha_mu(grupo):
        x = grupo[col_x].values
        t = grupo[col_t].values

        grupo['alpha_hat'] = np.nan
        grupo['mu_hat'] = np.nan

        if len(x) <= 1:
            return grupo

        X_i = x[:-1]
        X_i_plus_1 = x[1:]

        dt = np.mean(np.diff(t))
        if dt <= 0:
            return grupo

        A = np.vstack([X_i, np.ones(len(X_i))]).T

        try:
            phi_c_est, _, _, _ = np.linalg.lstsq(A, X_i_plus_1, rcond=None)
            phi_est, c_est = phi_c_est
        except np.linalg.LinAlgError:
            return grupo

        #el modelo OU requiere phi = e^(-alpha*dt), lo que implica phi > 0. además, para la estabilidad, phi < 1
        
        if phi_est <= 0 or phi_est >= 1:
            return grupo
        # Deriva los parámetros del proceso OU a partir de los coeficientes
        # Se renombra alpha_est a theta_est en la teoría, pero mantenemos
        # 'alpha' por consistencia con tu código.
        alpha_hat = -np.log(phi_est) / dt
        
        if 1 - phi_est == 0:
            return grupo
        
        mu_hat = c_est / (1 - phi_est)
        

        # Asignamos los resultados al DataFrame del grupo
        # Manejamos el caso de alpha_hat negativo que puede ocurrir con OLS
        if alpha_hat > 0:
            grupo['alpha_hat'] = alpha_hat
            grupo['mu_hat'] = mu_hat
            
        return grupo

    df_salida = df.groupby(col_id, group_keys=False).apply(calcular_alpha_mu)
    return df_salida

def logverosimilitud_OU(params, data, dt, sigma):
    alpha, omega_infinito = params
    
    penalizacion = 0.0
    
    if alpha <= 0:
        penalizacion = 10e8
        
    loglik = 0.0
    e_t = exp(-alpha * dt)
    var_coef = sigma**2 / (2 * alpha)
    for i in range(1, len(data)):
        Xi_1 = data[i - 1]
        media = omega_infinito + (Xi_1 - omega_infinito) * e_t
        varianza = var_coef * (1 - e_t**2)
        if varianza <= 0:
            return np.inf
        diff = data[i] - media
        loglik += -0.5 * (np.log(2 * np.pi * varianza) + diff**2 / varianza)
    return -loglik + penalizacion

def evaluar_estimadores_OU_paper_una_trayectoria(simular_OU, trayectoria, dt, alpha, omega_infinito, sigma, variable,
                                                  especie, tratamiento, salinidad, M=1000):
    omega_vals, dt_tray = trayectoria
    N_rep = len(omega_vals) - 1
    omega0 = omega_vals[0] #acá el valor inicial es el primer término de la trayectoria simplemente

    sims = np.stack([simular_OU(N_rep, dt, alpha, omega_infinito, sigma, omega0) for _ in range(M)])
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

    r_trunc = omega_vals[:N_rep + 1]
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
    
# def evaluar_estimadores_OU_paper(simular_OU, trayectorias, dt, alpha, mu, sigma, variable,
#                                    df_grp, especie, tratamiento, salinidad, M=1000):

#     N_rep = min(len(theta) for theta, _ in trayectorias) - 1
#     theta0_vals = np.array([theta[0] for theta, _ in trayectorias])

#     kde = gaussian_kde(theta0_vals)
#     x_grid = np.linspace(min(theta0_vals), max(theta0_vals), N_rep)
#     theta0 = x_grid[np.argmax(kde(x_grid))] #extraemos el valor mas "probable" del kde

#     sims = np.stack([simular_OU(N_rep, dt, alpha, mu, sigma, theta0) for _ in range(M)])
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
