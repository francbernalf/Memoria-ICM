import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import scikit_posthocs as sp
from scipy.stats import bartlett, kruskal, levene, shapiro
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def testear_supuestos(df, variable, factor, alpha=0.05):
    resultados = []
    normalidad_ok = True
    homocedasticidad_ok = True
    try:
        grupos = [grupo[variable].dropna() for _, grupo in df.groupby(factor) if not grupo[variable].dropna().empty]

        # ─ Normalidad por grupo ─
        for nivel, grupo in df.groupby(factor):
            datos = grupo[variable].dropna()
            n = len(datos)
            if n < 3:
                # CORREGIDO: sin al menos 3 datos no se puede comprobar normalidad -> no se asume
                normalidad_ok = False
                resultados.append({"variable": variable, "factor": factor, "grupo": nivel, "test": "Shapiro-Wilk",
                                   "p_value": np.nan, "cumple_supuesto": False, "n": n})
                continue
            stat, p = shapiro(datos)
            cumple = p > alpha
            if not cumple:
                normalidad_ok = False
            resultados.append({"variable": variable, "factor": factor, "grupo": nivel, "test": "Shapiro-Wilk",
                               "p_value": p, "cumple_supuesto": cumple, "n": n})

        # ─ Homocedasticidad ─
        if len(grupos) >= 2:
            n_total = sum(len(g) for g in grupos)
            if normalidad_ok:
                stat_test, p_test = bartlett(*grupos)
                test_nombre = "Bartlett"
            else:
                stat_test, p_test = levene(*grupos)
                test_nombre = "Levene"
            cumple = p_test > alpha
            if not cumple:
                homocedasticidad_ok = False
            resultados.append({"variable": variable, "factor": factor, "grupo": "todos", "test": test_nombre,
                               "p_value": p_test, "cumple_supuesto": cumple, "n": n_total})

        if normalidad_ok and homocedasticidad_ok:
            test_sugerido = "ANOVA"
        elif normalidad_ok:
            test_sugerido = "Welch's ANOVA"
        else:
            test_sugerido = "Kruskal-Wallis"
        for row in resultados:
            row["test_sugerido"] = test_sugerido

    except Exception as e:
        print(f"⚠️ Error con variable {variable}: {e}")

    return pd.DataFrame(resultados)


def elegir_tests(df, variables, factor, alpha=0.05):
    """CORREGIDO (nueva): {test: [variables]} con los supuestos evaluados sobre ESTE df.
    Llamarla con el subconjunto que se va a comparar (p. ej. un grupo y un sitio), no con la base completa."""
    listas = {}
    for var in variables:
        sup = testear_supuestos(df, var, factor, alpha)
        if sup.empty:
            continue
        listas.setdefault(sup["test_sugerido"].iloc[0], []).append(var)
    return listas


def calcular_eta_squared(anova_table, factor):
    ss_total = anova_table['sum_sq'].sum()
    ss_factor = anova_table.loc[factor, 'sum_sq']
    return ss_factor / ss_total

def calcular_omega_squared(anova_table, factor):
    ss_total = anova_table['sum_sq'].sum()
    ss_factor = anova_table.loc[factor, 'sum_sq']
    df_factor = anova_table.loc[factor, 'df']
    ms_error = anova_table.loc['Residual', 'sum_sq'] / anova_table.loc['Residual', 'df']
    omega_sq = (ss_factor - df_factor * ms_error) / (ss_total + ms_error)
    return max(0, omega_sq)

# CORREGIDO: nombres. Tomczak y Tomczak (2014): épsilon² = H/(n-1); eta²_H = (H-k+1)/(n-k), corregida por sesgo.
def calcular_epsilon_squared_kruskal(H, n):
    return H / (n - 1)

def calcular_eta2_H_kruskal(H, n, k):
    return (H - k + 1) / (n - k) if n > k else 0

def analizar_kruskal(df, variables, factor):
    resultados = []
    for var in variables:
        grupos = [g[var].dropna() for _, g in df.groupby(factor) if not g[var].dropna().empty]
        H_stat, p_kw = kruskal(*grupos)
        n = sum(len(g) for g in grupos)
        k = len(grupos)
        resultados.append({
            'variable': var,
            'test': "Kruskal-Wallis",
            'p_value': p_kw,
            'efecto': calcular_epsilon_squared_kruskal(H_stat, n),           # épsilon² (antes rotulado η²)
            'efecto_corregido': calcular_eta2_H_kruskal(H_stat, n, k),       # eta²_H (antes rotulado ω²/ε²)
            'medidas': "ε² / η²_H",
        })
    return resultados

def analizar_welch(df, variables, factor):
    # CORREGIDO: ANOVA de Welch propiamente tal. anova_lm(..., robust='hc3') era un ANOVA con errores robustos, no Welch.
    resultados = []
    for var in variables:
        data = df[[var, factor]].dropna()
        w = pg.welch_anova(data=data, dv=var, between=factor)
        F, k, N = w.loc[0, 'F'], data[factor].nunique(), len(data)
        omega_sq = max(0.0, (k - 1) * (F - 1) / ((k - 1) * (F - 1) + N))   # ω² estimado a partir del F de Welch
        resultados.append({
            'variable': var,
            'test': "Welch's ANOVA",
            'p_value': w.loc[0, 'p-unc'],
            'efecto': w.loc[0, 'np2'],                 # η²
            'efecto_corregido': omega_sq,              # ω²
            'medidas': "η² / ω²",
        })
    return resultados

def analizar_anova(df, variables, factor):
    resultados = []
    for var in variables:
        modelo = smf.ols(f'{var} ~ C({factor})', data=df).fit()
        tabla = sm.stats.anova_lm(modelo, typ=2)
        resultados.append({
            'variable': var,
            'test': "ANOVA",
            'p_value': tabla.loc[f'C({factor})', 'PR(>F)'],
            'efecto': calcular_eta_squared(tabla, factor=f'C({factor})'),
            'efecto_corregido': calcular_omega_squared(tabla, factor=f'C({factor})'),
            'medidas': "η² / ω²",
        })
    return resultados

def analizar_grupo(df, listas_por_test, factor):
    resultados = []
    if 'Kruskal-Wallis' in listas_por_test:
        resultados += analizar_kruskal(df, listas_por_test['Kruskal-Wallis'], factor)
    if "Welch's ANOVA" in listas_por_test:
        resultados += analizar_welch(df, listas_por_test["Welch's ANOVA"], factor)
    if 'ANOVA' in listas_por_test:
        resultados += analizar_anova(df, listas_por_test['ANOVA'], factor)
    return pd.DataFrame(resultados).sort_values(by='p_value', ascending=True)

# Uso correcto (por grupo):
#   for grupo, d in resumen.groupby("grupo"):
#       res = analizar_grupo(d, elegir_tests(d, variables, "nivel_salinidad"), "nivel_salinidad")

def posthoc_anova_tukey(df, variables, factor, alpha=0.05):
    resultados = []
    for var in variables:
        data = df[[var, factor]].dropna()
        tukey = pairwise_tukeyhsd(endog=data[var], groups=data[factor], alpha=alpha)
        df_resultado = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
        df_resultado["variable"] = var
        resultados.append(df_resultado)
    return pd.concat(resultados, ignore_index=True)

def posthoc_welch_gameshowell(df, variables, factor, alpha=0.05):
    resultados = []
    for var in variables:
        gh = pg.pairwise_gameshowell(dv=var, between=factor, data=df[[var, factor]].dropna())
        gh["variable"] = var
        resultados.append(gh)
    return pd.concat(resultados, ignore_index=True)

def posthoc_kruskal_dunn(df, variables, factor, alpha=0.05):
    resultados = []
    for var in variables:
        data = df[[var, factor]].dropna()
        dunn = sp.posthoc_dunn(data, val_col=var, group_col=factor, p_adjust='holm')
        # CORREGIDO: cada par una sola vez (triángulo superior), sin la diagonal
        mask = np.triu(np.ones(dunn.shape, dtype=bool), k=1)
        dunn = dunn.where(mask).stack().reset_index()
        dunn.columns = ['grupo1', 'grupo2', 'p-valor']
        dunn["variable"] = var
        resultados.append(dunn)
    return pd.concat(resultados, ignore_index=True)

def plot_corrs_por_grupo(df, vars_list, col_group):
    corrs = {}
    for g, sub in df.groupby(col_group, observed=True):
        corrs[g] = sub[vars_list].corr()
    fig, axes = plt.subplots(nrows=len(corrs), ncols=1, figsize=(6, 4 * len(corrs)))
    if len(corrs) == 1:
        axes = [axes]
    for ax, (g, corrmat) in zip(axes, corrs.items()):
        sns.heatmap(corrmat, annot=True, fmt=".2f", cmap="vlag", ax=ax, cbar=(ax is axes[-1]))
        ax.set_title(f"Correlaciones @ {col_group} = {g}")
    plt.tight_layout()
    plt.show()
