import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib_inline.backend_inline import set_matplotlib_formats
import os
import numpy as np
from typing import Literal

paleta_colores = {"CC": "#0097b2", "LA": "#ffb039", "general": "#008fd6", "general2": "#009ead", "E": "#ff7f0e", "R": "#1f77b4"}

def aplicar_configuracion_global():
    sns.set(context='notebook', style='white', palette='viridis', rc={"figure.dpi": 100, 'savefig.dpi': 500})
    set_matplotlib_formats('retina')

def set_contexto(contexto: Literal['paper', 'notebook', 'talk', 'poster'] = 'notebook'):
    aplicar_configuracion_global()
    sns.set_context(contexto)

def set_estilo(estilo: Literal['white', 'dark', 'whitegrid', 'darkgrid', 'ticks'] = 'white'):
    aplicar_configuracion_global()
    sns.set_style(estilo)

def set_paleta(paleta='viridis'):
    aplicar_configuracion_global()
    sns.set_palette(paleta)

def reset_configuracion():
    sns.reset_defaults()
    aplicar_configuracion_global()

def guardar_figura(nombre, subcarpeta, formato='png', dpi=500):
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ruta_figuras = os.path.join(BASE_DIR, "resultados", "figuras")
    carpeta_destino = os.path.join(ruta_figuras, subcarpeta)
    if not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino)

    ruta = os.path.join(carpeta_destino, f"{nombre}.{formato}")
    plt.savefig(ruta, dpi=dpi, bbox_inches='tight')
    print(f"Figura guardada en: {ruta}")


def crear_subplots(n_subplots, orientacion='horizontal', figsize_base=(6, 4)):
    aplicar_configuracion_global()
    if n_subplots >= 4 or orientacion == 'grid':
        cols = int(np.ceil(np.sqrt(n_subplots)))
        rows = int(np.ceil(n_subplots / cols))
    elif orientacion == 'horizontal':
        rows, cols = 1, n_subplots
    elif orientacion == 'vertical':
        rows, cols = n_subplots, 1
    else:
        raise ValueError("Orientación no válida. Usa 'horizontal', 'vertical' o 'grid'.")

    fig, axs = plt.subplots(rows, cols, figsize=(figsize_base[0]*cols, figsize_base[1]*rows))
    axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]
    return fig, axs[:n_subplots]

def plot(x=None, y=None, kind='plot', label=None, title=None, xlabel=None, ylabel=None,
         color=None, ax=None, auto_labels=True, **kwargs):
    if y is None:
        raise ValueError("Se requiere al menos el argumento y")
    if x is None:
        x = np.arange(len(y)) if hasattr(y, '__len__') else np.array([0])
    if ax is None:
        fig, ax = plt.subplots()
    
    #tipo de gráfico, por ahora solo dejé dos
    if kind == 'plot':
        ax.plot(x, y, label=label, color=color, **kwargs)
    elif kind == 'line':
        sns.lineplot(x=x, y=y, ax=ax, label=label, color=color, **kwargs)
    else:
        raise ValueError("Tipo de gráfico no soportado. Usa 'plot', 'line' o modifica graficos.py.")
    
    if label: 
        ax.legend()
    if title: 
        ax.set_title(title)
    if xlabel: 
        ax.set_xlabel(xlabel)
    if ylabel: 
        ax.set_ylabel(ylabel)
    
    return ax