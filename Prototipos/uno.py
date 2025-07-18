import json
import random
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from tabulate import tabulate

# -----------------------------
# Carga de datos --> listo
# -----------------------------
def load_positions_and_parties(filename):
    """
    Carga las posiciones (x, y) y los partidos de los votos desde un archivo JSON.
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    votes = data['rollcalls'][0]['votes']
    coordenadas = np.array([[v['x'], v['y']] for v in votes])
    partidos = np.array(['Demócrata' if v.get('party_code',0)==100 else 'Republicano' if v.get('party_code',0)==200 else 'Otro' for v in votes])
    return coordenadas, partidos

# -----------------------------
# Utilidades y fitness --> listo
# -----------------------------
def fitness(coalition, positions, fitness_cache=None):
    """
    Calcula el fitness de una coalición: suma de distancias entre todos los miembros.
    Usa caché para acelerar cálculos repetidos.
    """
    if fitness_cache is not None:
        key = tuple(coalition)
        if key in fitness_cache:
            return fitness_cache[key]
    indices = np.where(coalition == 1)[0]
    if len(indices) < 2:
        val = float('inf')
    else:
        puntos = positions[indices]
        val = np.sum(pdist(puntos, metric='euclidean')) # Distancia Euclidiana con pdist
    if fitness_cache is not None:
        fitness_cache[key] = val
    return val

# -----------------------------
# Operadores genéticos
# -----------------------------
def initialize_population(pop_size, n, quorum):
    """
    Inicializa la población de coaliciones aleatorias, cada una con tamaño igual al quórum.
    """
    population = []
    for _ in range(pop_size):
        coalition = np.zeros(n, dtype=int)
        indices = random.sample(range(n), quorum)
        coalition[indices] = 1
        population.append(coalition)
    return population

def tournament_selection(pop, scores, k=None):
    """
    Selección por torneo: elige el mejor de k individuos aleatorios.
    """
    competitors = random.sample(list(zip(pop, scores)), k)
    return min(competitors, key=lambda x: x[1])[0]

def crossover(parent1, parent2, quorum):
    """
    Cruza dos padres para generar un hijo, corrigiendo el tamaño de la coalición.
    """
    point = random.randint(1, len(parent1) - 1)
    child = np.concatenate((parent1[:point], parent2[point:]))
    return correct_coalition(child, quorum)

def mutate(coalition, n, mutation_rate=0.17):
    """
    Realiza una mutación intercambiando un miembro dentro/fuera de la coalición con cierta probabilidad.
    """
    coalition = coalition.copy()
    if random.random() < mutation_rate:
        indices_1 = np.where(coalition == 1)[0]
        indices_0 = np.where(coalition == 0)[0]
        if len(indices_1) > 0 and len(indices_0) > 0:
            idx_1 = random.choice(indices_1)
            idx_0 = random.choice(indices_0)
            coalition[idx_1], coalition[idx_0] = 0, 1
    return coalition

def correct_coalition(coalition, quorum):
    """
    Corrige la coalición para que tenga exactamente el tamaño del quórum.
    """
    coalition = coalition.copy()
    current = np.sum(coalition)
    if current > quorum:
        indices_1 = np.where(coalition == 1)[0]
        to_turn_off = random.sample(list(indices_1), current - quorum)
        coalition[to_turn_off] = 0
    elif current < quorum:
        indices_0 = np.where(coalition == 0)[0]
        to_turn_on = random.sample(list(indices_0), quorum - current)
        coalition[to_turn_on] = 1
    return coalition

# -----------------------------
# Algoritmo genético principal
# -----------------------------
def genetic_mwc(positions, quorum, pop_size=38, gens=100, mutation_rate=0.17, selection_prob=0.141):
    """
    Algoritmo genético para encontrar la coalición ganadora mínima (MWC).
    """
    n = positions.shape[0] # Total de congresistas
    population = initialize_population(pop_size, n, quorum)

    # Variables de seguimiendo a proximos mejores resultados
    best = None
    best_score = float('inf')
    k = max(2, int(selection_prob * pop_size))
    fitness_cache = {}

    for gen in range(gens):
        scores = [fitness(ind, positions, fitness_cache) for ind in population]
        elite_idx = np.argmin(scores)
        elite = population[elite_idx]
        elite_score = scores[elite_idx]
        if elite_score < best_score:
            best_score, best = elite_score, elite.copy()
        new_pop = [elite.copy()]
        while len(new_pop) < pop_size:
            p1 = tournament_selection(population, scores, k=k)
            p2 = tournament_selection(population, scores, k=k)
            child = crossover(p1, p2, quorum)
            child = mutate(child, n, mutation_rate=mutation_rate)
            new_pop.append(child)
        population = new_pop
        print(f"Generación {gen+1}/{gens} - Mejor Fitness: {elite_score:.4f}")
    return np.where(best == 1)[0], best_score

# -----------------------------
# Visualización
# -----------------------------
def compute_hull_vertices(positions, coalition):
    """
    Calcula los vértices de la envolvente convexa de la coalición.
    """
    coalition = [int(i) for i in coalition]
    pts = positions[coalition]
    hull = ConvexHull(pts)
    return [coalition[i] for i in hull.vertices], hull

def plot_solution(positions, coalition, hull, partidos=None, pertenece=None):
    """
    Grafica la solución encontrada y su envolvente convexa, diferenciando partidos y pertenencia.
    """
    plt.figure(figsize=(8,8))
    ax = plt.gca()
    ax.set_facecolor('#bdbdbd')
    plt.grid(True, color='white', linewidth=1.2)
    if partidos is None:
        partidos = np.array(['Otro']*positions.shape[0])
    if pertenece is None:
        pertenece = np.zeros(positions.shape[0], dtype=bool)
        pertenece[coalition] = True
    partido_color = {'Demócrata':'blue', 'Republicano':'red', 'Otro':'gray'}
    partido_label = {'Demócrata':'Demócrata', 'Republicano':'Republicano', 'Otro':'Otro'}
    marker_dict = {False:'x', True:'o'}
    marker_label = {False:'No pertenece', True:'Pertenece'}
    for partido in np.unique(partidos):
        for pert in [False, True]:
            idx = np.where((partidos==partido) & (pertenece==pert))[0]
            if len(idx) > 0:
                plt.scatter(
                    positions[idx,0], positions[idx,1],
                    c=partido_color.get(partido, 'gray'),
                    marker=marker_dict[pert],
                    label=f"{partido_label.get(partido, partido)}{' - ' + marker_label[pert] if partido in partido_color else ''}",
                    edgecolor='black' if pert else None,
                    s=50 if pert else 30,
                    alpha=1 if pert else 0.7
                )
    hull_pts = positions[coalition][hull.vertices]
    hull_loop = np.append(hull_pts, hull_pts[0].reshape(1,2), axis=0)
    plt.fill(hull_loop[:,0], hull_loop[:,1], color='purple', alpha=0.2, zorder=2)
    plt.plot(hull_loop[:,0], hull_loop[:,1], color='purple', linewidth=2, zorder=3)
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Demócrata', markerfacecolor='blue', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Republicano', markerfacecolor='red', markersize=8),
        Line2D([0], [0], marker='x', color='gray', label='No pertenece', markersize=8, markeredgewidth=2),
        Line2D([0], [0], marker='o', color='k', label='Pertenece', markerfacecolor='k', markersize=8)
    ]
    partido_patch = mpatches.Patch(color='none', label='Partido')
    cgm_patch = mpatches.Patch(color='none', label='CGM')
    plt.legend(
        handles=[partido_patch, legend_elements[0], legend_elements[1], cgm_patch, legend_elements[2], legend_elements[3]],
        loc='best', frameon=True
    )
    plt.xlabel('Dimensión 1')
    plt.ylabel('Dimensión 2')
    plt.title('Solución MWC y su Envolvente Convexa')
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.tight_layout()
    plt.show()

# -----------------------------
# Main
# -----------------------------
if __name__ == '__main__':
    # Parámetros principales
    QUORUM = 216
    POS_FILE = 'votes.json'
    POP_SIZE = 38
    MUTATION_RATE = 0.1700019
    SELECTION_PROB = 0.141

    VALOR_ESPERADO = 9686.93831

    # Carga de datos de posiciones y partidos
    coordenadas, partidos = load_positions_and_parties(POS_FILE)
    n_diputados = coordenadas.shape[0]

    # Ejecución del algoritmo genético
    t0 = time.time()
    gens = 10000
    best_coalition, best_fit = genetic_mwc(
        coordenadas, QUORUM,
        pop_size=POP_SIZE,
        gens=gens,
        mutation_rate=MUTATION_RATE,
        selection_prob=SELECTION_PROB,
    )
    t1 = time.time()

    # Cálculo de métricas de desempeño
    fitness_arr = np.array([best_fit])
    iter_arr = np.array([gens])
    time_arr = np.array([t1 - t0])

    precision_arr = 100 * (VALOR_ESPERADO / fitness_arr)
    promedio_precision = np.mean(precision_arr)
    std_precision = np.std(precision_arr)
    promedio_fitness = np.mean(fitness_arr)
    std_fitness = np.std(fitness_arr)
    promedio_iter = np.mean(iter_arr)
    std_iter = np.std(iter_arr)
    promedio_tiempo = np.mean(time_arr)
    std_tiempo = np.std(time_arr)

    # Presentación de métricas en tabla
    tabla_metricas = [
        ["Precisión promedio", "Desviación estándar precisión", ""],
        [f"{promedio_precision:.2f}%", f"{std_precision:.4f}", ""],
        ["Promedio fitness", "Desviación estándar fitness", ""],
        [f"{promedio_fitness:.5f}", f"{std_fitness:.4f}", ""],
        ["Promedio iteraciones", "Desv. estándar iteraciones", ""],
        [f"{promedio_iter:.2f}", f"{std_iter:.3f}", ""],
        ["Promedio tiempo (s)", "Desv. estándar tiempo", ""],
        [f"{promedio_tiempo:.5f}", f"{std_tiempo:.6f}", ""]
    ]

    print("\nResumen de métricas sobre 1 corrida:\n")
    print(tabulate(tabla_metricas, tablefmt="fancy_grid"))

    # Visualización de la mejor solución encontrada
    best_coalition = [int(i) for i in best_coalition]
    pertenece = np.zeros(n_diputados, dtype=bool)
    pertenece[best_coalition] = True

    hull_vertices, hull = compute_hull_vertices(coordenadas, best_coalition)
    print(f"\nFitness última corrida: {best_fit}")
    print(f"Coalición indices: {sorted(best_coalition)}")
    print(f"Vértices hull: {sorted([int(i) for i in hull_vertices])}")
    plot_solution(coordenadas, best_coalition, hull, partidos=partidos, pertenece=pertenece)
