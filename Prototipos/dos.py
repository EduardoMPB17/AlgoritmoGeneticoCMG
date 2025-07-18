import json
import random
import time

import numpy as np
import matplotlib.pyplot as plt
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

def tournament_selection(pop, scores, selection_prob):
    """
    Selección basada en ranking con distribución geométrica truncada.
    Cada individuo tiene probabilidad > 0, decreciente con su ranking.
    """
    rng = np.random.default_rng()
    # Ordenar la población de mejor a peor
    sorted_indices = np.argsort(scores)
    sorted_population = [pop[i] for i in sorted_indices]

    # Calcular probabilidades geométricas para todos los individuos
    n = len(pop)
    weights = np.array([selection_prob * (1 - selection_prob)**i for i in range(n)])
    weights = weights / weights.sum()

    # Seleccionar dos padres con esas probabilidades
    selected_indices = rng.choice(n, size=2, replace=True, p=weights)
    return sorted_population[selected_indices[0]], sorted_population[selected_indices[1]]

def crossover(parent1, parent2, quorum):
    """
    Cruza dos padres para generar dos hijos, corrigiendo el tamaño de la coalición.
    """
    point = random.randint(1, len(parent1) - 1)
    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))
    return correct_coalition(child1, quorum), correct_coalition(child2, quorum)

def mutate(coalition, mutation_rate):
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
def genetic_mwc(positions, quorum, pop_size, gens, mutation_rate, selection_prob):
    n = positions.shape[0]
    population = initialize_population(pop_size, n, quorum)

    best = None
    best_score = float('inf')
    fitness_cache = {}

    # Aquí almacenamos evolución
    best_fitness_history = []
    best_gen_history = []

    for gen in range(gens):
        scores = [fitness(ind, positions, fitness_cache) for ind in population]
        elite_idx = np.argmin(scores)
        elite = population[elite_idx]
        elite_score = scores[elite_idx]
        
        if elite_score < best_score:
            best_score, best = elite_score, elite.copy()
            # Guardamos el cambio cuando mejora
            best_fitness_history.append(best_score)
            best_gen_history.append(gen + 1)  # +1 para que la iteración empiece en 1
            print(f"Generación {gen+1}/{gens} - Mejor Fitness: {elite_score:.5f}")
        
        new_pop = [elite.copy()]
        num_pairs = (pop_size -1 ) // 2
        for _ in range(num_pairs):
            p1, p2 = tournament_selection(population, scores, selection_prob)
            child1, child2 = crossover(p1, p2, quorum)
            new_pop.extend([
                mutate(child1, mutation_rate),
                mutate(child2, mutation_rate)
            ])
        population = new_pop[:pop_size]
    
    # Devolvemos además el historial de fitness para graficar
    return np.where(best == 1)[0], best_score, best_gen_history, best_fitness_history

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

def plot_solution(ax, positions, coalition, hull, partidos=None, pertenece=None):
    """
    Versión modificada para trabajar con subplots
    """
    ax.set_facecolor('#bdbdbd')
    ax.grid(True, color='white', linewidth=1.2)
    
    if partidos is None:
        partidos = np.array(['Otro']*positions.shape[0])
    if pertenece is None:
        pertenece = np.zeros(positions.shape[0], dtype=bool)
        pertenece[coalition] = True
        
    partido_color = {'Demócrata':'blue', 'Republicano':'red', 'Otro':'gray'}
    marker_dict = {False:'x', True:'o'}
    marker_label = {False:'No pertenece', True:'Pertenece'}
    
    for partido in np.unique(partidos):
        for pert in [False, True]:
            idx = np.where((partidos==partido) & (pertenece==pert))[0]
            if len(idx) > 0:
                ax.scatter(
                    positions[idx,0], positions[idx,1],
                    c=partido_color.get(partido, 'gray'),
                    marker=marker_dict[pert],
                    label=f"{partido}{' - ' + marker_label[pert] if partido in partido_color else ''}",
                    edgecolor='black' if pert else None,
                    s=50 if pert else 30,
                    alpha=1 if pert else 0.7
                )
    
    hull_pts = positions[coalition][hull.vertices]
    hull_loop = np.append(hull_pts, hull_pts[0].reshape(1,2), axis=0)
    ax.fill(hull_loop[:,0], hull_loop[:,1], color='purple', alpha=0.2, zorder=2)
    ax.plot(hull_loop[:,0], hull_loop[:,1], color='purple', linewidth=2, zorder=3)
    ax.legend(loc='best', frameon=True)
    ax.set_xlabel('Dimensión 1')
    ax.set_ylabel('Dimensión 2')
    ax.set_title('Solución MWC y su Envolvente Convexa')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)

# -----------------------------
# Main
# -----------------------------
# En el main:
if __name__ == '__main__':
    # Parámetros principales
    QUORUM = 216
    POS_FILE = 'votes.json'
    POP_SIZE = 39
    MUTATION_RATE = 0.1700019
    VALOR_ESPERADO = 9686.93831
    SELECTION_PROB = 0.141
    gens = 10000
    n_iter = 100  # Número de iteraciones a ejecutar

    # Carga de datos de posiciones y partidos
    coordenadas, partidos = load_positions_and_parties(POS_FILE)


    # Almacenar resultados de todas las iteraciones
    resultados = []
    tiempos = []
    exito = []  # 1 si alcanzó el valor esperado, 0 si no
    fitness_arr = []
    gen_arr = []  # Aquí se almacenará la generación en la que se detuvo cada iteración
    time_arr = []
    precision_arr = []

    # Ejecutar múltiples iteraciones
    for i in range(n_iter):
        print(f"\n--- Iteración {i+1}/{n_iter} ---")
        t0 = time.time()
        best_coalition, best_fit, gen_hist, fitness_hist = genetic_mwc(
            coordenadas, QUORUM,
            pop_size=POP_SIZE,
            gens=gens,
            mutation_rate=MUTATION_RATE,
            selection_prob=SELECTION_PROB
        )
        t1 = time.time()
        tiempo_ejecucion = t1 - t0

        # Guardar resultados de esta iteración
        resultados.append({
            'iteracion': i+1,
            'fitness': best_fit,
            'tiempo': tiempo_ejecucion,
            'alcanzado': abs(best_fit - VALOR_ESPERADO) < 1e-6,
            'coalicion': best_coalition,
            'gen_hist': gen_hist,
            'fitness_hist': fitness_hist
        })
        tiempos.append(tiempo_ejecucion)
        exito.append(1 if resultados[-1]['alcanzado'] else 0)
        fitness_arr.append(best_fit)
        # Guardar la generación en la que se alcanzó el mejor fitness (última de gen_hist)
        if gen_hist:
            gen_arr.append(gen_hist[-1])
        else:
            gen_arr.append(gens)
        time_arr.append(tiempo_ejecucion)
        precision_arr.append(100 * (VALOR_ESPERADO / best_fit))

        print(f"Fitness obtenido: {best_fit:.5f}")
        print(f"¿Alcanzó el valor esperado? {'Sí' if resultados[-1]['alcanzado'] else 'No'}")
        print(f"Tiempo de ejecución: {tiempo_ejecucion:.4f} segundos")

    # Buscar la primera iteración exitosa (si existe)
    primera_exitosa = next((r for r in resultados if r['alcanzado']), None)

    # Si hay una iteración exitosa, graficar esa; si no, graficar la mejor (menor fitness)
    if primera_exitosa is not None:
        best_coalition = primera_exitosa['coalicion']
        best_fit = primera_exitosa['fitness']
        gen_hist = primera_exitosa['gen_hist']
        fitness_hist = primera_exitosa['fitness_hist']
    else:
        # Buscar la mejor solución (menor fitness)
        mejor = min(resultados, key=lambda r: r['fitness'])
        best_coalition = mejor['coalicion']
        best_fit = mejor['fitness']
        gen_hist = mejor['gen_hist']
        fitness_hist = mejor['fitness_hist']

    # Visualización de la mejor solución encontrada
    best_coalition = [int(i) for i in best_coalition]
    pertenece = np.zeros(coordenadas.shape[0], dtype=bool)
    pertenece[best_coalition] = True

    hull_vertices, hull = compute_hull_vertices(coordenadas, best_coalition)
    print(f"\nFitness última corrida: {best_fit}")
    print(f"Coalición indices: {sorted(best_coalition)}")
    print(f"Vértices hull: {sorted([int(i) for i in hull_vertices])}")

    # Mostrar métricas finales como promedio de todas las iteraciones
    fitness_arr = np.array(fitness_arr)
    gen_arr = np.array(gen_arr)
    time_arr = np.array(time_arr)
    precision_arr = np.array(precision_arr)

    promedio_precision = np.mean(precision_arr)
    std_precision = np.std(precision_arr)
    promedio_fitness = np.mean(fitness_arr)
    std_fitness = np.std(fitness_arr)
    promedio_gen = np.mean(gen_arr)
    std_gen = np.std(gen_arr)
    promedio_tiempo = np.mean(time_arr)
    std_tiempo = np.std(time_arr)

    print("\n--- Métricas Promedio de las Iteraciones ---")
    print(f"Promedio precisión: {promedio_precision:.2f}% ± {std_precision:.2f}")
    print(f"Promedio fitness: {promedio_fitness:.5f} ± {std_fitness:.5f}")
    print(f"Promedio generaciones: {promedio_gen:.2f} ± {std_gen:.2f}")
    print(f"Promedio tiempo: {promedio_tiempo:.4f} ± {std_tiempo:.4f} segundos")



    # Crear figura con ambos gráficos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Gráfico 1: Solución MWC
    plot_solution(ax1, coordenadas, best_coalition, hull, partidos=partidos, pertenece=pertenece)

    # Gráfico 2: Evolución del Fitness
    ax2.plot(gen_hist, fitness_hist, marker='o', linestyle='-', label='Fitness evolutivo')
    ax2.axhline(y=VALOR_ESPERADO, color='r', linestyle='--', label='Valor esperado')

    # Marcar punto de objetivo alcanzado (si aplica)
    for gen, fit in zip(gen_hist, fitness_hist):
        if abs(fit - VALOR_ESPERADO) < 1e-6:  
            ax2.scatter(gen, fit, marker='*', s=200, color='gold', zorder=5)
            ax2.text(gen, fit + 200, f'({gen}, {fit:.5f})', 
                    ha='center', va='bottom', fontsize=12, color='green', weight='bold')
            break

    ax2.set_xlabel('Generación')
    ax2.set_ylabel('Mejor Fitness encontrado')
    ax2.set_title('Evolución del Fitness')
    ax2.grid(True)
    ax2.set_xlim(1, gens)
    ax2.legend()

    plt.tight_layout(pad=3.0)
    plt.savefig(f"resultado_iteracion_{n_iter}.png")
    plt.show()

    # Guardar métricas promedio en un archivo de texto
    with open(f"metricas_promedio_{n_iter}.txt", "w", encoding="utf-8") as f:
        f.write("--- Métricas Promedio de las Iteraciones ---\n")
        f.write(f"Promedio precisión: {promedio_precision:.2f}% ± {std_precision:.2f}\n")
        f.write(f"Promedio fitness: {promedio_fitness:.5f} ± {std_fitness:.5f}\n")
        f.write(f"Promedio generaciones: {promedio_gen:.2f} ± {std_gen:.2f}\n")
        f.write(f"Promedio tiempo: {promedio_tiempo:.4f} ± {std_tiempo:.4f} segundos\n")

    # Guardar resultados de todas las iteraciones en un archivo CSV
    import csv
    with open(f"resultados_iteraciones_{n_iter}.csv", "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["iteracion", "fitness", "tiempo", "alcanzado", "generacion_mejor_fitness", "precision"])
        for i, r in enumerate(resultados):
            writer.writerow([
                r['iteracion'],
                r['fitness'],
                r['tiempo'],
                int(r['alcanzado']),
                gen_arr[i],
                precision_arr[i]
            ])

    # Guardar la evolución del fitness de la mejor corrida en un archivo CSV
    with open(f"evolucion_fitness_{n_iter}.csv", "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["generacion", "fitness"])
        for g, fit in zip(gen_hist, fitness_hist):
            writer.writerow([g, fit])