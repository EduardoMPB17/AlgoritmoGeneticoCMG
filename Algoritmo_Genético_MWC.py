import json
import random

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

#* -----------------------------
#* Carga de datos
#* -----------------------------
def load_positions_and_parties(filename):
    """
    Carga las posiciones (x, y) y los partidos de los votos desde un archivo JSON.
    """
    with open(filename, 'r') as f:
        data = json.load(f)

    # Lista de legisladores y su información (incluye partido, coordenadas ideológicas y voto emitido en una única votación)
    votes = data['rollcalls'][0]['votes']

    # Array de coordenadas (x, y) donde cada punto representa la ideología de un legislador en el espacio político
    # X → representa la dimensión económica izquierda-derecha
    # Y → representa una segunda dimensión ideológica
    coordenadas = np.array([[v['x'], v['y']] for v in votes])

    # Array del nombre del partido político de cada legislador según su código de partido:
    # 100 = Demócrata
    # 200 = Republicano
    # otro = Otro 
    partidos = np.array(['Demócrata' if v.get('party_code',0)==100 else 'Republicano' if v.get('party_code',0)==200 else 'Otro' for v in votes])
    return coordenadas, partidos

#* -----------------------------
#* Inicializar población
#* -----------------------------
def initialize_population(pop_size, n, quorum):
    """
    Inicializa la población de coaliciones aleatorias.

    Parámetros:
    - pop_size: cantidad de coaliciones a generar (39)
    - n: número total de diputados (431)
    - quorum: tamaño de cada coalición (216)

    Retorna:
    - population: lista con pop_size vectores binarios de longitud n,
      donde cada vector representa una coalición con 'quorum' diputados.
    """
    population = []                                # Lista de 39 coaliciones (población inicial)
    for _ in range(pop_size):
        coalition = np.zeros(n, dtype=int)         # Vector de 431 ceros
        indices = random.sample(range(n), quorum)  # 216 índices únicos aleatorios
        coalition[indices] = 1                     # Marcar esos índices como 1
        population.append(coalition)               # Agrega la coalición a la población
    return population                              # Lista con 39 coaliciones (cada una de largo 431)

#* -----------------------------
#* Fitness
#* -----------------------------
def fitness(coalition, positions, fitness_cache=None):
    """
    Calcula el fitness (cohesión) de una coalición como la suma total de distancias euclidianas 
    entre sus miembros en el espacio ideológico (2D). Un valor más bajo indica mayor cohesión.

    Parámetros:
    - coalition: vector binario de longitud n (431), donde 1 indica inclusión en la coalición.
    - positions: array de shape (n, 2) con coordenadas ideológicas (x, y).
    - fitness_cache: diccionario opcional para guardar resultados previamente calculados.

    Retorna:
    - val: suma de distancias entre todos los pares de miembros en la coalición.
    """
    # Si se proporciona caché, intentamos reutilizar el resultado si ya se calculó antes.
    if fitness_cache is not None:
        key = tuple(coalition)         # Convertimos a tupla para usar como clave
        if key in fitness_cache:
            return fitness_cache[key]  # Devuelve el valor guardado sin recalcular

    indices = np.where(coalition == 1)[0] # Índices de los diputados en la coalición

    if len(indices) < 2:
        val = float('inf') # No se puede calcular distancia si hay menos de 2
    else:
        puntos = positions[indices]                     # Coordenadas de los diputados en la coalición
        val = np.sum(pdist(puntos, metric='euclidean')) # C(n, 2) = (n × (n - 1)) / 2 ⇒ 23,220

    # Guarda el resultado en caché si se está utilizando
    if fitness_cache is not None:
        fitness_cache[key] = val

    # Devuelve el valor de fitness calculado
    # n° grande  → Congresistas muy dispersos en el espacio ideológico
    # n° pequeño → Congresistas más agrupados ideológicamente
    return val

#* -----------------------------
#* Selección por torneo
#* -----------------------------
def tournament_selection(pop, scores, selection_prob):
    """
    Selección basada en ranking con distribución geométrica truncada.
    Cada individuo tiene probabilidad > 0, decreciente con su ranking.
    """
    rng = np.random.default_rng() # Generador de números aleatorios 
    
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

#* -----------------------------
#* Corrección de coalición
#* -----------------------------
def correct_coalition(coalition, quorum):
    """
    Ajusta la coalición para que tenga exactamente 'quorum' miembros seleccionados.
    
    Parámetros:
    - coalition: vector binario (array de 0s y 1s) que representa la coalición actual.
    - quorum: número exacto de miembros que debe tener la coalición.
    
    Funcionamiento:
    - Si la coalición tiene más miembros que el quorum, apaga (pone a 0) aleatoriamente los excedentes.
    - Si tiene menos miembros que el quorum, enciende (pone a 1) aleatoriamente los faltantes.
    - Devuelve una nueva coalición corregida con exactamente 'quorum' miembros.
    """
    coalition = coalition.copy()                      # Crear una copia para no modificar la original
    current = np.sum(coalition)                       # Contar cuántos miembros están activos (1s)

    if current > quorum:                              # Si hay más miembros que el quorum
        indices_1 = np.where(coalition == 1)[0]       # Obtener índices de miembros activos
        to_turn_off = random.sample(list(indices_1), current - quorum)  # Elegir aleatoriamente los índices para apagar
        coalition[to_turn_off] = 0                    # Apagar los seleccionados para ajustar tamaño

    elif current < quorum:                            # Si hay menos miembros que el quorum
        indices_0 = np.where(coalition == 0)[0]       # Obtener índices de miembros inactivos
        to_turn_on = random.sample(list(indices_0), quorum - current)   # Elegir aleatoriamente cuáles activar
        coalition[to_turn_on] = 1                     # Activar los seleccionados para ajustar tamaño

    return coalition                                  # Devolver la coalición corregida

#* -----------------------------
#* Cruzamiento
#* -----------------------------
def crossover(parent1, parent2, quorum):
    """
    Cruza dos padres para generar dos hijos, corrigiendo el tamaño de la coalición.
    """
    point = random.randint(1, len(parent1) - 1)                                 # Punto de cruce aleatorio
    child1 = np.concatenate((parent1[:point], parent2[point:]))                 # 1° hijo
    child2 = np.concatenate((parent2[:point], parent1[point:]))                 # 2° hijo
    return correct_coalition(child1, quorum), correct_coalition(child2, quorum) # Hijos corregidos 

#* -----------------------------
#* Mutación
#* -----------------------------
def mutate(coalition, mutation_rate):
    """
    Aplica una mutación a la coalición con una probabilidad dada.
    La mutación consiste en intercambiar un diputado dentro de la coalición por otro fuera de ella,
    manteniendo el tamaño fijo de la coalición.
    
    Parámetros:
    - coalition: vector binario que representa la coalición actual.
    - n: tamaño total de diputados (longitud del vector).
    - mutation_rate: probabilidad de que ocurra la mutación
    
    Retorna:
    - Una coalición mutada (o sin cambios si no ocurre mutación).
    """
    coalition = coalition.copy()                      # Crear una copia para no modificar la original
    if random.random() < mutation_rate:               # Verificar si se realiza la mutación según la probabilidad
        indices_1 = np.where(coalition == 1)[0]       # Índices de diputados que están en la coalición
        indices_0 = np.where(coalition == 0)[0]       # Índices de diputados que NO están en la coalición
        if len(indices_1) > 0 and len(indices_0) > 0: # Asegurar que haya diputados dentro y fuera para intercambiar
            idx_1 = random.choice(indices_1)          # Elegir al azar un diputado dentro de la coalición
            idx_0 = random.choice(indices_0)          # Elegir al azar un diputado fuera de la coalición
            coalition[idx_1], coalition[idx_0] = 0, 1 # Intercambiar: sacar uno y meter al otro
    return coalition                                  # Devolver la coalición mutada (o la original si no hubo mutación)

#* -----------------------------
#* Algoritmo genético principal
#* -----------------------------
def genetic_mwc(positions, quorum, pop_size, gens, mutation_rate, selection_prob):
    n = positions.shape[0] # Total de congresistas (431)

    # population = Inicializa la población con 39 coaliciones aleatorias 
    # Cada coalición es un vector binario de longitud 431
    # Cada vector binario cuenta con: 
    # Exactamente 216 valores en 1, representando a los diputados que forman parte de esa coalición
    # 431 - 216 = 215 valores en 0, representando a los diputados que NO forman parte de esa coalición 
    population = initialize_population(pop_size, n, quorum)

    best = None                # Guardará la mejor coalición de todas las generaciones
    best_score = float('inf')  # El fitness de esa mejor coalición
    fitness_cache = {}         # Diccionario para almacenar resultados y evitar recálculos

    # Aquí almacenamos evolución
    best_fitness_history = []
    best_gen_history = []

    for gen in range(gens):
        # scores = array de largo 39
        # Calcula el fitness (suma total de distancias internas) de cada coalición en la población
        scores = [fitness(ind, positions, fitness_cache) for ind in population]

        elite_idx = np.argmin(scores)            # Índice de la coalición con mejor fitness
        elite = population[elite_idx]            # Coalición con fitness mínimo (el mejor) en la generación actual
        elite_score = scores[elite_idx]          # Valor del mejor fitness

        # Si esta coalición es mejor que la mejor global hasta ahora, la actualizamos
        if elite_score < best_score:
            best_score, best = elite_score, elite.copy()
            best_fitness_history.append(best_score)       # Guardamos el nuevo mejor fitness
            best_gen_history.append(gen + 1)              # Y en qué generación ocurrió
            print(f"Generación {gen+1}/{gens} - Mejor Fitness: {elite_score:.5f}")

        # Nueva población para la siguiente generación, comenzando con la elite (elitismo)
        new_pop = [elite.copy()]
        # Número de pares de padres necesarios para generar el resto de la población
        num_pairs = (pop_size -1 ) // 2

        for _ in range(num_pairs):
            p1, p2 = tournament_selection(population, scores, selection_prob) # Seleccionamos dos padres mediante torneo
            child1, child2 = crossover(p1, p2, quorum)                        # Cruzamos los padres para generar dos hijos

            # Aplicamos mutación a cada hijo
            new_pop.extend([              
                mutate(child1, mutation_rate),
                mutate(child2, mutation_rate)
            ])
        population = new_pop[:pop_size] # Reemplazamos la población con los nuevos individuos generados

    # Devolvemos:
    # - índices de diputados en la mejor coalición encontrada
    # - su fitness
    # - y la evolución histórica del fitness para análisis o gráficos
    return np.where(best == 1)[0], best_score, best_gen_history, best_fitness_history

#* -----------------------------
#* Visualización
#* -----------------------------
def compute_hull_vertices(positions, coalition):
    """
    Calcula los vértices de la envolvente convexa (polígono convexo mínimo) que rodea a los diputados de una coalición.

    Parámetros:
    - positions: array con coordenadas (x, y) de todos los diputados.
    - coalition: lista o array con los índices de los diputados que forman parte de la coalición.

    Retorna:
    - Lista con los índices de los diputados que son vértices del polígono convexo.
    - El objeto ConvexHull calculado (para uso posterior si se desea).
    """
    coalition = [int(i) for i in coalition]  # Asegura que los índices de la coalición sean enteros
    pts = positions[coalition]               # Obtiene las coordenadas (x,y) de los diputados en la coalición
    hull = ConvexHull(pts)                   # Calcula la envolvente convexa que encierra esos puntos

    # Devuelve los índices originales de los diputados que forman los vértices del polígono, y el objeto hull
    return [coalition[i] for i in hull.vertices], hull

def plot_combined_visualization(positions, coalition, hull, gen_hist, fitness_hist, partidos=None, pertenece=None, gens=None, VALOR_ESPERADO=None):
    """
    Visualiza:
    1. La solución MWC y su envolvente convexa.
    2. La evolución del fitness a lo largo de las generaciones.
    
    Agrega:
    - Línea horizontal indicando el valor esperado.
    - Estrella dorada en el punto donde se alcanza ese valor.
    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))

    # ------------------ Subplot 1: Envolvente convexa ------------------
    ax1 = axs[0]
    ax1.set_facecolor('#bdbdbd')
    ax1.grid(True, color='white', linewidth=1.2)

    if partidos is None:
        partidos = np.array(['Otro'] * positions.shape[0])
    if pertenece is None:
        pertenece = np.zeros(positions.shape[0], dtype=bool)
        pertenece[coalition] = True

    partido_color = {'Demócrata': 'blue', 'Republicano': 'red', 'Otro': 'gray'}
    marker_dict = {False: 'x', True: 'o'}
    marker_label = {False: 'No pertenece', True: 'Pertenece'}

    for partido in np.unique(partidos):
        for pert in [False, True]:
            idx = np.where((partidos == partido) & (pertenece == pert))[0]
            if len(idx) > 0:
                ax1.scatter(
                    positions[idx, 0], positions[idx, 1],
                    c=partido_color.get(partido, 'gray'),
                    marker=marker_dict[pert],
                    label=f"{partido} - {marker_label[pert]}",
                    edgecolor='black' if pert else None,
                    s=50 if pert else 30,
                    alpha=1 if pert else 0.7
                )

    hull_pts = positions[coalition][hull.vertices]
    hull_loop = np.append(hull_pts, hull_pts[0].reshape(1, 2), axis=0)
    ax1.fill(hull_loop[:, 0], hull_loop[:, 1], color='purple', alpha=0.2, zorder=2)
    ax1.plot(hull_loop[:, 0], hull_loop[:, 1], color='purple', linewidth=2, zorder=3)

    ax1.set_xlabel('Dimensión 1')
    ax1.set_ylabel('Dimensión 2')
    ax1.set_title('Solución MWC y su Envolvente Convexa')
    ax1.set_xlim(-1.1, 1.1)
    ax1.set_ylim(-1.1, 1.1)
    ax1.legend(loc='best', frameon=True)

    # ------------------ Subplot 2: Evolución del fitness ------------------
    ax2 = axs[1]
    ax2.plot(gen_hist, fitness_hist, marker='o', linestyle='-', color='green', label='Fitness')
    ax2.set_xlabel('Generación')
    ax2.set_ylabel('Mejor Fitness encontrado')
    ax2.set_title('Evolución del Mejor Fitness')
    ax2.grid(True)
    ax2.margins(x=0.03)
    ax2.set_xlim(left=min(gen_hist), right=gens)

    # Línea horizontal para el valor esperado
    if VALOR_ESPERADO is not None:
        ax2.axhline(y=VALOR_ESPERADO, color='r', linestyle='--', label='Valor esperado')

        # Buscar si se alcanzó el valor esperado
        for gen, fit in zip(gen_hist, fitness_hist):
            if abs(fit - VALOR_ESPERADO) < 1e-6:
                ax2.scatter(gen, fit, marker='*', s=200, color='gold', zorder=5, label='Fitness alcanzado')
                ax2.text(gen, fit + 200, f'({gen}, {fit:.5f})',
                        ha='center', va='bottom', fontsize=12, color='green', weight='bold')
                break

    ax2.legend()
    plt.tight_layout()
    plt.savefig('Gráficas.png', dpi=300)
    plt.show()

#* -----------------------------
#* Main
#* -----------------------------
if __name__ == '__main__':
    # Parámetros principales
    QUORUM = 216                 # Número mínimo de legisladores para formar una coalición válida
    POS_FILE = 'votes.json'      # Archivo donde están las posiciones ideológicas de los legisladores
    POP_SIZE = 39                # Tamaño de la población del algoritmo genético
    MUTATION_RATE = 0.1700019    # Probabilidad de mutación en cada generación
    SELECTION_PROB = 0.141       # Afecta cuántos individuos se eligen en cada torneo de selección
    VALOR_ESPERADO = 9686.93831  # Valor ideal de fitness

    # Carga de datos de posiciones y partidos
    coordenadas, partidos = load_positions_and_parties(POS_FILE)

    # Ejecución del algoritmo genético
    gens = 10000
    best_coalition, best_fit, gen_hist, fitness_hist = genetic_mwc(
        coordenadas, QUORUM,
        pop_size=POP_SIZE,
        gens=gens,
        mutation_rate=MUTATION_RATE,
        selection_prob=SELECTION_PROB
    )

    # Array booleano (True si el diputado pertenece a la coalición, False si no) para usarlo en el gráfico
    best_coalition = [int(i) for i in best_coalition]
    pertenece = np.zeros(coordenadas.shape[0], dtype=bool)
    pertenece[best_coalition] = True

    # Calcula los vértices del polígono que encierra a los diputados de la coalición
    hull_vertices, hull = compute_hull_vertices(coordenadas, best_coalition)

    # Mostrar resultados
    print(f"Coalición indices: {sorted(best_coalition)}")
    print(f"Vértices hull: {sorted([int(i) for i in hull_vertices])}")

    # Graficar
    plot_combined_visualization(
        coordenadas,
        best_coalition,
        hull,
        gen_hist,
        fitness_hist,
        partidos=partidos,
        pertenece=pertenece,
        gens=gens,
        VALOR_ESPERADO=VALOR_ESPERADO
    )