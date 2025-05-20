# Importamos las bibliotecas necesarias para nuestro programa
import matplotlib.pyplot as plt  # Esta biblioteca nos permite crear gráficos y visualizaciones
import numpy as np  # Numpy nos ayuda a trabajar con matrices y operaciones matemáticas eficientes
import random  # Utilizamos random para añadir aleatoriedad en la generación del laberinto
import heapq  # Heapq implementa el algoritmo de cola de prioridad que necesitamos para A*
from matplotlib.patches import Circle  # Importamos Circle para dibujar nuestro cursor


def crear_maze(dim, num_rutas=3):
    # Esta función crea un laberinto de dimensión dim x dim
    # Inicializamos el laberinto como una matriz llena de unos (paredes)
    # La dimensión es (2*dim+1) porque necesitamos espacio para las paredes entre celdas
    laberinto = np.ones((dim * 2 + 1, dim * 2 + 1), dtype=int)

    # Comenzamos en la esquina superior izquierda (0,0)
    x, y = 0, 0
    # Marcamos esta celda como parte del camino (0 = camino, 1 = pared)
    laberinto[2 * x + 1, 2 * y + 1] = 0
    # Inicializamos una pila para el algoritmo de backtracking
    pila = [(x, y)]

    # Algoritmo de generación de laberinto por backtracking
    while pila:
        # Tomamos la celda actual de la parte superior de la pila
        x, y = pila[-1]
        # Definimos las cuatro direcciones posibles (derecha, abajo, izquierda, arriba)
        direcciones = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        # Mezclamos las direcciones para añadir aleatoriedad
        random.shuffle(direcciones)

        # Intentamos movernos en una dirección aleatoria
        for dx, dy in direcciones:
            # Calculamos las coordenadas de la nueva celda
            nx, ny = x + dx, y + dy
            # Verificamos si la nueva celda está dentro de los límites y no ha sido visitada
            if 0 <= nx < dim and 0 <= ny < dim and laberinto[2 * nx + 1, 2 * ny + 1] == 1:
                # Marcamos la nueva celda como parte del camino
                laberinto[2 * nx + 1, 2 * ny + 1] = 0
                # También eliminamos la pared entre la celda actual y la nueva
                laberinto[2 * x + 1 + dx, 2 * y + 1 + dy] = 0
                # Añadimos la nueva celda a la pila
                pila.append((nx, ny))
                # Salimos del bucle for ya que encontramos una dirección válida
                break
        else:
            # Si no encontramos ninguna dirección válida, retrocedemos (backtracking)
            pila.pop()

    # Creamos la entrada y salida del laberinto
    laberinto[1, 0] = 0  # Entrada en la parte superior izquierda
    laberinto[-2, -1] = 0  # Salida en la parte inferior derecha

    # Ahora eliminamos algunas paredes adicionales para crear múltiples rutas
    paredes_eliminadas = 0
    # Aumentamos el número de paredes a eliminar para crear más caminos alternativos
    paredes_objetivo = max(5, dim // 2 + 2)

    while paredes_eliminadas < paredes_objetivo:
        # Seleccionamos una posición aleatoria en el laberinto
        x = random.randint(1, laberinto.shape[0] - 2)
        y = random.randint(1, laberinto.shape[1] - 2)

        # Verificamos si es una pared
        if laberinto[x, y] == 1:
            # Contamos cuántas celdas adyacentes son caminos
            celdas_adyacentes = 0
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < laberinto.shape[0] and
                        0 <= ny < laberinto.shape[1] and
                        laberinto[nx, ny] == 0):
                    celdas_adyacentes += 1

            # Si hay al menos 2 celdas adyacentes que son caminos, eliminamos la pared
            # Esto crea conexiones entre caminos existentes
            if celdas_adyacentes >= 2:
                laberinto[x, y] = 0
                paredes_eliminadas += 1

    # Devolvemos el laberinto generado
    return laberinto


def encontrar_multiples_paths(laberinto, num_paths=3):
    # Esta función encuentra múltiples rutas en el laberinto usando el algoritmo A*
    # Definimos las cuatro direcciones posibles (derecha, abajo, izquierda, arriba)
    direcciones = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    # Definimos el punto de inicio y fin
    inicio = (1, 1)  # Justo después de la entrada
    fin = (laberinto.shape[0] - 2, laberinto.shape[1] - 2)  # Justo antes de la salida
    # Lista para almacenar los caminos encontrados
    caminos = []

    # Función para calcular la distancia Manhattan entre dos puntos
    # Esta es nuestra heurística para el algoritmo A*
    def manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # Función para encontrar un camino usando el algoritmo A*
    def encontrar_camino_astar():
        # Inicializamos la frontera con el punto de inicio
        # Cada elemento es (prioridad, costo, posición, camino)
        frontera = [(manhattan(inicio, fin), 0, inicio, [inicio])]
        # Conjunto para rastrear celdas visitadas
        visitado = set([inicio])

        while frontera:
            # Extraemos el nodo con menor prioridad (A* es un algoritmo greedy)
            _, costo, actual, camino = heapq.heappop(frontera)

            # Si llegamos al final, devolvemos el camino
            if actual == fin:
                return camino

            # Exploramos las cuatro direcciones posibles
            for dx, dy in direcciones:
                siguiente = (actual[0] + dx, actual[1] + dy)
                # Verificamos si la nueva posición es válida
                if (0 <= siguiente[0] < laberinto.shape[0] and
                        0 <= siguiente[1] < laberinto.shape[1] and
                        laberinto[siguiente] == 0 and  # Es un camino, no una pared
                        siguiente not in visitado):  # No ha sido visitada
                    # Marcamos como visitada
                    visitado.add(siguiente)
                    # Calculamos el nuevo costo
                    nuevo_costo = costo + 1
                    # Calculamos la prioridad (costo + heurística)
                    prioridad = nuevo_costo + manhattan(siguiente, fin)
                    # Añadimos a la frontera
                    heapq.heappush(frontera, (prioridad, nuevo_costo, siguiente, camino + [siguiente]))

        # Si no encontramos camino, devolvemos None
        return None

    # Encontramos el primer camino
    primer_camino = encontrar_camino_astar()
    if primer_camino:
        caminos.append(primer_camino)

    # Función para encontrar caminos alternativos
    def encontrar_camino_alternativo(caminos_previos):
        # Creamos una matriz de penalización para evitar usar las mismas celdas
        penalizacion = np.zeros_like(laberinto, dtype=float)
        for camino in caminos_previos:
            for celda in camino:
                # Añadimos una penalización a cada celda ya utilizada
                penalizacion[celda] += 5.0  # Reducido para permitir más superposición

        # Similar al A* original, pero con penalizaciones
        frontera = [(manhattan(inicio, fin), 0, inicio, [inicio])]
        visitado = set([inicio])

        while frontera:
            _, costo, actual, camino = heapq.heappop(frontera)

            if actual == fin:
                return camino

            for dx, dy in direcciones:
                siguiente = (actual[0] + dx, actual[1] + dy)
                if (0 <= siguiente[0] < laberinto.shape[0] and
                        0 <= siguiente[1] < laberinto.shape[1] and
                        laberinto[siguiente] == 0 and
                        siguiente not in visitado):
                    visitado.add(siguiente)
                    # Aquí añadimos la penalización al costo
                    nuevo_costo = costo + 1 + penalizacion[siguiente]
                    prioridad = nuevo_costo + manhattan(siguiente, fin)
                    heapq.heappush(frontera, (prioridad, nuevo_costo, siguiente, camino + [siguiente]))

        return None

    # Intentamos encontrar caminos alternativos
    intentos = 0
    max_intentos = 20  # Aumentado para dar más oportunidades

    while len(caminos) < num_paths and intentos < max_intentos:
        # Buscamos un camino alternativo
        camino_alternativo = encontrar_camino_alternativo(caminos)
        if camino_alternativo:
            # Verificamos si el camino es suficientemente diferente
            es_diferente = True
            for camino_existente in caminos:
                # Calculamos el porcentaje de celdas comunes
                celdas_comunes = set(camino_alternativo) & set(camino_existente)
                porcentaje_comun = len(celdas_comunes) / min(len(camino_alternativo), len(camino_existente))
                # Si es muy similar a un camino existente, lo descartamos
                if porcentaje_comun > 0.8:  # Más permisivo
                    es_diferente = False
                    break

            # Si es suficientemente diferente, lo añadimos
            if es_diferente:
                caminos.append(camino_alternativo)

        intentos += 1

    # Devolvemos todos los caminos encontrados
    return caminos


def encontrar_ruta_mas_corta(rutas):
    # Esta función identifica cuál de las rutas es la más corta
    if not rutas:
        return None, 0

    # Calculamos la longitud de cada ruta
    longitudes = [len(ruta) for ruta in rutas]
    # Encontramos el índice de la ruta más corta
    indice_mas_corto = longitudes.index(min(longitudes))
    return indice_mas_corto, longitudes


def dibujar_maze(laberinto, caminos=None, indice_mas_corto=None, longitudes=None):
    # Esta función dibuja el laberinto y las rutas encontradas
    # Creamos una figura y un eje para dibujar
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_edgecolor('white')
    fig.patch.set_linewidth(0)
    # Mostramos el laberinto como una imagen binaria
    ax.imshow(laberinto, cmap=plt.cm.binary, interpolation='nearest')

    # Definimos colores para las diferentes rutas
    colores = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']

    # Si hay caminos, los dibujamos
    if caminos:
        for i, camino in enumerate(caminos):
            # Seleccionamos un color para esta ruta
            color = colores[i % len(colores)]
            # Extraemos las coordenadas x e y
            x_coords = [x[1] for x in camino]
            y_coords = [x[0] for x in camino]

            # Estilo especial para la ruta más corta
            if i == indice_mas_corto:
                ax.plot(x_coords, y_coords, color=color, linewidth=3, linestyle='-',
                        label=f'Ruta {i + 1} ({longitudes[i]} pasos) - MÁS CORTA')
            else:
                ax.plot(x_coords, y_coords, color=color, linewidth=2,
                        label=f'Ruta {i + 1} ({longitudes[i]} pasos)')

        # Añadimos una leyenda para identificar las rutas
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

    # Eliminamos los ejes para una visualización más limpia
    ax.set_xticks([])
    ax.set_yticks([])
    # Añadimos flechas para indicar la entrada y salida
    ax.arrow(0, 1, 0.4, 0, fc='green', ec='green', head_width=0.3, head_length=0.3)
    ax.arrow(laberinto.shape[1] - 1, laberinto.shape[0] - 2, -0.4, 0, fc='blue', ec='blue',
             head_width=0.3, head_length=0.3)

    # Añadimos un título y mostramos la figura
    plt.title("Laberinto con múltiples rutas")
    plt.show()


# Función para control manual del cursor
def control_manual_cursor(laberinto, caminos, indice_mas_corto, longitudes):
    # Esta función permite al usuario controlar un cursor en el laberinto
    # Creamos una figura y un eje para dibujar
    fig, ax = plt.subplots(figsize=(10, 10))
    # Mostramos el laberinto como una imagen binaria
    ax.imshow(laberinto, cmap=plt.cm.binary, interpolation='nearest')

    # Definimos colores para las diferentes rutas
    colores = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']

    # Dibujamos todas las rutas
    for i, camino in enumerate(caminos):
        # Seleccionamos un color para esta ruta
        color = colores[i % len(colores)]
        # Extraemos las coordenadas x e y
        x_coords = [x[1] for x in camino]
        y_coords = [x[0] for x in camino]

        # Estilo especial para la ruta más corta
        if i == indice_mas_corto:
            ax.plot(x_coords, y_coords, color=color, linewidth=3, linestyle='-',
                    label=f'Ruta {i + 1} ({longitudes[i]} pasos) - MÁS CORTA')
        else:
            ax.plot(x_coords, y_coords, color=color, linewidth=2,
                    label=f'Ruta {i + 1} ({longitudes[i]} pasos)')

    # Añadimos una leyenda para identificar las rutas
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

    # Eliminamos los ejes para una visualización más limpia
    ax.set_xticks([])
    ax.set_yticks([])
    # Añadimos flechas para indicar la entrada y salida
    ax.arrow(0, 1, 0.4, 0, fc='green', ec='green', head_width=0.3, head_length=0.3)
    ax.arrow(laberinto.shape[1] - 1, laberinto.shape[0] - 2, -0.4, 0, fc='blue', ec='blue',
             head_width=0.3, head_length=0.3)

    # Creamos un diccionario para almacenar todas las celdas de las rutas
    # Esto nos ayudará a saber en qué ruta(s) está el cursor
    celdas_rutas = {}
    for i, camino in enumerate(caminos):
        for j, celda in enumerate(camino):
            if celda not in celdas_rutas:
                celdas_rutas[celda] = []
            # Guardamos (índice_ruta, posición_en_ruta) para cada celda
            celdas_rutas[celda].append((i, j))

    # Creamos un cursor (círculo) en la posición inicial
    posicion_cursor = caminos[0][0]  # Comenzamos en la primera celda de la primera ruta
    # El cursor es un círculo amarillo
    cursor = Circle((posicion_cursor[1], posicion_cursor[0]), 0.4,
                    color='yellow', alpha=0.8, zorder=10)
    ax.add_patch(cursor)

    # Creamos textos para mostrar información sobre la posición actual
    info_ruta = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                        color='black', fontsize=12, fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.7))

    # Texto para mostrar el número de pasos dados
    info_pasos = ax.text(0.02, 0.93, "", transform=ax.transAxes,
                         color='black', fontsize=12, fontweight='bold',
                         bbox=dict(facecolor='white', alpha=0.7))

    # Texto con instrucciones para el usuario
    instrucciones = ax.text(0.5, 0.02, "Usa las teclas ↑ ↓ ← → para mover el cursor\n"
                                       "Presiona 'r' para reiniciar, 'q' para salir",
                            transform=ax.transAxes, color='black', fontsize=12,
                            ha='center', bbox=dict(facecolor='white', alpha=0.7))

    # Variables para seguimiento del movimiento
    pasos_dados = 0  # Contador de pasos
    ruta_actual = 0  # Índice de la ruta actual
    posicion_en_ruta = 0  # Posición dentro de la ruta actual
    historial_movimientos = [posicion_cursor]  # Historial de posiciones visitadas

    # Función para actualizar la información mostrada
    def actualizar_info():
        # Encontrar en qué rutas está la posición actual
        rutas_actuales = []
        if posicion_cursor in celdas_rutas:
            for idx_ruta, pos_ruta in celdas_rutas[posicion_cursor]:
                rutas_actuales.append(f"Ruta {idx_ruta + 1}")

        # Actualizamos el texto con la información de las rutas
        if rutas_actuales:
            info_ruta.set_text(f"Posición actual en: {', '.join(rutas_actuales)}")
        else:
            info_ruta.set_text("Posición actual: Fuera de ruta")

        # Actualizamos el contador de pasos
        info_pasos.set_text(f"Pasos dados: {pasos_dados}")
        # Redibujamos la figura para mostrar los cambios
        fig.canvas.draw_idle()

    # Inicializamos la información
    actualizar_info()

    # Función para manejar eventos de teclado
    def on_key(event):
        # Usamos nonlocal para modificar variables de la función externa
        nonlocal posicion_cursor, pasos_dados, historial_movimientos

        # Obtenemos la posición actual
        y, x = posicion_cursor

        # Movemos según la tecla presionada
        if event.key == 'up':
            nueva_pos = (y - 1, x)  # Mover hacia arriba
        elif event.key == 'down':
            nueva_pos = (y + 1, x)  # Mover hacia abajo
        elif event.key == 'left':
            nueva_pos = (y, x - 1)  # Mover hacia la izquierda
        elif event.key == 'right':
            nueva_pos = (y, x + 1)  # Mover hacia la derecha
        elif event.key == 'r':  # Reiniciar
            # Volvemos a la posición inicial
            posicion_cursor = caminos[0][0]
            cursor.center = (posicion_cursor[1], posicion_cursor[0])
            pasos_dados = 0
            historial_movimientos = [posicion_cursor]
            actualizar_info()
            return
        elif event.key == 'q':  # Salir
            plt.close(fig)
            return
        else:
            return  # Ignoramos otras teclas

        # Verificamos si la nueva posición es válida
        # Debe estar dentro del laberinto y ser un camino (no una pared)
        if (0 <= nueva_pos[0] < laberinto.shape[0] and
                0 <= nueva_pos[1] < laberinto.shape[1] and
                laberinto[nueva_pos] == 0):

            # Verificamos si la nueva posición está adyacente a la posición actual
            # Solo permitimos movimientos ortogonales (no diagonales)
            if abs(nueva_pos[0] - y) + abs(nueva_pos[1] - x) == 1:
                # Movemos el cursor
                posicion_cursor = nueva_pos
                cursor.center = (posicion_cursor[1], posicion_cursor[0])

                # Incrementamos el contador de pasos solo si es una nueva posición
                # Esto evita contar múltiples veces la misma celda
                if nueva_pos not in historial_movimientos:
                    pasos_dados += 1

                # Añadimos la nueva posición al historial
                historial_movimientos.append(posicion_cursor)

                # Actualizamos la información mostrada
                actualizar_info()

    # Conectamos el evento de teclado a nuestra función
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Añadimos un título y mostramos la figura
    plt.title("Control manual del cursor - Usa las teclas de dirección")
    plt.tight_layout()
    plt.show()


# Función para intentar generar un laberinto con múltiples rutas
def generar_laberinto_con_multiples_rutas(dim, num_rutas_deseadas, max_intentos=5):
    # Esta función intenta generar un laberinto con el número deseado de rutas
    # Si no lo logra en el primer intento, lo intenta varias veces
    for intento in range(max_intentos):
        print(f"\n🔄 Intento {intento + 1}/{max_intentos} de generar laberinto con {num_rutas_deseadas} rutas...")

        # Generamos un laberinto con más paredes eliminadas para aumentar posibilidades
        # Pedimos más rutas de las necesarias para aumentar la probabilidad
        laberinto = crear_maze(dim, num_rutas_deseadas + 2)
        rutas = encontrar_multiples_paths(laberinto, num_rutas_deseadas)

        # Si encontramos suficientes rutas, terminamos
        if len(rutas) >= num_rutas_deseadas:
            print(f"✅ Éxito! Se encontraron {len(rutas)} rutas diferentes.")
            return laberinto, rutas

        print(f"⚠ Solo se encontraron {len(rutas)} rutas. Intentando de nuevo...")

    # Si después de varios intentos no se logra, usamos el último resultado
    print(f"⚠ Después de {max_intentos} intentos, solo se pudieron generar {len(rutas)} rutas.")
    return laberinto, rutas


# Bloque principal
if __name__ == "__main__":
    # Este bloque se ejecuta cuando corremos el script directamente
    while True:
        # Definimos las dimensiones del laberinto y el número de rutas deseadas
        dim = 10  # Laberinto de 10x10 celdas
        num_rutas_deseadas = 3  # Queremos 3 rutas diferentes

        # Usamos la función para generar un laberinto con múltiples rutas
        laberinto, rutas = generar_laberinto_con_multiples_rutas(dim, num_rutas_deseadas)

        # Encontramos la ruta más corta
        indice_mas_corto, longitudes = encontrar_ruta_mas_corta(rutas)

        # Mostramos información sobre todas las rutas
        if len(rutas) > 1:
            print("\n📏 Longitud de cada ruta:")
            for i, longitud in enumerate(longitudes):
                if i == indice_mas_corto:
                    print(f"   Ruta {i + 1}: {longitud} pasos 🏆 (MÁS CORTA)")
                else:
                    print(f"   Ruta {i + 1}: {longitud} pasos")
        elif len(rutas) == 1:
            print(f"ℹ Solo hay una ruta posible con {longitudes[0]} pasos.")
        else:
            print("❌ No se encontraron rutas.")

        # Primero mostramos el laberinto con todas las rutas
        dibujar_maze(laberinto, rutas, indice_mas_corto, longitudes)

        # Luego mostramos el control manual del cursor
        print("\n🎮 Iniciando control manual del cursor...")
        print("Usa las teclas de dirección (↑ ↓ ← →) para mover el cursor por el laberinto.")
        print("Presiona 'r' para reiniciar y 'q' para salir.")
        control_manual_cursor(laberinto, rutas, indice_mas_corto, longitudes)

        # Preguntamos si el usuario quiere generar otro laberinto
        seguir = input("¿Quieres generar otro laberinto? (s/n): ")
        if seguir.lower() != 's':
            print("👋 Saliendo del generador de laberintos. ¡Hasta la próxima!")
            break
