import matplotlib.pyplot as plt
import numpy as np
import random
import heapq
from matplotlib.patches import Circle

def crear_maze(dim, num_rutas=3):
    # El código de esta función se mantiene igual
    laberinto = np.ones((dim * 2 + 1, dim * 2 + 1), dtype=int)
    x, y = 0, 0
    laberinto[2 * x + 1, 2 * y + 1] = 0
    pila = [(x, y)]

    while pila:
        x, y = pila[-1]
        direcciones = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(direcciones)

        for dx, dy in direcciones:
            nx, ny = x + dx, y + dy
            if 0 <= nx < dim and 0 <= ny < dim and laberinto[2 * nx + 1, 2 * ny + 1] == 1:
                laberinto[2 * nx + 1, 2 * ny + 1] = 0
                laberinto[2 * x + 1 + dx, 2 * y + 1 + dy] = 0
                pila.append((nx, ny))
                break
        else:
            pila.pop()

    laberinto[1, 0] = 0
    laberinto[-2, -1] = 0

    # Aumentar el número de paredes eliminadas para crear más caminos alternativos
    paredes_eliminadas = 0
    paredes_objetivo = max(5, dim // 2 + 2)  # Aumentado para crear más caminos

    while paredes_eliminadas < paredes_objetivo:
        x = random.randint(1, laberinto.shape[0] - 2)
        y = random.randint(1, laberinto.shape[1] - 2)

        if laberinto[x, y] == 1:
            celdas_adyacentes = 0
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < laberinto.shape[0] and 0 <= ny < laberinto.shape[1] and laberinto[nx, ny] == 0:
                    celdas_adyacentes += 1

            if celdas_adyacentes >= 2:  # Cambiado de == 2 a >= 2 para ser más permisivo
                laberinto[x, y] = 0
                paredes_eliminadas += 1

    return laberinto

def encontrar_multiples_paths(laberinto, num_paths=3):
    direcciones = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    inicio = (1, 1)
    fin = (laberinto.shape[0] - 2, laberinto.shape[1] - 2)
    caminos = []

    def manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def encontrar_camino_astar():
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
                    nuevo_costo = costo + 1
                    prioridad = nuevo_costo + manhattan(siguiente, fin)
                    heapq.heappush(frontera, (prioridad, nuevo_costo, siguiente, camino + [siguiente]))

        return None

    primer_camino = encontrar_camino_astar()
    if primer_camino:
        caminos.append(primer_camino)

    def encontrar_camino_alternativo(caminos_previos):
        # Reducir la penalización para permitir más superposición entre rutas
        penalizacion = np.zeros_like(laberinto, dtype=float)
        for camino in caminos_previos:
            for celda in camino:
                penalizacion[celda] += 5.0  # Reducido de 10.0 a 5.0

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
                    nuevo_costo = costo + 1 + penalizacion[siguiente]
                    prioridad = nuevo_costo + manhattan(siguiente, fin)
                    heapq.heappush(frontera, (prioridad, nuevo_costo, siguiente, camino + [siguiente]))

        return None

    intentos = 0
    max_intentos = 20  # Aumentado de 10 a 20 para dar más oportunidades

    while len(caminos) < num_paths and intentos < max_intentos:
        camino_alternativo = encontrar_camino_alternativo(caminos)
        if camino_alternativo:
            es_diferente = True
            for camino_existente in caminos:
                celdas_comunes = set(camino_alternativo) & set(camino_existente)
                porcentaje_comun = len(celdas_comunes) / min(len(camino_alternativo), len(camino_existente))
                # Ser más permisivo con la similitud entre rutas
                if porcentaje_comun > 0.8:  # Cambiado de 0.7 a 0.8
                    es_diferente = False
                    break

            if es_diferente:
                caminos.append(camino_alternativo)

        intentos += 1

    return caminos


def encontrar_ruta_mas_corta(rutas):
    if not rutas:
        return None, 0

    longitudes = [len(ruta) for ruta in rutas]
    indice_mas_corto = longitudes.index(min(longitudes))
    return indice_mas_corto, longitudes


def dibujar_maze(laberinto, caminos=None, indice_mas_corto=None, longitudes=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_edgecolor('white')
    fig.patch.set_linewidth(0)
    ax.imshow(laberinto, cmap=plt.cm.binary, interpolation='nearest')

    colores = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']

    if caminos:
        for i, camino in enumerate(caminos):
            color = colores[i % len(colores)]
            x_coords = [x[1] for x in camino]
            y_coords = [x[0] for x in camino]

            # Estilo especial para la ruta más corta
            if i == indice_mas_corto:
                ax.plot(x_coords, y_coords, color=color, linewidth=3, linestyle='-',
                        label=f'Ruta {i + 1} ({longitudes[i]} pasos) - MÁS CORTA')
            else:
                ax.plot(x_coords, y_coords, color=color, linewidth=2,
                        label=f'Ruta {i + 1} ({longitudes[i]} pasos)')

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.arrow(0, 1, 0.4, 0, fc='green', ec='green', head_width=0.3, head_length=0.3)
    ax.arrow(laberinto.shape[1] - 1, laberinto.shape[0] - 2, -0.4, 0, fc='blue', ec='blue',
             head_width=0.3, head_length=0.3)

    plt.title("Laberinto con múltiples rutas")
    plt.show()


# Función para control manual del cursor
def control_manual_cursor(laberinto, caminos, indice_mas_corto, longitudes):
    # Crear la figura y el eje
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(laberinto, cmap=plt.cm.binary, interpolation='nearest')

    colores = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']

    # Dibujar todas las rutas
    for i, camino in enumerate(caminos):
        color = colores[i % len(colores)]
        x_coords = [x[1] for x in camino]
        y_coords = [x[0] for x in camino]

        # Estilo especial para la ruta más corta
        if i == indice_mas_corto:
            ax.plot(x_coords, y_coords, color=color, linewidth=3, linestyle='-',
                    label=f'Ruta {i + 1} ({longitudes[i]} pasos) - MÁS CORTA')
        else:
            ax.plot(x_coords, y_coords, color=color, linewidth=2,
                    label=f'Ruta {i + 1} ({longitudes[i]} pasos)')

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

    # Configuración inicial
    ax.set_xticks([])
    ax.set_yticks([])
    ax.arrow(0, 1, 0.4, 0, fc='green', ec='green', head_width=0.3, head_length=0.3)
    ax.arrow(laberinto.shape[1] - 1, laberinto.shape[0] - 2, -0.4, 0, fc='blue', ec='blue',
             head_width=0.3, head_length=0.3)

    # Crear un diccionario para almacenar todas las celdas de las rutas
    celdas_rutas = {}
    for i, camino in enumerate(caminos):
        for j, celda in enumerate(camino):
            if celda not in celdas_rutas:
                celdas_rutas[celda] = []
            celdas_rutas[celda].append((i, j))  # Guarda (índice_ruta, posición_en_ruta)

    # Crear un cursor (círculo) en la posición inicial
    posicion_cursor = caminos[0][0]  # Comenzar en la primera celda de la primera ruta
    cursor = Circle((posicion_cursor[1], posicion_cursor[0]), 0.4,
                    color='yellow', alpha=0.8, zorder=10)
    ax.add_patch(cursor)

    # Crear textos para mostrar información
    info_ruta = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                        color='black', fontsize=12, fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.7))

    info_pasos = ax.text(0.02, 0.93, "", transform=ax.transAxes,
                         color='black', fontsize=12, fontweight='bold',
                         bbox=dict(facecolor='white', alpha=0.7))

    instrucciones = ax.text(0.5, 0.02, "Usa las teclas ↑ ↓ ← → para mover el cursor\n"
                                       "Presiona 'r' para reiniciar, 'q' para salir",
                            transform=ax.transAxes, color='black', fontsize=12,
                            ha='center', bbox=dict(facecolor='white', alpha=0.7))

    # Variables para seguimiento
    pasos_dados = 0
    ruta_actual = 0
    posicion_en_ruta = 0
    historial_movimientos = [posicion_cursor]

    # Función para actualizar la información mostrada
    def actualizar_info():
        # Encontrar en qué rutas está la posición actual
        rutas_actuales = []
        if posicion_cursor in celdas_rutas:
            for idx_ruta, pos_ruta in celdas_rutas[posicion_cursor]:
                rutas_actuales.append(f"Ruta {idx_ruta + 1}")

        if rutas_actuales:
            info_ruta.set_text(f"Posición actual en: {', '.join(rutas_actuales)}")
        else:
            info_ruta.set_text("Posición actual: Fuera de ruta")

        info_pasos.set_text(f"Pasos dados: {pasos_dados}")
        fig.canvas.draw_idle()

    # Inicializar la información
    actualizar_info()

    # Función para manejar eventos de teclado
    def on_key(event):
        nonlocal posicion_cursor, pasos_dados, historial_movimientos

        # Obtener la posición actual
        y, x = posicion_cursor

        # Mover según la tecla presionada
        if event.key == 'up':
            nueva_pos = (y - 1, x)
        elif event.key == 'down':
            nueva_pos = (y + 1, x)
        elif event.key == 'left':
            nueva_pos = (y, x - 1)
        elif event.key == 'right':
            nueva_pos = (y, x + 1)
        elif event.key == 'r':  # Reiniciar
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
            return

        # Verificar si la nueva posición es válida (dentro del laberinto y es un camino)
        if (0 <= nueva_pos[0] < laberinto.shape[0] and
                0 <= nueva_pos[1] < laberinto.shape[1] and
                laberinto[nueva_pos] == 0):

            # Verificar si la nueva posición está adyacente a la posición actual
            if abs(nueva_pos[0] - y) + abs(nueva_pos[1] - x) == 1:
                # Mover el cursor
                posicion_cursor = nueva_pos
                cursor.center = (posicion_cursor[1], posicion_cursor[0])

                # Incrementar contador de pasos solo si es una nueva posición
                if nueva_pos not in historial_movimientos:
                    pasos_dados += 1

                # Añadir al historial
                historial_movimientos.append(posicion_cursor)

                # Actualizar información
                actualizar_info()

    # Conectar el evento de teclado
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.title("Control manual del cursor - Usa las teclas de dirección")
    plt.tight_layout()
    plt.show()


# Función para intentar generar un laberinto con múltiples rutas
def generar_laberinto_con_multiples_rutas(dim, num_rutas_deseadas, max_intentos=5):
    for intento in range(max_intentos):
        print(f"\n🔄 Intento {intento + 1}/{max_intentos} de generar laberinto con {num_rutas_deseadas} rutas...")

        # Generar laberinto con más paredes eliminadas para aumentar posibilidades
        laberinto = crear_maze(dim, num_rutas_deseadas + 2)
        rutas = encontrar_multiples_paths(laberinto, num_rutas_deseadas)

        if len(rutas) >= num_rutas_deseadas:
            print(f"✅ Éxito! Se encontraron {len(rutas)} rutas diferentes.")
            return laberinto, rutas

        print(f"⚠️ Solo se encontraron {len(rutas)} rutas. Intentando de nuevo...")

    # Si después de varios intentos no se logra, usar el último resultado
    print(f"⚠️ Después de {max_intentos} intentos, solo se pudieron generar {len(rutas)} rutas.")
    return laberinto, rutas


# Bloque principal modificado
if __name__ == "__main__":
    while True:
        dim = 10
        num_rutas_deseadas = 3

        # Usar la nueva función para generar un laberinto con múltiples rutas
        laberinto, rutas = generar_laberinto_con_multiples_rutas(dim, num_rutas_deseadas)

        # Encontrar la ruta más corta
        indice_mas_corto, longitudes = encontrar_ruta_mas_corta(rutas)

        # Mostrar información sobre todas las rutas
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

        # Primero mostrar el laberinto con todas las rutas
        dibujar_maze(laberinto, rutas, indice_mas_corto, longitudes)

        # Luego mostrar el control manual del cursor
        print("\n🎮 Iniciando control manual del cursor...")
        print("Usa las teclas de dirección (↑ ↓ ← →) para mover el cursor por el laberinto.")
        print("Presiona 'r' para reiniciar y 'q' para salir.")
        control_manual_cursor(laberinto, rutas, indice_mas_corto, longitudes)

        seguir = input("¿Quieres generar otro laberinto? (s/n): ")
        if seguir.lower() != 's':
            print("👋 Saliendo del generador de laberintos. ¡Hasta la próxima!")
            break

# Para demostración, ejecutemos una vez sin el bucle interactivo
dim = 10
num_rutas_deseadas = 3
print("\n🔄 Generando laberinto de dimension 10x10 con rutas diferentes...")

# Usar la nueva función para generar un laberinto con múltiples rutas
laberinto, rutas = generar_laberinto_con_multiples_rutas(dim, num_rutas_deseadas)

# Encontrar la ruta más corta
indice_mas_corto, longitudes = encontrar_ruta_mas_corta(rutas)

# Mostrar información sobre todas las rutas
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

# Primero mostrar el laberinto con todas las rutas
dibujar_maze(laberinto, rutas, indice_mas_corto, longitudes)

# Luego mostrar el control manual del cursors
print("\n🎮 Iniciando control manual del cursor...")
print("Usa las teclas de dirección (↑ ↓ ← →) para mover el cursor por el laberinto.")
print("Presiona 'r' para reiniciar y 'q' para salir.")
control_manual_cursor(laberinto, rutas, indice_mas_corto, longitudes)