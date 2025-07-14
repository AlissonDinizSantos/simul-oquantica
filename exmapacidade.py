# -*- coding: utf-8 -*-
"""
exmapacidade

Este módulo demonstra como buscar melhores rotas de um local até o destino evitando congestionamentos.
Ao modelar o mapa da cidade e aplicar algoritmos de busca, é possível encontrar caminhos mais eficientes,
desviando de bloqueios e áreas congestionadas.

Arquivo original: https://colab.research.google.com/drive/1pIRJ_7dWlhcVKHFS8IL_d0WU9B4v61NY
"""

import collections
import matplotlib.pyplot as plt
import numpy as np

# --- INÍCIO DA SEÇÃO DE MODIFICAÇÃO ---

# Definição do nosso "mapa da cidade"
# '#' representa congestionamento/bloqueio
# ' ' representa uma rua livre
# 'E' é o ponto de Partida (Entrada)
# 'S' é o Destino (Saída)
mapa_cidade = [
    ['E', ' ', ' ', '#', ' ', ' ', ' ', ' ', ' ', '#'],
    ['#', '#', ' ', '#', ' ', '#', '#', '#', ' ', ' '],
    [' ', ' ', ' ', '#', ' ', ' ', ' ', '#', '#', ' '],
    [' ', '#', '#', '#', ' ', '#', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', '#', ' ', '#', '#', ' '],
    ['#', '#', '#', ' ', '#', '#', '#', '#', ' ', '#'],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', '#', '#', '#', '#', '#', ' ', '#', '#', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', '#', 'S', ' '],
]

# Função que encontra a melhor rota no mapa usando Busca em Largura (BFS)
def encontrar_melhor_rota(mapa):
    """
    Resolve um mapa da cidade usando Busca em Largura (BFS) e retorna:
    - A rota ideal da partida até o destino
    - As rotas descartadas (tentativas de caminhos explorados)
    - O mapa original
    - Instantâneos dos cruzamentos explorados durante a busca (para plotagem)
    """
    linhas = len(mapa)  # Obtém o número de linhas do mapa
    colunas = len(mapa[0])  # Obtém o número de colunas do mapa

    # Encontrando as posições da partida ('E') e do destino ('S')
    partida = None
    destino = None
    for r in range(linhas):  # Percorrendo as linhas
        for c in range(colunas):  # Percorrendo as colunas
            if mapa[r][c] == 'E':  # Se encontrar a partida
                partida = (r, c)
            elif mapa[r][c] == 'S':  # Se encontrar o destino
                destino = (r, c)

    if not partida or not destino:  # Verifica se a partida ou destino não foi encontrado
        print("Erro: Ponto de partida ou destino não encontrados no mapa.")  # Mensagem de erro
        return None, [], mapa, []  # Retorna erro

    # Fila de BFS com a posição da partida e a rota atual
    fila = collections.deque([(partida[0], partida[1], [partida])])

    # Conjunto para armazenar os cruzamentos já visitados e evitar ciclos
    visitados = set([partida])

    rota_ideal = None
    todos_segmentos_explorados = []  # Lista para armazenar todas as rotas exploradas

    # Lista para capturar os estados dos cruzamentos visitados ao longo da busca
    exploration_snapshots = []
    snapshot_interval = 5  # Intervalo para capturar instantâneos

    step_counter = 0  # Contador de passos para controle do intervalo

    while fila:
        r_atual, c_atual, rota_atual = fila.popleft()  # Pega o próximo cruzamento da fila

        todos_segmentos_explorados.append(rota_atual)  # Armazena a rota atual explorada

        # Captura de instantâneos para visualização da busca
        if step_counter % snapshot_interval == 0 or (r_atual, c_atual) == destino:
            exploration_snapshots.append(set(visitados))
        step_counter += 1

        # Verifica se chegou ao destino
        if (r_atual, c_atual) == destino:
            rota_ideal = rota_atual
            if set(visitados) not in exploration_snapshots:
                 exploration_snapshots.append(set(visitados))  # Captura o último estado
            break  # Interrompe o loop

        # Direções possíveis: (norte, sul, leste, oeste)
        direcoes = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in direcoes:
            r_novo, c_novo = r_atual + dr, c_atual + dc

            # Verifica se a nova posição é válida e não visitada
            if (0 <= r_novo < linhas and 0 <= c_novo < colunas and
                mapa[r_novo][c_novo] != '#' and  # O cruzamento não deve ser um congestionamento
                (r_novo, c_novo) not in visitados):

                visitados.add((r_novo, c_novo))
                nova_rota = list(rota_atual)
                nova_rota.append((r_novo, c_novo))
                fila.append((r_novo, c_novo, nova_rota))

    # Filtra as rotas descartadas (que não levaram ao destino)
    rotas_descartadas = []
    if rota_ideal:
        rota_ideal_set = set(rota_ideal)
        for path_segment in todos_segmentos_explorados:
            is_wrong_path_segment = False
            for point in path_segment:
                if point not in rota_ideal_set:
                    is_wrong_path_segment = True
                    break

            if is_wrong_path_segment and path_segment != rota_ideal:
                rotas_descartadas.append(path_segment)
    else:
        rotas_descartadas = [path for path in todos_segmentos_explorados if len(path) > 1]

    return rota_ideal, rotas_descartadas, mapa, exploration_snapshots


# Função para plotar um instantâneo do mapa, mostrando a área explorada
def plotar_mapa_snapshot(mapa, visited_cells, step_title, partida_destino_coords):
    """
    Plota um instantâneo do mapa, mostrando os cruzamentos visitados até o momento.
    """
    linhas = len(mapa)
    colunas = len(mapa[0])
    partida, destino = partida_destino_coords

    map_matrix = np.zeros((linhas, colunas))

    # Preenche a matriz com valores para congestionamentos, partida e destino
    for r in range(linhas):
        for c in range(colunas):
            if mapa[r][c] == '#':  # Se for um congestionamento
                map_matrix[r][c] = 1
            elif (r, c) == partida:
                map_matrix[r][c] = 2
            elif (r, c) == destino:
                map_matrix[r][c] = 3
            elif (r, c) in visited_cells:  # Se o cruzamento foi visitado
                map_matrix[r][c] = 4
            else:
                map_matrix[r][c] = 0

    # Definir as cores para o gráfico
    cmap = plt.cm.colors.ListedColormap(['white', 'black', 'blue', 'darkgreen', 'cyan'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(colunas * 0.6, linhas * 0.6))
    ax.imshow(map_matrix, cmap=cmap, norm=norm)

    ax.set_xticks(np.arange(-.5, colunas, 1), minor=True)
    ax.set_yticks(np.arange(-.5, linhas, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", size=0)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title(f"Busca pela Melhor Rota (Passo: {step_title})")

    # Adiciona a legenda
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', label='Rua Livre', markerfacecolor='white', markersize=10),
        plt.Line2D([0], [0], marker='s', color='w', label='Congestionamento', markerfacecolor='black', markersize=10),
        plt.Line2D([0], [0], marker='s', color='w', label='Partida', markerfacecolor='blue', markersize=10),
        plt.Line2D([0], [0], marker='s', color='w', label='Destino', markerfacecolor='darkgreen', markersize=10),
        plt.Line2D([0], [0], marker='s', color='w', label='Área Explorada', markerfacecolor='cyan', markersize=10)
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout()
    plt.show()
    plt.close(fig)

# Função para plotar o mapa final com a rota ideal
def plotar_mapa_final(mapa, rota_ideal=None, rotas_descartadas=None, final_visited_cells=None):
    """
    Plota o mapa final, destacando a rota ideal e as rotas descartadas.
    """
    linhas = len(mapa)
    colunas = len(mapa[0])

    map_matrix = np.zeros((linhas, colunas))

    for r in range(linhas):
        for c in range(colunas):
            if mapa[r][c] == '#':
                map_matrix[r][c] = 1  # Congestionamento
            elif mapa[r][c] == 'E':
                map_matrix[r][c] = 2  # Partida
            elif mapa[r][c] == 'S':
                map_matrix[r][c] = 3  # Destino
            else:
                map_matrix[r][c] = 0  # Rua Livre

    # Preenche as células exploradas (vermelho) e a rota ideal (verde claro)
    if final_visited_cells:
        for r, c in final_visited_cells:
            if map_matrix[r][c] == 0:
                map_matrix[r][c] = 5  # Rota descartada

    if rota_ideal:
        for r, c in rota_ideal:
            # Pinta por cima da área explorada e da rua livre
            if map_matrix[r][c] == 0 or map_matrix[r][c] == 5:
                map_matrix[r][c] = 4  # Rota ideal

    # Definir o mapa de cores
    cmap = plt.cm.colors.ListedColormap(['white', 'black', 'blue', 'darkgreen', 'lightgreen', 'red'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(colunas * 0.6, linhas * 0.6))
    ax.imshow(map_matrix, cmap=cmap, norm=norm)

    ax.set_xticks(np.arange(-.5, colunas, 1), minor=True)
    ax.set_yticks(np.arange(-.5, linhas, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", size=0)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title("Mapa Final com a Melhor Rota")

    # Adiciona a legenda para o gráfico final
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', label='Rua Livre', markerfacecolor='white', markersize=10),
        plt.Line2D([0], [0], marker='s', color='w', label='Congestionamento', markerfacecolor='black', markersize=10),
        plt.Line2D([0], [0], marker='s', color='w', label='Partida', markerfacecolor='blue', markersize=10),
        plt.Line2D([0], [0], marker='s', color='w', label='Destino', markerfacecolor='darkgreen', markersize=10),
        plt.Line2D([0], [0], marker='s', color='w', label='Rota Ideal', markerfacecolor='lightgreen', markersize=10),
        plt.Line2D([0], [0], marker='s', color='w', label='Rota Descartada', markerfacecolor='red', markersize=10)
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout()
    plt.show()
    plt.close(fig)

# --- FIM DA SEÇÃO DE MODIFICAÇÃO ---


# --- CÓDIGO DE EXECUÇÃO ---

# Encontrando coordenadas de partida e destino para a plotagem
partida_coords = None
destino_coords = None
for r in range(len(mapa_cidade)):
    for c in range(len(mapa_cidade[0])):
        if mapa_cidade[r][c] == 'E':
            partida_coords = (r, c)
        elif mapa_cidade[r][c] == 'S':
            destino_coords = (r, c)

partida_destino_coords = (partida_coords, destino_coords)

# Encontrar a melhor rota e obter os instantâneos de cada etapa
rota_ideal, rotas_descartadas_encontradas, mapa_original, exploration_snapshots = encontrar_melhor_rota(mapa_cidade)

# Gerar os gráficos de demonstração da busca pela rota
for i, visited_set_at_step in enumerate(exploration_snapshots):
    plotar_mapa_snapshot(mapa_original, visited_set_at_step, i + 1, partida_destino_coords)

# Plotar a solução final
if rota_ideal:
    # O último snapshot contém todas as células visitadas
    celulas_visitadas_final = exploration_snapshots[-1] if exploration_snapshots else set()
    plotar_mapa_final(mapa_original, rota_ideal, rotas_descartadas_encontradas, celulas_visitadas_final)
else:
    print("Não foi possível encontrar uma rota da partida ao destino.")