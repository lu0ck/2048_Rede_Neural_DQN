import os  # Módulo para operações com o sistema de arquivos
import pygame  # Biblioteca para criação de jogos e interfaces gráficas
import random  # Módulo para geração de números aleatórios
import numpy as np  # Biblioteca para operações numéricas e arrays
import torch  # Framework de aprendizado profundo
import torch.nn as nn  # Módulo de redes neurais do PyTorch
import torch.optim as optim  # Otimizadores do PyTorch
import torch.nn.functional as F  # Funções de ativação e utilitárias do PyTorch
from collections import deque  # Estrutura de dados para fila de dupla extremidade
import platform  # Módulo para informações sobre a plataforma
import asyncio  # Suporte para programação assíncrona
import logging  # Módulo para logging de eventos
import math  # Funções matemáticas
import tkinter as tk  # Interface gráfica para diálogos de arquivos
from tkinter import filedialog  # Diálogo para seleção de arquivos

# -------------------------
# Config / inicialização
# -------------------------
# Configura o logging para registrar eventos com nível INFO
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Cria a pasta 'models' se não existir para armazenar modelos salvos
os.makedirs("models", exist_ok=True)

# Inicializa o Pygame para gerenciamento de tela e eventos
pygame.init()
SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 720  # Dimensões da tela
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)  # Cria a janela redimensionável
pygame.display.set_caption("2048 - Treinamento Visual")  # Define o título da janela
clock = pygame.time.Clock()  # Relógio para controle de FPS

# Define fontes para texto na interface
font = pygame.font.SysFont("Arial", 26, bold=True)  # Fonte principal
small_font = pygame.font.SysFont("Arial", 16)  # Fonte pequena
title_font = pygame.font.SysFont("Arial", 20, bold=True)  # Fonte para títulos

# -------------------------
# Parâmetros do sistema
# -------------------------
GRID_SIZE = 4  # Tamanho da grade do jogo 2048 (4x4)
INITIAL_AGENTS = 100  # Número inicial de agentes (redes neurais)
TOP_AGENTS = 50  # Número de melhores agentes a serem selecionados (não usado explicitamente)
FPS = 30  # Taxa de frames por segundo inicial
GRAPH_HEIGHT = 220  # Altura do gráfico principal
MAX_GENERATIONS = 1000  # Número máximo de gerações de treinamento
INFO_PANEL_WIDTH = 350  # Largura do painel de informações (aumentada para caber textos)
AGENT_PANEL_WIDTH = 280  # Largura do painel do agente
AGENT_PANEL_HEIGHT = 280  # Altura do painel do agente
NEURAL_PANEL_WIDTH = 300  # Largura do painel da rede neural
NEURAL_PANEL_HEIGHT = 280  # Altura do painel da rede neural
MINI_GRAPH_WIDTH = 250  # Largura do mini gráfico de scores médios
MINI_GRAPH_HEIGHT = 140  # Altura do mini gráfico de scores médios
MIN_SPACING = 20  # Espaçamento mínimo entre elementos

# Posições ajustadas para elementos da UI
POS_TOP_BAR = (10, 10)  # Posição da barra superior com botões
POS_GRAPH = (10, 80)  # Posição do gráfico principal
POS_AGENT = (10, SCREEN_HEIGHT - AGENT_PANEL_HEIGHT - MIN_SPACING)  # Posição do painel do melhor agente
POS_HISTORY = (POS_AGENT[0] + AGENT_PANEL_WIDTH + MIN_SPACING, POS_AGENT[1])  # Posição do histórico de movimentos
POS_PIE = (POS_HISTORY[0] + 220 + MIN_SPACING, POS_AGENT[1])  # Posição do gráfico de pizza
POS_INFO = (SCREEN_WIDTH - INFO_PANEL_WIDTH - MIN_SPACING, POS_AGENT[1])  # Posição do painel de informações
POS_NEURAL = (SCREEN_WIDTH - NEURAL_PANEL_WIDTH - MIN_SPACING, POS_GRAPH[1])  # Posição do painel da rede neural
POS_MINI_GRAPH = (POS_NEURAL[0] - MINI_GRAPH_WIDTH - MIN_SPACING, POS_NEURAL[1])  # Posição do mini gráfico (ao lado da rede neural)

# -------------------------
# Cores (novo tema tecnológico)
# -------------------------
# Dicionário de cores para o tema tecnológico
COLORS = {
    'bg': (10, 15, 20),  # Fundo escuro com tom azulado
    'panel': (25, 30, 35),  # Painéis com cinza escuro
    'panel_border': (0, 255, 255, 100),  # Borda ciano com transparência
    'up_candle': (0, 255, 128),  # Verde neon para candles ascendentes
    'down_candle': (255, 64, 64),  # Vermelho neon para candles descendentes
    'graph_text': (200, 255, 255),  # Texto ciano claro
    'info_text': (180, 220, 255),  # Texto ciano suave
    'title_text': (100, 200, 255),  # Títulos em azul claro
    'axis': (80, 100, 120),  # Eixos em cinza azulado
    'tile_colors': {  # Cores para as tiles do jogo 2048
        0: (50, 50, 50), 2: (100, 150, 200), 4: (80, 180, 220), 8: (60, 200, 240),
        16: (40, 220, 255), 32: (20, 240, 255), 64: (0, 255, 255), 128: (0, 200, 255),
        256: (0, 180, 255), 512: (0, 160, 255), 1024: (0, 140, 255), 2048: (0, 120, 255)
    },
    'pie_colors': [(0, 255, 255), (255, 128, 0), (128, 0, 255), (255, 255, 0)],  # Cores vibrantes para pizza
    'button_bg': (30, 40, 50),  # Fundo dos botões
    'button_border': (0, 255, 255, 150),  # Borda ciano dos botões
    'button_hover': (50, 70, 90),  # Cor de hover dos botões
    'button_text': (200, 255, 255),  # Texto dos botões
    'progress_bg': (40, 50, 60),  # Fundo da barra de progresso
    'progress_fill': (0, 255, 200),  # Preenchimento da barra de progresso
    'neural_node': (0, 200, 255),  # Nós da rede neural
    'neural_edge': (0, 255, 255, 100),  # Conexões da rede
    'neural_active': (255, 255, 0),  # Nós ativos
    'neural_blink': (255, 100, 0)  # Cor para piscar nos nós de entrada
}

# -------------------------
# Rede neural (DQN)
# -------------------------
# Definição da classe DQN, uma rede neural simples para Q-Learning
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 16)  # Camada fully connected de entrada para oculta
        self.out = nn.Linear(16, 4)  # Camada fully connected oculta para saída (4 ações)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Ativação ReLU na camada oculta
        return self.out(x)  # Saída linear

# -------------------------
# Funções do jogo 2048
# -------------------------
# Cria uma grade vazia para o jogo
def create_grid(): return [[0] * GRID_SIZE for _ in range(GRID_SIZE)]

# Adiciona uma nova tile (2 ou 4) em uma posição aleatória vazia
def add_new_tile(grid):
    empty = [(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE) if grid[i][j] == 0]
    if empty:
        i, j = random.choice(empty)
        grid[i][j] = 2 if random.random() < 0.9 else 4

# Comprime uma linha, fundindo tiles iguais e calculando recompensa
def compress_line(line):
    new_line = [n for n in line if n != 0]
    i = 0
    reward = 0
    while i < len(new_line) - 1:
        if new_line[i] == new_line[i + 1]:
            new_line[i] *= 2
            reward += new_line[i]
            new_line.pop(i + 1)
        i += 1
    return new_line + [0] * (GRID_SIZE - len(new_line)), reward

# Movimenta a grade para a esquerda
def move_left(grid):
    new_grid = [compress_line(row)[0] for row in grid]
    total_reward = sum(compress_line(row)[1] for row in grid)
    return new_grid, total_reward

# Movimenta a grade para a direita (inverte linhas)
def move_right(grid):
    new_grid = [compress_line(row[::-1])[0][::-1] for row in grid]
    total_reward = sum(compress_line(row[::-1])[1] for row in grid)
    return new_grid, total_reward

# Movimenta a grade para cima (transpõe e move esquerda)
def move_up(grid):
    t = list(map(list, zip(*grid)))
    new_t = [compress_line(row)[0] for row in t]
    total_reward = sum(compress_line(row)[1] for row in t)
    return list(map(list, zip(*new_t))), total_reward

# Movimenta a grade para baixo (transpõe e move direita)
def move_down(grid):
    t = list(map(list, zip(*grid)))
    new_t = [compress_line(row[::-1])[0][::-1] for row in t]
    total_reward = sum(compress_line(row[::-1])[1] for row in t)
    return list(map(list, zip(*new_t))), total_reward

# Verifica se duas grades são iguais
def grids_equal(a, b): return all(r1 == r2 for r1, r2 in zip(a, b))

# Verifica se o jogo terminou (sem movimentos possíveis)
def is_game_over(grid):
    if any(0 in row for row in grid): return False
    for fn in (move_left, move_right, move_up, move_down):
        new_grid, _ = fn(grid)
        if not grids_equal(grid, new_grid): return False
    return True

# Obtém o valor máximo de tile na grade
def get_max_tile(grid):
    maxv = 0
    for row in grid:
        for v in row:
            if v > maxv: maxv = v
    return maxv

# Converte a grade para entrada da rede neural (log2 dos valores)
def board_to_input(grid):
    return np.array([0 if v == 0 else np.log2(v) for row in grid for v in row], dtype=np.float32)

# -------------------------
# UI utilitários
# -------------------------
# Desenha um botão com hover e texto
def draw_button(surface, rect, text, hovering=False):
    color = COLORS['button_hover'] if hovering else COLORS['button_bg']
    pygame.draw.rect(surface, color, rect, border_radius=8)
    pygame.draw.rect(surface, COLORS['button_border'], rect, 2, border_radius=8)
    label = small_font.render(text, True, COLORS['button_text'])
    surface.blit(label, label.get_rect(center=rect.center))

# Verifica se o botão foi clicado
def button_clicked(rect, mx, my, mb):
    return rect.collidepoint(mx, my) and mb[0]

# Desenha uma barra de progresso
def draw_progress(surface, x, y, w, h, fraction):
    pygame.draw.rect(surface, COLORS['progress_bg'], (x, y, w, h), border_radius=8)
    inner_w = max(2, int(w * fraction))
    pygame.draw.rect(surface, COLORS['progress_fill'], (x+2, y+2, inner_w-4, h-4), border_radius=8)

# Desenha um mini gráfico de linha para scores médios
def draw_mini_line(surface, data, x, y, w, h, color=COLORS['progress_fill']):
    pygame.draw.rect(surface, COLORS['panel'], (x, y, w, h))
    pygame.draw.rect(surface, COLORS['panel_border'], (x, y, w, h), 1)
    title = title_font.render("Scores Médios", True, COLORS['title_text'])
    surface.blit(title, (x + 10, y - 25))
    if not data:
        return
    mx = max(data) if max(data) > 0 else 1
    points = []
    for i, v in enumerate(data[-(w//4):]):
        px = x + int(i * (w / max(1, len(data[-(w//4):]) - 1)))
        py = y + h - int((v / mx) * (h - 4)) - 2
        points.append((px, py))
    if len(points) > 1:
        pygame.draw.lines(surface, color, False, points, 2)

# Calcula pontos para um setor de pizza
def sector_points(cx, cy, radius, start_deg, end_deg, steps=40):
    pts = [(cx, cy)]
    for d in np.linspace(start_deg, end_deg, max(2, steps)):
        rad = math.radians(d)
        pts.append((cx + radius * math.cos(rad), cy + radius * math.sin(rad)))
    return pts

# Desenha a representação visual da rede neural com animação baseada na grade
def draw_neural_network(surface, x, y, w, h, grid, last_action=None, frame_count=0):
    pygame.draw.rect(surface, COLORS['panel'], (x, y, w, h))
    pygame.draw.rect(surface, COLORS['panel_border'], (x, y, w, h), 1)
    
    title = title_font.render("Rede Neural (DQN)", True, COLORS['title_text'])
    surface.blit(title, (x + 10, y - 25))
    
    # Estrutura da rede: 16 (entrada) -> 16 (oculta) -> 4 (saída)
    layer_sizes = [16, 16, 4]
    node_spacing_v = h // (max(layer_sizes) + 1)
    layer_spacing_h = w // (len(layer_sizes) + 1)
    
    nodes = []
    for i, size in enumerate(layer_sizes):
        layer_x = x + (i + 1) * layer_spacing_h
        layer_nodes = []
        for j in range(size):
            node_y = y + (j + 1) * node_spacing_v
            layer_nodes.append((layer_x, node_y))
        nodes.append(layer_nodes)
    
    # Desenhar conexões entre camadas
    for i in range(len(nodes) - 1):
        for n1 in nodes[i]:
            for n2 in nodes[i + 1]:
                pygame.draw.line(surface, COLORS['neural_edge'], n1, n2, 1)
    
    # Desenhar nós com animação
    for i, layer in enumerate(nodes):
        if i == 0:  # Camada de entrada: piscar se tile > 0
            for j, node in enumerate(layer):
                row, col = divmod(j, GRID_SIZE)
                value = grid[row][col]
                if value > 0:
                    blink = (frame_count // 5) % 2 == 0
                    color = COLORS['neural_blink'] if blink else COLORS['neural_active']
                    radius = 10 if blink else 8
                else:
                    color = COLORS['neural_node']
                    radius = 8
                pygame.draw.circle(surface, color, node, radius)
        elif i == 1:  # Camada oculta: piscar levemente
            for node in layer:
                blink = (frame_count // 10) % 2 == 0
                color = COLORS['neural_active'] if blink else COLORS['neural_node']
                pygame.draw.circle(surface, color, node, 8)
        elif i == 2:  # Camada de saída: destacar ação última
            for j, node in enumerate(layer):
                color = COLORS['neural_active'] if j == last_action else COLORS['neural_node']
                pygame.draw.circle(surface, color, node, 8)

# -------------------------
# Funções de desenho principais
# -------------------------
# Desenha o gráfico de candles para evolução de scores
def draw_graph(screen, scores_history, width, height, x_offset, y_offset):
    pygame.draw.rect(screen, COLORS['panel'], (x_offset, y_offset, width, height))
    pygame.draw.rect(screen, COLORS['panel_border'], (x_offset, y_offset, width, height), 1)
    if len(scores_history) < 2:
        title_text = font.render("Crescimento de Pontuação", True, COLORS['title_text'])
        screen.blit(title_text, (x_offset + 10, y_offset - 30))
        return

    max_score = max(scores_history) or 1
    intervals = max(1, len(scores_history) - 1)
    candle_width = max(3, width // intervals)

    pygame.draw.line(screen, COLORS['axis'], (x_offset, y_offset), (x_offset, y_offset + height), 2)
    pygame.draw.line(screen, COLORS['axis'], (x_offset, y_offset + height), (x_offset + width, y_offset + height), 2)

    for i in range(5):
        val = int(max_score * i / 4)
        y = y_offset + height - int((val / max_score) * (height - 20))
        y = max(y_offset, min(y_offset + height, y))
        pygame.draw.line(screen, COLORS['axis'], (x_offset, y), (x_offset + width, y), 1)
        label = small_font.render(str(val), True, COLORS['graph_text'])
        screen.blit(label, (x_offset - 50, y - 8))

    step_x = max(1, intervals // 10)
    for i in range(0, len(scores_history), step_x):
        x = x_offset + i * candle_width
        if x < x_offset or x > x_offset + width: continue
        label = small_font.render(str(i), True, COLORS['graph_text'])
        screen.blit(label, (x, y_offset + height + 5))

    title_text = font.render("Crescimento de Pontuação", True, COLORS['title_text'])
    screen.blit(title_text, (x_offset + 10, y_offset - 30))

    for i in range(len(scores_history) - 1):
        current = scores_history[i]
        nxt = scores_history[i + 1]
        y_current = y_offset + height - int((current / max_score) * (height - 20))
        y_next = y_offset + height - int((nxt / max_score) * (height - 20))
        y_current = max(y_offset, min(y_offset + height, y_current))
        y_next = max(y_offset, min(y_offset + height, y_next))
        x = x_offset + i * candle_width
        color = COLORS['up_candle'] if nxt >= current else COLORS['down_candle']
        rect = pygame.Rect(x, min(y_current, y_next), max(1, candle_width - 2), max(2, abs(y_next - y_current)))
        pygame.draw.rect(screen, color, rect)
        pygame.draw.line(screen, color, (x + candle_width // 2, y_current), (x + candle_width // 2, y_next), 2)

# Desenha a grade do melhor agente
def draw_agent_grid(screen, grid, x_offset, y_offset, tile_size=65):
    bg = COLORS['panel']
    pygame.draw.rect(screen, bg, (x_offset - 5, y_offset - 5, tile_size * GRID_SIZE + 10, tile_size * GRID_SIZE + 10))
    pygame.draw.rect(screen, COLORS['panel_border'], (x_offset - 5, y_offset - 5, tile_size * GRID_SIZE + 10, tile_size * GRID_SIZE + 10), 1)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            value = int(grid[i][j])
            rect = pygame.Rect(x_offset + j * tile_size, y_offset + i * tile_size, tile_size - 5, tile_size - 5)
            pygame.draw.rect(screen, COLORS['tile_colors'].get(value, COLORS['tile_colors'][0]), rect, border_radius=6)
            if value:
                txt = small_font.render(str(value), True, (0, 0, 0))
                txt_rect = txt.get_rect(center=rect.center)
                screen.blit(txt, txt_rect)

# Desenha o gráfico de pizza para contagem de ações
def draw_pie_chart(screen, action_counts, x_offset, y_offset, radius=60):
    pygame.draw.rect(screen, COLORS['panel'], (x_offset - radius - 10, y_offset - radius - 10, radius*2 + 180, radius*2 + 20))
    pygame.draw.rect(screen, COLORS['panel_border'], (x_offset - radius - 10, y_offset - radius - 10, radius*2 + 180, radius*2 + 20), 1)
    total = sum(action_counts.values())
    if total == 0:
        return
    start = 0.0
    actions = list(action_counts.keys())
    for i, action in enumerate(actions):
        count = action_counts[action]
        angle = 360.0 * (count / total)
        pts = sector_points(x_offset, y_offset, radius, start, start + angle, steps=30)
        pygame.draw.polygon(screen, COLORS['pie_colors'][i % len(COLORS['pie_colors'])], pts)
        start += angle

    legend_x = x_offset + radius + 10
    legend_y = y_offset - radius
    for i, action in enumerate(actions):
        pygame.draw.rect(screen, COLORS['pie_colors'][i % len(COLORS['pie_colors'])], (legend_x, legend_y + i * 20, 16, 12))
        label = small_font.render(f"{action}: {action_counts[action]}", True, COLORS['graph_text'])
        screen.blit(label, (legend_x + 20, legend_y + i * 20))

# Desenha o painel de informações gerais
def draw_info_panel(screen, font, current_episode, num_alive, max_tile_global, max_tile_gen, best_score_global, max_score_gen, moves_best, action_counts, top5_list, avg_score):
    info_text = [
        f"Indivíduos Vivos: {num_alive}/{INITIAL_AGENTS}",
        f"Geração: {current_episode + 1}",
        f"Maior Peça (Total): {max_tile_global}",
        f"Maior Peça (Geração): {max_tile_gen}",
        f"Maior Score (Total): {best_score_global}",
        f"Maior Score (Geração): {max_score_gen}",
        f"Movimentos Melhor: {moves_best}"
    ]
    pygame.draw.rect(screen, COLORS['panel'], (POS_INFO[0], POS_INFO[1], INFO_PANEL_WIDTH, 300))
    pygame.draw.rect(screen, COLORS['panel_border'], (POS_INFO[0], POS_INFO[1], INFO_PANEL_WIDTH, 300), 1)
    y_offset = POS_INFO[1] + 10
    for line in info_text:
        text = font.render(line, True, COLORS['info_text'])
        screen.blit(text, (POS_INFO[0] + 10, y_offset))
        y_offset += 30

    total = sum(action_counts.values())
    if total > 0:
        dominant = max(action_counts, key=action_counts.get)
        percentage = action_counts[dominant] / total
        strat_text = f"Estratégia dominante: {dominant} ({percentage:.0%})"
        text = small_font.render(strat_text, True, COLORS['info_text'])
        screen.blit(text, (POS_INFO[0] + 10, y_offset + 5))

    y_top = POS_INFO[1] + 200
    top_label = small_font.render("Top 5 Agentes:", True, COLORS['title_text'])
    screen.blit(top_label, (POS_INFO[0] + 10, y_top))
    y_top += 16
    for i, (idx, sc) in enumerate(top5_list[:5]):
        t = small_font.render(f"{i+1}. Agent {idx} — {sc}", True, COLORS['info_text'])
        screen.blit(t, (POS_INFO[0] + 15, y_top))
        y_top += 16

# Desenha o histórico de movimentos
def draw_movement_history(screen, history, x_offset, y_offset):
    pygame.draw.rect(screen, COLORS['panel'], (x_offset - 5, y_offset - 5, 200, 280))
    pygame.draw.rect(screen, COLORS['panel_border'], (x_offset - 5, y_offset - 5, 200, 280), 1)
    for i, move in enumerate(history[-18:]):
        text = small_font.render(f"Move {i+1}: {['Left','Right','Up','Down'][move]}", True, COLORS['info_text'])
        screen.blit(text, (x_offset + 5, y_offset + i * 14))

# -------------------------
# Train loop (assíncrono)
# -------------------------
# Função principal assíncrona para treinamento da população
async def train_population():
    # Inicializa agentes, targets, otimizadores e buffers
    agents = [DQN() for _ in range(INITIAL_AGENTS)]
    targets = [DQN() for _ in range(INITIAL_AGENTS)]
    for i in range(INITIAL_AGENTS):
        targets[i].load_state_dict(agents[i].state_dict())
    optimizers = [optim.Adam(agent.parameters(), lr=0.01) for agent in agents]
    buffers = [deque(maxlen=200) for _ in range(INITIAL_AGENTS)]

    # Históricos e parâmetros de treinamento
    scores_history = [0]
    avg_scores_history = []
    epsilon = 0.3  # Taxa de exploração inicial
    gamma = 0.95  # Fator de desconto
    batch_size = 4  # Tamanho do batch para treinamento
    best_score_global = 0
    max_tile_global = 0
    episode = 0
    frame_count = 0  # Contador para animação

    # Estados da UI
    fps = FPS
    manual_loaded_model = None
    pause = False

    # Definição dos botões na barra superior
    btn_w, btn_h = 140, 30
    btn_spacing = 10
    btn_y = POS_TOP_BAR[1]
    btn_newgen = pygame.Rect(POS_TOP_BAR[0], btn_y, btn_w, btn_h)
    btn_load = pygame.Rect(POS_TOP_BAR[0] + btn_w + btn_spacing, btn_y, btn_w, btn_h)
    btn_save = pygame.Rect(POS_TOP_BAR[0] + 2 * (btn_w + btn_spacing), btn_y, btn_w, btn_h)
    btn_pause = pygame.Rect(POS_TOP_BAR[0] + 3 * (btn_w + btn_spacing), btn_y, btn_w, btn_h)
    slider_rect = pygame.Rect(POS_TOP_BAR[0] + 4 * (btn_w + btn_spacing), btn_y + 4, 200, 10)
    slider_handle_x = slider_rect.x + int((fps / 60.0) * slider_rect.w)

    # Inicializa Tkinter para diálogos de arquivos (oculto)
    tk_root = tk.Tk()
    tk_root.withdraw()

    # Função para carregar modelo via diálogo
    def carregar_model_dialog():
        path = filedialog.askopenfilename(initialdir="models", title="Selecione um modelo (.pth)", filetypes=[("PyTorch", "*.pth")])
        if path:
            try:
                state = torch.load(path, map_location='cpu')
                mdl = DQN()
                mdl.load_state_dict(state)
                mdl.eval()
                logging.info(f"Modelo carregado: {path}")
                return mdl
            except Exception as e:
                logging.error(f"Erro ao carregar modelo: {e}")
        return None

    # Loop principal por gerações
    while episode < MAX_GENERATIONS:
        num_agents_current = len(agents)
        logging.info(f"Iniciando Geração {episode + 1} com {num_agents_current} agentes")

        # Prepara grades, estados e métricas para a geração
        grids = [create_grid() for _ in range(num_agents_current)]
        for g in grids:
            add_new_tile(g); add_new_tile(g)
        states = [board_to_input(g) for g in grids]
        dones = [False] * num_agents_current
        scores = [0] * num_agents_current
        moves_count = [0] * num_agents_current
        buffers = [buffers[i] if i < len(buffers) else deque(maxlen=200) for i in range(num_agents_current)]

        active_agents = list(range(num_agents_current))
        movement_history = []
        action_counts = {'Left': 0, 'Right': 0, 'Up': 0, 'Down': 0}

        # Loop enquanto houver agentes vivos ou não pausado
        while active_agents and not pause:
            frame_count += 1
            mx, my = pygame.mouse.get_pos()
            mb = pygame.mouse.get_pressed()

            # Processa eventos do Pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_c:
                        mdl = carregar_model_dialog()
                        if mdl:
                            manual_loaded_model = mdl
                            for i in range(len(agents)):
                                agents[i].load_state_dict(manual_loaded_model.state_dict())
                            logging.info("Modelo carregado aplicado a todos os agentes.")
                    if event.key == pygame.K_EQUALS or event.key == pygame.K_KP_PLUS:
                        fps = min(120, fps + 5)
                    if event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                        fps = max(1, fps - 5)
                    if event.key == pygame.K_SPACE:
                        pause = not pause
                        logging.info(f"Treinamento {'pausado' if pause else 'retomado'}")

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if button_clicked(btn_newgen, mx, my, mb):
                        logging.info("Botão: Nova Geração")
                        active_agents.clear()
                    if button_clicked(btn_load, mx, my, mb):
                        mdl = carregar_model_dialog()
                        if mdl:
                            manual_loaded_model = mdl
                            for i in range(len(agents)):
                                agents[i].load_state_dict(manual_loaded_model.state_dict())
                            logging.info("Modelo carregado via botão")
                    if button_clicked(btn_save, mx, my, mb):
                        if scores:
                            best_idx = int(np.argmax(scores))
                            save_name = f"models/manual_save_gen{episode+1}_agent{best_idx}.pth"
                            torch.save(agents[best_idx].state_dict(), save_name)
                            logging.info(f"Salvo modelo manual: {save_name}")
                    if button_clicked(btn_pause, mx, my, mb):
                        pause = not pause
                        logging.info(f"Treinamento {'pausado' if pause else 'retomado'}")

                if event.type == pygame.MOUSEMOTION:
                    if event.buttons[0] and slider_rect.collidepoint(event.pos):
                        rel = (event.pos[0] - slider_rect.x) / slider_rect.w
                        rel = max(0.0, min(1.0, rel))
                        fps = int(rel * 60) if int(rel * 60) >= 1 else 1

            # Se pausado, desenha a tela e aguarda
            if pause:
                screen.fill(COLORS['bg'])
                draw_graph(screen, scores_history, SCREEN_WIDTH - INFO_PANEL_WIDTH - NEURAL_PANEL_WIDTH - MINI_GRAPH_WIDTH - 3 * MIN_SPACING, GRAPH_HEIGHT, POS_GRAPH[0], POS_GRAPH[1])
                best_idx = max([idx for idx in range(len(scores)) if not dones[idx]], key=lambda k: scores[k], default=int(np.argmax(scores)))
                max_score_gen = scores[best_idx] if scores else 0
                moves_best = moves_count[best_idx] if best_idx < len(moves_count) else 0
                ranked = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)
                top5 = ranked[:5]
                num_alive = len(active_agents)
                avg_score = np.mean(scores) if scores else 0
                draw_info_panel(screen, font, episode, num_alive, max_tile_global, max(get_max_tile(g) for g in grids) if grids else 0, best_score_global, max_score_gen, moves_best, action_counts, top5, avg_scores_history)
                agent_title = title_font.render("Melhor Agente", True, COLORS['title_text'])
                screen.blit(agent_title, (POS_AGENT[0], POS_AGENT[1] - 25))
                draw_agent_grid(screen, grids[best_idx], POS_AGENT[0], POS_AGENT[1])
                history_title = title_font.render("Histórico de Movimentos", True, COLORS['title_text'])
                screen.blit(history_title, (POS_HISTORY[0], POS_HISTORY[1] - 25))
                draw_movement_history(screen, movement_history, POS_HISTORY[0], POS_HISTORY[1])
                draw_pie_chart(screen, action_counts, POS_PIE[0] + 70, POS_PIE[1] + 70)
                last_action = movement_history[-1] if movement_history else None
                draw_neural_network(screen, POS_NEURAL[0], POS_NEURAL[1], NEURAL_PANEL_WIDTH, NEURAL_PANEL_HEIGHT, grids[best_idx], last_action, frame_count)
                draw_mini_line(screen, avg_scores_history, POS_MINI_GRAPH[0], POS_MINI_GRAPH[1], MINI_GRAPH_WIDTH, MINI_GRAPH_HEIGHT)
                prog_x = POS_INFO[0] + 10
                prog_y = POS_INFO[1] - 25
                draw_progress(screen, prog_x, prog_y, INFO_PANEL_WIDTH - 20, 10, (num_alive / max(1, INITIAL_AGENTS)))
                hovering_new = btn_newgen.collidepoint(mx, my)
                hovering_load = btn_load.collidepoint(mx, my)
                hovering_save = btn_save.collidepoint(mx, my)
                hovering_pause = btn_pause.collidepoint(mx, my)
                draw_button(screen, btn_newgen, "Nova Geração", hovering_new)
                draw_button(screen, btn_load, "Carregar Modelo", hovering_load)
                draw_button(screen, btn_save, "Salvar Modelo", hovering_save)
                draw_button(screen, btn_pause, "Pausar" if not pause else "Retomar", hovering_pause)
                pygame.draw.rect(screen, COLORS['panel'], slider_rect, border_radius=5)
                slider_handle_x = slider_rect.x + int((fps / 60.0) * slider_rect.w)
                pygame.draw.circle(screen, COLORS['button_bg'], (slider_handle_x, slider_rect.y + slider_rect.h // 2), 6)
                speed_label = small_font.render(f"Velocidade: {fps} FPS", True, COLORS['graph_text'])
                screen.blit(speed_label, (slider_rect.x, slider_rect.y - 20))
                pygame.display.flip()
                await asyncio.sleep(1.0 / max(1, fps))
                continue

            # Simula um passo para cada agente ativo
            for i in active_agents.copy():
                action = choose_action(agents[i], states[i], epsilon=epsilon)
                next_grid, reward, done = step(grids[i], action)
                next_state = board_to_input(next_grid)
                buffers[i].append((states[i], action, reward, next_state, done))

                scores[i] += reward
                states[i] = next_state
                grids[i] = next_grid
                dones[i] = done
                moves_count[i] += 1

                best_alive = max(active_agents, key=lambda k: scores[k]) if active_agents else None
                if best_alive is not None and i == best_alive:
                    movement_history.append(action)
                    action_counts[['Left', 'Right', 'Up', 'Down'][action]] += 1

                # Treina a rede se buffer suficiente e recompensa positiva
                if len(buffers[i]) >= batch_size and reward > 0:
                    batch = random.sample(buffers[i], batch_size)
                    s, a, r, s2, d = zip(*batch)
                    s_t = torch.tensor(np.array(s)).float()
                    a_t = torch.tensor(a).long()
                    r_t = torch.tensor(r).float()
                    s2_t = torch.tensor(np.array(s2)).float()
                    d_t = torch.tensor(d).float()
                    q = agents[i](s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        q_next = targets[i](s2_t).max(1)[0]
                    q_target = r_t + gamma * q_next * (1 - d_t)
                    loss = nn.MSELoss()(q, q_target)
                    optimizers[i].zero_grad()
                    loss.backward()
                    optimizers[i].step()

                if done:
                    try:
                        active_agents.remove(i)
                    except ValueError:
                        pass

            # Atualiza métricas globais
            current_max_score = max(scores) if scores else 0
            max_tile_gen = max(get_max_tile(g) for g in grids) if grids else 0
            if current_max_score > best_score_global:
                best_score_global = current_max_score
                logging.info(f"Geração {episode + 1} - Nova Melhor Pontuação: {best_score_global}")
            if max_tile_gen > max_tile_global:
                max_tile_global = max_tile_gen

            # Desenha a tela completa
            screen.fill(COLORS['bg'])
            draw_graph(screen, scores_history, SCREEN_WIDTH - INFO_PANEL_WIDTH - NEURAL_PANEL_WIDTH - MINI_GRAPH_WIDTH - 3 * MIN_SPACING, GRAPH_HEIGHT, POS_GRAPH[0], POS_GRAPH[1])
            alive_indices = [idx for idx in range(len(scores)) if not dones[idx]]
            best_idx = max(alive_indices, key=lambda k: scores[k], default=int(np.argmax(scores)))
            max_score_gen = scores[best_idx] if scores else 0
            moves_best = moves_count[best_idx] if best_idx < len(moves_count) else 0
            ranked = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)
            top5 = ranked[:5]
            num_alive = len(active_agents)
            avg_score = np.mean(scores) if scores else 0
            draw_info_panel(screen, font, episode, num_alive, max_tile_global, max_tile_gen, best_score_global, max_score_gen, moves_best, action_counts, top5, avg_scores_history)
            agent_title = title_font.render("Melhor Agente", True, COLORS['title_text'])
            screen.blit(agent_title, (POS_AGENT[0], POS_AGENT[1] - 25))
            draw_agent_grid(screen, grids[best_idx], POS_AGENT[0], POS_AGENT[1])
            history_title = title_font.render("Histórico de Movimentos", True, COLORS['title_text'])
            screen.blit(history_title, (POS_HISTORY[0], POS_HISTORY[1] - 25))
            draw_movement_history(screen, movement_history, POS_HISTORY[0], POS_HISTORY[1])
            draw_pie_chart(screen, action_counts, POS_PIE[0] + 70, POS_PIE[1] + 70)
            last_action = movement_history[-1] if movement_history else None
            draw_neural_network(screen, POS_NEURAL[0], POS_NEURAL[1], NEURAL_PANEL_WIDTH, NEURAL_PANEL_HEIGHT, grids[best_idx], last_action, frame_count)
            draw_mini_line(screen, avg_scores_history, POS_MINI_GRAPH[0], POS_MINI_GRAPH[1], MINI_GRAPH_WIDTH, MINI_GRAPH_HEIGHT)
            prog_x = POS_INFO[0] + 10
            prog_y = POS_INFO[1] - 25
            draw_progress(screen, prog_x, prog_y, INFO_PANEL_WIDTH - 20, 10, (num_alive / max(1, INITIAL_AGENTS)))
            hovering_new = btn_newgen.collidepoint(mx, my)
            hovering_load = btn_load.collidepoint(mx, my)
            hovering_save = btn_save.collidepoint(mx, my)
            hovering_pause = btn_pause.collidepoint(mx, my)
            draw_button(screen, btn_newgen, "Nova Geração", hovering_new)
            draw_button(screen, btn_load, "Carregar Modelo", hovering_load)
            draw_button(screen, btn_save, "Salvar Modelo", hovering_save)
            draw_button(screen, btn_pause, "Pausar" if not pause else "Retomar", hovering_pause)
            pygame.draw.rect(screen, COLORS['panel'], slider_rect, border_radius=5)
            slider_handle_x = slider_rect.x + int((fps / 60.0) * slider_rect.w)
            pygame.draw.circle(screen, COLORS['button_bg'], (slider_handle_x, slider_rect.y + slider_rect.h // 2), 6)
            speed_label = small_font.render(f"Velocidade: {fps} FPS", True, COLORS['graph_text'])
            screen.blit(speed_label, (slider_rect.x, slider_rect.y - 20))

            pygame.display.flip()
            clock.tick(fps)
            await asyncio.sleep(1.0 / max(1, fps))

        # Fim da geração: salva melhor modelo e evolui população
        agent_scores = list(zip(agents[:num_agents_current], scores[:num_agents_current], moves_count[:num_agents_current]))
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        if agent_scores:
            best_agent_model = agent_scores[0][0]
            save_name = f"models/best_gen{episode + 1}.pth"
            try:
                torch.save(best_agent_model.state_dict(), save_name)
                logging.info(f"Modelo salvo: {save_name}")
            except Exception as e:
                logging.error(f"Falha ao salvar modelo: {e}")

        current_max_score = max(scores) if scores else 0
        scores_history.append(current_max_score)
        avg_score_gen = float(np.mean(scores)) if scores else 0.0
        avg_scores_history.append(avg_score_gen)

        # Evolui para próxima geração clonando e mutando o melhor
        if agent_scores:
            best_model_state = agent_scores[0][0].state_dict()
            new_agents = []
            while len(new_agents) < INITIAL_AGENTS:
                clone = DQN()
                clone.load_state_dict(best_model_state)
                for p in clone.parameters():
                    p.data += torch.randn_like(p) * 0.02
                new_agents.append(clone)
            agents = new_agents
        else:
            agents = [DQN() for _ in range(INITIAL_AGENTS)]

        # Reinicializa targets, otimizadores e buffers
        targets = [DQN() for _ in range(len(agents))]
        for i in range(len(agents)):
            targets[i].load_state_dict(agents[i].state_dict())
        optimizers = [optim.Adam(agent.parameters(), lr=0.01) for agent in agents]
        buffers = [deque(maxlen=200) for _ in range(len(agents))]

        epsilon = max(0.05, epsilon * 0.98)
        episode += 1
        logging.info(f"Geração {episode} concluída - Melhor Pontuação Global: {best_score_global}")

    logging.info("Treinamento finalizado.")

# -------------------------
# Helpers usados no loop
# -------------------------
# Escolhe uma ação com epsilon-greedy
def choose_action(model, state, epsilon=0.1):
    if random.random() < epsilon:
        return random.randint(0, 3)
    with torch.no_grad():
        st = torch.tensor(state).float().unsqueeze(0)
        out = model(st)
        return int(torch.argmax(out, dim=1).item())

# Executa um passo no jogo: move, adiciona tile se mudou, verifica fim
def step(grid, action):
    if action == 0:
        new_grid, reward = move_left(grid)
    elif action == 1:
        new_grid, reward = move_right(grid)
    elif action == 2:
        new_grid, reward = move_up(grid)
    else:
        new_grid, reward = move_down(grid)
    done = is_game_over(new_grid)
    if not grids_equal(grid, new_grid):
        add_new_tile(new_grid)
    return new_grid, reward, done

# -------------------------
# Execução
# -------------------------
# Executa o loop assíncrono dependendo da plataforma
if platform.system() == "Emscripten":
    asyncio.ensure_future(train_population())
else:
    if __name__ == "__main__":
        asyncio.run(train_population())