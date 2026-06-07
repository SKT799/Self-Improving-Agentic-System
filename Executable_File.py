import pygame
import numpy as np
import random
import sys
import csv
import time

"""
Controls:
  - Click buttons in right panel: Start/Train & Run, Reset, Export CSV
  - Keyboard shortcuts: S (start/stop), R (reset), E (export CSV)
  - Click "Scenario" yellow box to cycle: Warehouse Picking / Sortation Hub / Last-Mile Dock
  - Adjust sliders: Episodes, Epsilon, Pause Prob, Humans, FPS
"""

# ------------------------ INIT ------------------------
pygame.init()

# Grid / world params
GRID_SIZE = 25
CELL_SIZE = int(24 * 1.3)
WIDTH, HEIGHT = GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE

# Theme / colors
WHITE = (255, 255, 255)
GRAY = (30, 30, 30)
BLUE = (50, 100, 220)
ORANGE = (255, 165, 0)
PURPLE = (160, 32, 240)
BLACK = (0, 0, 0)
GREEN = (0, 180, 0)
RED = (200, 50, 50)
CYAN = (0, 200, 200)

# UI theme
FLIPKART_BLUE = (40, 116, 240)
FLIPKART_YELLOW = (255, 204, 0)
PANEL_W = 320
UI_WIDTH = WIDTH + PANEL_W
screen = pygame.display.set_mode((UI_WIDTH, HEIGHT))
pygame.display.set_caption("Flipkart RL Routing — Multi-Agent (Q-Learning + Dynamic Policy)")

# Rewards / markers
OBSTACLE = -1
GOAL = 100

# Actions
actions = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1)
}

# Q-learning hyperparams (default; change via UI)
alpha, gamma, epsilon = 0.3, 0.9, 0.5
episodes = 5000
percentage_of_pause = 0.2
number_of_humans = 60

# Goals and colors
goal_positions = [(19, 7), (19, 10), (10, 13)]
goal_colors = [BLUE, ORANGE, PURPLE]

# ------------------------ WORLD BUILDING ------------------------

def build_grid(scenario: str):
    """Return a fresh grid ndarray with obstacles according to scenario."""
    g = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    if scenario == "Warehouse Picking":
        g[5, 7:18] = OBSTACLE
        g[8, 18:25] = OBSTACLE
        g[15, 6:14] = OBSTACLE
        g[15, 20:25] = OBSTACLE
        g[10, 0:11] = OBSTACLE
        g[8:12, 16] = OBSTACLE
        g[20, 6:16] = OBSTACLE
    elif scenario == "Sortation Hub":
        # Denser parallel lanes with cross-cuts
        for r in [4, 6, 8, 10, 12, 14, 16, 18]:
            g[r, 2:23] = OBSTACLE
        # Cross gaps
        for c in [5, 10, 15, 20]:
            g[4:19:2, c] = 0
    elif scenario == "Last-Mile Dock":
        # More open, a few big blocks
        g[6:9, 6:19] = OBSTACLE
        g[14:17, 3:10] = OBSTACLE
        g[14:17, 15:22] = OBSTACLE
    else:
        pass
    # Place goals
    for goal in goal_positions:
        g[goal] = GOAL
    return g

# ------------------------ TRAINING ------------------------

def init_Q():
    return {(i, j): {a: 0.0 for a in actions} for i in range(GRID_SIZE) for j in range(GRID_SIZE)}

def take_action(grid, pos, action, goal, other_goals):
    x, y = pos
    dx, dy = actions[action]
    nx, ny = x + dx, y + dy

    if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE):
        return pos, -10
    if grid[nx, ny] == OBSTACLE:
        return pos, -100
    if (nx, ny) in other_goals:
        return pos, -50
    if (nx, ny) == goal:
        return (nx, ny), 100
    return (nx, ny), -1

# Progress bar during training
def draw_progress_bar(surface, episode, total_episodes, agent_name, width=400, height=30):
    progress = episode / max(1, total_episodes)
    bar_width = int(width * progress)
    x = (WIDTH - width) // 2
    y = (HEIGHT - height) // 2

    pygame.draw.rect(surface, (235, 235, 235), (x, y, width, height))
    pygame.draw.rect(surface, GREEN, (x, y, bar_width, height))

    font = pygame.font.SysFont(None, 28)
    text = font.render(f'Training {agent_name} - {int(progress * 100)}%', True, (0, 0, 0))
    text_rect = text.get_rect(center=(WIDTH // 2, y - 25))
    surface.blit(text, text_rect)

    pygame.display.flip()

def train_agent(grid, start_pos, goal, other_goals, agent_name):
    Q = init_Q()
    for episode_idx in range(episodes):
        pos = start_pos
        while pos != goal:
            if random.random() < epsilon:
                action = random.choice(list(actions))
            else:
                action = max(Q[pos], key=Q[pos].get)
            next_pos, reward = take_action(grid, pos, action, goal, other_goals)
            Q[pos][action] += alpha * (reward + gamma * max(Q[next_pos].values()) - Q[pos][action])
            pos = next_pos
        if episode_idx % max(1, episodes // 50) == 0:
            # draw progress in left panel only
            left = pygame.Surface((WIDTH, HEIGHT))
            left.fill(WHITE)
            draw_progress_bar(left, episode_idx, episodes, agent_name)
            screen.blit(left, (0, 0))
            draw_panel()
            pygame.display.flip()
    return Q

# ------------------------ MOVING OBSTACLES ------------------------

def move_human_randomly(grid, pos):
    dirs = list(actions.values())
    random.shuffle(dirs)
    for dx, dy in dirs:
        nx, ny = pos[0] + dx, pos[1] + dy
        if (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and (nx, ny) not in goal_positions and grid[nx, ny] != OBSTACLE):
            return (nx, ny)
    return pos

# ------------------------ RENDERING ------------------------

def draw_grid(grid):
    # Background
    pygame.draw.rect(screen, BLACK, (0, 0, WIDTH, HEIGHT))
    # Cells
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            rect = pygame.Rect(j*CELL_SIZE, i*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, GRAY, rect, 1)
            if grid[i, j] == OBSTACLE:
                pygame.draw.rect(screen, (80, 80, 80), rect)
    # Goals
    for goal, color in zip(goal_positions, goal_colors):
        rect = pygame.Rect(goal[1]*CELL_SIZE, goal[0]*CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, color, rect)

# Minimalist entity rendering (no external images)
HOME_COLOR = (240, 240, 240)
ROBOT_COLORS = [BLUE, ORANGE, PURPLE]
HUMAN_COLOR = (120, 200, 120)

def draw_home(pos):
    x, y = pos[1]*CELL_SIZE, pos[0]*CELL_SIZE
    r = pygame.Rect(x+4, y+4, CELL_SIZE-8, CELL_SIZE-8)
    pygame.draw.rect(screen, HOME_COLOR, r, border_radius=6)
    pygame.draw.rect(screen, (100, 100, 100), r, 2, border_radius=6)

def draw_robot(idx, pos):
    x, y = pos[1]*CELL_SIZE, pos[0]*CELL_SIZE
    r = pygame.Rect(x+6, y+6, CELL_SIZE-12, CELL_SIZE-12)
    pygame.draw.rect(screen, ROBOT_COLORS[idx%3], r, border_radius=8)
    pygame.draw.rect(screen, (20,20,20), r, 2, border_radius=8)


def draw_human(pos):
    cx, cy = pos[1]*CELL_SIZE + CELL_SIZE//2, pos[0]*CELL_SIZE + CELL_SIZE//2
    pygame.draw.circle(screen, HUMAN_COLOR, (cx, cy), CELL_SIZE//3)
    pygame.draw.circle(screen, (40, 100, 40), (cx, cy), CELL_SIZE//3, 2)


def draw_line_path(surface, path, color, width=4):
    if len(path) < 2:
        return
    points = [(pos[1]*CELL_SIZE + CELL_SIZE//2, pos[0]*CELL_SIZE + CELL_SIZE//2) for pos in path]
    pygame.draw.lines(surface, color, False, points, width)

# ------------------------ COLLISION ------------------------

def detect_collision(agent_positions, human_positions, idx):
    agent_pos = agent_positions[idx]
    others = agent_positions[:idx] + agent_positions[idx+1:]
    if agent_pos in human_positions:
        return True
    if agent_pos in others:
        return True
    return False

# ------------------------ UI LAYER ------------------------

class ControlState:
    def __init__(self):
        self.episodes = episodes
        self.epsilon = epsilon
        self.pause_prob = percentage_of_pause
        self.humans = number_of_humans
        self.fps = 6
        self.seed = 42
        self.scenario = "Warehouse Picking"
        self.running = False
        self.training_done = False
    def apply_to_globals(self):
        global episodes, epsilon, percentage_of_pause, number_of_humans
        episodes = int(self.episodes)
        epsilon = float(self.epsilon)
        percentage_of_pause = float(self.pause_prob)
        number_of_humans = int(self.humans)
        random.seed(self.seed)
        np.random.seed(self.seed)

CTRL = ControlState()

class Button:
    def __init__(self, rect, label):
        self.rect = pygame.Rect(rect)
        self.label = label
    def draw(self, surf):
        pygame.draw.rect(surf, FLIPKART_BLUE, self.rect, border_radius=10)
        font = pygame.font.SysFont(None, 24)
        txt = font.render(self.label, True, (255, 255, 255))
        surf.blit(txt, txt.get_rect(center=self.rect.center))
    def hit(self, pos):
        return self.rect.collidepoint(pos)

class Slider:
    def __init__(self, x, y, w, label, vmin, vmax, value, step=1):
        self.rect = pygame.Rect(x, y, w, 24)
        self.label = label
        self.vmin, self.vmax = vmin, vmax
        self.value = value
        self.step = step
        self.drag = False
    def draw(self, surf):
        font = pygame.font.SysFont(None, 22)
        # Track
        track = pygame.Rect(self.rect.x, self.rect.y + 12, self.rect.w, 4)
        pygame.draw.rect(surf, (190, 190, 190), track)
        # Knob
        t = (self.value - self.vmin) / (self.vmax - self.vmin)
        kx = int(track.x + t * track.w)
        pygame.draw.circle(surf, FLIPKART_BLUE, (kx, track.y + 2), 8)
        # Text
        txt = font.render(f"{self.label}: {self.value}", True, (30, 30, 30))
        surf.blit(txt, (self.rect.x, self.rect.y - 18))
    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self.drag = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.drag = False
        elif event.type == pygame.MOUSEMOTION and self.drag:
            t = (event.pos[0] - self.rect.x) / self.rect.w
            t = min(max(t, 0), 1)
            raw = self.vmin + t * (self.vmax - self.vmin)
            if isinstance(self.step, float):
                # round to nearest step for float sliders
                steps = round((raw - self.vmin)/self.step)
                self.value = round(self.vmin + steps*self.step, 2)
            else:
                self.value = int(round(raw / self.step) * self.step)

SCENARIOS = {
    "Warehouse Picking": dict(humans=60),
    "Sortation Hub": dict(humans=90),
    "Last-Mile Dock": dict(humans=40),
}

btn_start = Button((WIDTH + 20, 20, PANEL_W - 40, 40), "Start / Train & Run (S)")
btn_reset = Button((WIDTH + 20, 70, PANEL_W - 40, 40), "Reset (R)")
btn_export = Button((WIDTH + 20, 120, PANEL_W - 40, 40), "Export CSV (E)")

s_eps = Slider(WIDTH + 20, 190, PANEL_W - 40, "Episodes", 100, 20000, CTRL.episodes, step=100)
s_eps_greedy = Slider(WIDTH + 20, 240, PANEL_W - 40, "Epsilon", 0.0, 1.0, CTRL.epsilon, step=0.05)
s_pause = Slider(WIDTH + 20, 290, PANEL_W - 40, "Pause Prob", 0.0, 0.9, CTRL.pause_prob, step=0.05)
s_humans = Slider(WIDTH + 20, 340, PANEL_W - 40, "Humans", 0, 120, CTRL.humans, step=5)
s_fps = Slider(WIDTH + 20, 390, PANEL_W - 40, "FPS", 1, 30, CTRL.fps, step=1)

scenario_box = pygame.Rect(WIDTH + 20, 450, PANEL_W - 40, 34)

class Metrics:
    def __init__(self):
        self.reset()
    def reset(self):
        self.steps = 0
        self.pauses = 0
        self.collisions_avoided = 0
        self.path_len = [0, 0, 0]
        self.done = False
        self.tick = 0
    def draw(self, surf):
        font = pygame.font.SysFont(None, 22)
        y = 500
        lines = [
            f"Scenario: {CTRL.scenario}",
            f"Steps: {self.steps}",
            f"Pauses: {self.pauses}",
            f"Collisions avoided: {self.collisions_avoided}",
            f"Path len A1/A2/A3: {self.path_len[0]}/{self.path_len[1]}/{self.path_len[2]}",
            f"ETA ticks: {'—' if self.done else self.tick}",
        ]
        for line in lines:
            txt = font.render(line, True, (30, 30, 30))
            surf.blit(txt, (WIDTH + 20, y))
            y += 24

MET = Metrics()

def draw_panel():
    pygame.draw.rect(screen, (245, 246, 248), (WIDTH, 0, PANEL_W, HEIGHT))
    title_font = pygame.font.SysFont(None, 26)
    t = title_font.render("Flipkart Control Panel", True, (15, 15, 15))
    screen.blit(t, (WIDTH + 20, 12))

    btn_start.draw(screen)
    btn_reset.draw(screen)
    btn_export.draw(screen)

    for s in (s_eps, s_eps_greedy, s_pause, s_humans, s_fps):
        s.draw(screen)

    pygame.draw.rect(screen, FLIPKART_YELLOW, scenario_box, border_radius=8)
    sfont = pygame.font.SysFont(None, 22)
    sc = sfont.render(f"Scenario: {CTRL.scenario} (click)", True, (0, 0, 0))
    screen.blit(sc, sc.get_rect(center=scenario_box.center))

    MET.draw(screen)


def handle_panel_event(event):
    for s in (s_eps, s_eps_greedy, s_pause, s_humans, s_fps):
        s.handle(event)
    if event.type == pygame.MOUSEBUTTONDOWN:
        if btn_start.hit(event.pos):
            start_or_toggle()
        elif btn_reset.hit(event.pos):
            hard_reset()
        elif btn_export.hit(event.pos):
            export_csv()
        elif scenario_box.collidepoint(event.pos):
            cycle_scenario()


def sync_from_widgets():
    CTRL.episodes = int(s_eps.value)
    CTRL.epsilon = float(s_eps_greedy.value)
    CTRL.pause_prob = float(s_pause.value)
    CTRL.humans = int(s_humans.value)
    CTRL.fps = int(s_fps.value)


def cycle_scenario():
    keys = list(SCENARIOS.keys())
    i = keys.index(CTRL.scenario)
    CTRL.scenario = keys[(i + 1) % len(keys)]
    preset = SCENARIOS[CTRL.scenario]
    s_humans.value = preset["humans"]


def start_or_toggle():
    sync_from_widgets()
    CTRL.apply_to_globals()
    CTRL.running = not CTRL.running


def hard_reset():
    CTRL.running = False
    CTRL.training_done = False
    MET.reset()


def export_csv():
    path = f"kpi_export_{int(time.time())}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario","episodes","epsilon","pause_prob","humans","fps","steps","pauses","collisions_avoided","pathA1","pathA2","pathA3","ticks"])
        w.writerow([CTRL.scenario, CTRL.episodes, CTRL.epsilon, CTRL.pause_prob, CTRL.humans, CTRL.fps, MET.steps, MET.pauses, MET.collisions_avoided, *MET.path_len, MET.tick])
    print(f"Exported KPIs -> {path}")

# ------------------------ INIT HELPERS ------------------------

def init_humans_with_UI(grid):
    global number_of_humans
    number_of_humans = CTRL.humans
    human_positions = []
    tries = 0
    while len(human_positions) < number_of_humans and tries < 10000:
        pos = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
        if grid[pos] == 0 and pos not in goal_positions and pos not in [(0, 0), (0, GRID_SIZE - 1), (GRID_SIZE - 1, GRID_SIZE - 1)] and pos not in human_positions:
            human_positions.append(pos)
        tries += 1
    return human_positions

# ------------------------ MAIN LOOP ------------------------

def main():
    clock = pygame.time.Clock()

    trained = False
    grid = build_grid(CTRL.scenario)

    # Start/home positions for 3 agents
    start_positions = [(0, 0), (0, GRID_SIZE-1), (GRID_SIZE-1, GRID_SIZE-1)]
    agent_positions = list(start_positions)
    paths = [[pos] for pos in agent_positions]

    # Pre-define policies (filled after training)
    policy1 = policy2 = policy3 = None

    # Humans
    human_positions = init_humans_with_UI(grid)

    while True:
        clock.tick(CTRL.fps if CTRL.running else 30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            handle_panel_event(event)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s: start_or_toggle()
                elif event.key == pygame.K_r: hard_reset()
                elif event.key == pygame.K_e: export_csv()

        if not CTRL.running:
            # Idle preview: draw grid + homes + panel
            grid = build_grid(CTRL.scenario)
            draw_grid(grid)
            for start in start_positions:
                draw_home(start)
            draw_panel()
            pygame.display.flip()
            continue

        # First tick after start or after hard reset => (re)train
        if not trained or not CTRL.training_done:
            sync_from_widgets()
            CTRL.apply_to_globals()
            MET.reset()

            # Rebuild grid per scenario
            grid = build_grid(CTRL.scenario)

            # Train agents sequentially
            Q1 = train_agent(grid, start_positions[0], goal_positions[0], goal_positions[1:], "Agent 1")
            Q2 = train_agent(grid, start_positions[1], goal_positions[1], [goal_positions[0], goal_positions[2]], "Agent 2")
            Q3 = train_agent(grid, start_positions[2], goal_positions[2], goal_positions[:2], "Agent 3")

            policy1 = {s: max(Q1[s], key=Q1[s].get) for s in Q1}
            policy2 = {s: max(Q2[s], key=Q2[s].get) for s in Q2}
            policy3 = {s: max(Q3[s], key=Q3[s].get) for s in Q3}

            # Reset agents and humans
            agent_positions = list(start_positions)
            paths = [[pos] for pos in agent_positions]
            human_positions = init_humans_with_UI(grid)

            trained = True
            CTRL.training_done = True

        # ---------------- Simulation tick ----------------
        MET.tick += 1

        # Draw base grid and homes
        draw_grid(grid)
        for start in start_positions:
            draw_home(start)

        # Move humans and draw
        human_positions = [move_human_randomly(grid, pos) for pos in human_positions]
        for hp in human_positions:
            draw_human(hp)

        # A surface to draw paths with alpha
        path_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

        # Previous positions for collision handling
        previous_positions = agent_positions.copy()

        policies = [policy1, policy2, policy3]

        # Greedy policy step for each agent
        for i in range(len(agent_positions)):
            if agent_positions[i] != goal_positions[i]:
                action = policies[i][agent_positions[i]]
                other_goals = [g for idx, g in enumerate(goal_positions) if idx != i]
                next_pos, _ = take_action(grid, agent_positions[i], action, goal_positions[i], other_goals)
                agent_positions[i] = next_pos

        # Collision-aware dynamic policy (pause or safe detour)
        for i in range(len(agent_positions)):
            if detect_collision(agent_positions, human_positions, i):
                if random.random() < percentage_of_pause:
                    agent_positions[i] = previous_positions[i]
                    MET.pauses += 1
                else:
                    found_safe = False
                    for dx, dy in actions.values():
                        nx, ny = previous_positions[i][0] + dx, previous_positions[i][1] + dy
                        if (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and
                            grid[nx, ny] != OBSTACLE and (nx, ny) not in human_positions and (nx, ny) not in agent_positions):
                            agent_positions[i] = (nx, ny)
                            paths[i].append(agent_positions[i])
                            MET.steps += 1
                            MET.path_len[i] += 1
                            MET.collisions_avoided += 1
                            found_safe = True
                            break
                    if not found_safe:
                        agent_positions[i] = previous_positions[i]
                        paths[i].append(agent_positions[i])
                        MET.steps += 1
                        MET.path_len[i] += 1
                        MET.pauses += 1
            else:
                paths[i].append(agent_positions[i])
                MET.steps += 1
                MET.path_len[i] += 1

        # Draw paths and robots
        path_colors = [(50, 100, 220, 180), (255, 165, 0, 180), (160, 32, 240, 180)]
        for i in range(len(agent_positions)):
            draw_line_path(path_surface, paths[i], path_colors[i])
            draw_robot(i, agent_positions[i])
        screen.blit(path_surface, (0, 0))

        # Done?
        if all(agent_positions[i] == goal_positions[i] for i in range(len(agent_positions))):
            MET.done = True
            CTRL.running = False

        # Draw UI panel
        draw_panel()
        pygame.display.flip()

if __name__ == "__main__":
    main()
