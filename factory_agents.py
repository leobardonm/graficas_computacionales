# factory_agents.py
# Modelo de fábrica con n-agentes y m-cajitas + Q-learning con persistencia en archivo.

import agentpy as ap
import json
import argparse
from typing import Optional, Tuple, List, Dict, Set
import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
except Exception:
    plt = None
    FuncAnimation = None

# ---------- Utilidades ----------

def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def step_towards(src: Tuple[int, int], dst: Tuple[int, int]) -> Tuple[int, int]:
    """ Da un paso (1 celda) desde src hacia dst con una heurística simple. """
    x, y = src
    tx, ty = dst
    dx = 0 if x == tx else (1 if tx > x else -1)
    dy = 0 if y == ty else (1 if ty > y else -1)
    if abs(tx - x) >= abs(ty - y):
        return (x + dx, y)
    else:
        return (x, y + dy)



    # ---------- Funciones Q-table ----------

def load_qtable(filename: str) -> Dict[Tuple, float]:
    q_table = {}
    try:
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                # Buscar el cierre de la tupla (primer ')')
                close_idx = line.find(")")
                if close_idx == -1:
                    continue
                state_str = line[:close_idx+1]
                rest = line[close_idx+2:]  # saltar la coma después del paréntesis
                parts = rest.split(",")
                if len(parts) < 2:
                    continue
                action = parts[0]
                val = parts[1]
                state = eval(state_str)  # cuidado: asume que el archivo está bien formado
                q_table[(state, action)] = float(val)
    except FileNotFoundError:
        pass  # si no existe, empezamos con tabla vacía
    return q_table


def save_qtable(filename: str, q_table: Dict[Tuple, float]):
    with open(filename, "w") as f:
        for (state, action), val in q_table.items():
            f.write(f"{state},{action},{val}\n")


def average_qtables(qtables: List[Dict[Tuple, float]]) -> Dict[Tuple, float]:
    """ Promedia valores de múltiples q-tables """
    combined: Dict[Tuple, List[float]] = {}
    for qt in qtables:
        for key, val in qt.items():
            combined.setdefault(key, []).append(val)
    averaged = {k: sum(v)/len(v) for k, v in combined.items()}
    return averaged


# ---------- Agentes ----------

class ManagerAgent(ap.Agent):
    """ Gerente: anuncia cajas, recibe pujas y adjudica. """
    def setup(self):
        self.pending_announcements: List[Tuple[int, Tuple[int, int]]] = []
        self.collected_bids: Dict[int, List[Tuple[int, int]]] = {}
        self.assignments: Dict[int, int] = {}

    def step(self):
        model = self.model
        for box_id, pos in list(model.box_positions.items()):
            if box_id not in self.assignments and (box_id, pos) not in self.pending_announcements:
                self.pending_announcements.append((box_id, pos))

        if self.pending_announcements:
            box_id, pos = self.pending_announcements.pop(0)
            self.collected_bids[box_id] = []
            for w in model.workers:
                if w.available:
                    w.receive_task_announcement(box_id, pos)

        for box_id, bids in list(self.collected_bids.items()):
            if bids:
                agent_id, best_bid = min(bids, key=lambda t: t[1])
                agent = self.model.worker_by_id[agent_id]
                if agent.available and box_id not in self.assignments:
                    self.assignments[box_id] = agent_id
                    agent.accept_task(box_id, self.model.box_positions[box_id])
                del self.collected_bids[box_id]

    def receive_bid(self, box_id: int, agent_id: int, bid: int):
        if box_id in self.model.box_positions:
            self.collected_bids.setdefault(box_id, []).append((agent_id, bid))


class WorkerAgent(ap.Agent):
    """ Worker con Q-learning. """
    def setup(self):
        self.available: bool = True
        self.carrying: bool = False
        self.target_box_id: Optional[int] = None
        self.target_pos: Optional[Tuple[int, int]] = None

        # Parámetros de Q-learning
        self.q_table = {}  # asignada en FactoryModel.setup
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_decay = 0.90
        self.epsilon_min = 0.01
        self.last_state = None  
        self.last_action = None

    def get_state(self):
        grid = self.model.grid
        pos = tuple(grid.positions[self])
        carrying = int(self.carrying)
        if self.carrying:
            dest = self.model.drop_zone
        else:
            dest = self.target_pos if self.target_pos else pos
        dx = max(-1, min(1, dest[0] - pos[0]))
        dy = max(-1, min(1, dest[1] - pos[1]))
        return (pos[0], pos[1], carrying, dx, dy)

    def select_action(self, state):
        actions = ['up', 'down', 'left', 'right', 'stay']
        qvals = [self.q_table.get((state, a), 0) for a in actions]
        max_q = max(qvals)
        best_actions = [a for a, q in zip(actions, qvals) if q == max_q]
        return np.random.choice(best_actions)  # desempata aleatoriamente si hay empate


    def move(self, action):
        grid = self.model.grid
        pos = tuple(grid.positions[self])
        if action == 'up':
            proposed = (pos[0], pos[1] - 1)
        elif action == 'down':
            proposed = (pos[0], pos[1] + 1)
        elif action == 'left':
            proposed = (pos[0] - 1, pos[1])
        elif action == 'right':
            proposed = (pos[0] + 1, pos[1])
        else:
            proposed = pos
        max_x, max_y = self.model.size
        px = max(0, min(proposed[0], max_x - 1))
        py = max(0, min(proposed[1], max_y - 1))
        return (px, py)

    # --- Contract Net hooks ---
    def receive_task_announcement(self, box_id: int, box_pos: Tuple[int, int]):
        if not self.available:
            return
        my_pos = tuple(self.model.grid.positions[self])
        bid = manhattan(my_pos, box_pos)
        self.model.manager.receive_bid(box_id, self.id, bid)

    def accept_task(self, box_id: int, box_pos: Tuple[int, int]):
        if self.available:
            self.available = False
            self.target_box_id = box_id
            self.target_pos = box_pos

    # --- Movimiento y entrega con anti-colisión ---
    def step(self):
        grid = self.model.grid
        drop_zone = self.model.drop_zone
        reward = -5  # Penalización base

        if self.target_pos is None and not self.carrying:
            return

        state = self.get_state()
        action = self.select_action(state)
        proposed = self.move(action)

        # Anticolisión
        collision = False
        if proposed not in self.model.reservations:
            self.model.reservations.add(proposed)
            try:
                grid.move_to(self, proposed)
            except Exception:
                collision = True
        else:
            collision = True

        if collision:
            reward -= 50

        current = tuple(grid.positions[self])
        if (not self.carrying and self.target_box_id in self.model.box_positions and
                self.model.box_positions[self.target_box_id] == current):
            del self.model.box_positions[self.target_box_id]
            self.carrying = True

        if self.carrying and current == drop_zone:
            self.carrying = False
            self.available = True
            self.target_box_id = None
            self.target_pos = None
            self.model.delivered += 1
            reward += 100

        # Q-learning update
        next_state = self.get_state()
        if self.last_state is not None and self.last_action is not None:
            prev_q = self.q_table.get((self.last_state, self.last_action), 0)
            next_qs = [self.q_table.get((next_state, a), 0) for a in ['up','down','left','right','stay']]
            max_next_q = max(next_qs)
            new_q = prev_q + self.alpha * (reward + self.gamma * max_next_q - prev_q)
            self.q_table[(self.last_state, self.last_action)] = new_q
        self.last_state = state
        self.last_action = action

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# ---------- Modelo ----------

class FactoryModel(ap.Model):
    def setup(self):
        p = self.p
        self.size = (p['grid_width'], p['grid_height'])
        self.drop_zone: Tuple[int, int] = tuple(p.get('drop_zone', (0, 0)))
        self.delivered: int = 0
        self.trace = []  # Inicializar el atributo trace

        self.grid = ap.Grid(self, self.size, track_empty=True)
        self.manager = ManagerAgent(self)
        self.manager.setup()

        # Workers
        self.workers = ap.AgentList(self, p['n_agents'], WorkerAgent)
        self.worker_by_id: Dict[int, WorkerAgent] = {}

        free_cells = [(x, y) for x in range(self.size[0]) for y in range(self.size[1])
                      if (x, y) != self.drop_zone]
        self.random.shuffle(free_cells)

        start_pos_agents = free_cells[:p['n_agents']]
        self.grid.add_agents(self.workers, start_pos_agents)

        # Cargar Q-table global
        self.global_qtable = load_qtable("qtable.txt")

        for w in self.workers:
            w.setup()
            w.q_table = self.global_qtable.copy()  # copia individual
            self.worker_by_id[w.id] = w

        # Cajas
        remaining = [cell for cell in free_cells[p['n_agents']:] if cell != self.drop_zone]
        self.random.shuffle(remaining)
        self.box_positions: Dict[int, Tuple[int, int]] = {}
        for i in range(p['m_boxes']):
            self.box_positions[i] = remaining[i]

    def _record_frame(self):
        agents_payload = []
        for w in self.workers:
            x, y = map(int, self.grid.positions[w])
            agents_payload.append({
                "id": int(w.id),
                "x": x, "y": y,
                "carrying": bool(w.carrying),
                "available": bool(w.available)
            })

        boxes_payload = [{"id": int(bid), "x": int(pos[0]), "y": int(pos[1])}
                         for bid, pos in self.box_positions.items()]

        self.trace.append({
            "agents": agents_payload,
            "boxes": boxes_payload,
            "delivered": int(self.delivered)
        })

    def step(self):
        self.reservations: Set[Tuple[int, int]] = set()
        self.manager.step()
        self.workers.step()
        self.record('delivered', self.delivered)
        self._record_frame()

        if not self.box_positions and all(not w.carrying for w in self.workers):
            self.stop()

    def end(self):
        # Promediar Q-tables y guardar
        all_qtables = [w.q_table for w in self.workers]
        averaged = average_qtables(all_qtables)
        save_qtable("qtable.txt", averaged)

        out = {
            "gridWidth": int(self.size[0]),
            "gridHeight": int(self.size[1]),
            "dropZone": {"x": int(self.drop_zone[0]), "y": int(self.drop_zone[1])},
            "frames": self.trace
        }
        with open("factory_trace.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--animate", action="store_true", help="Mostrar animación en Python (matplotlib)")
    args = parser.parse_args()

    params = {
        'grid_width': 15,
        'grid_height': 10,
        'n_agents': 5,
        'm_boxes': 12,
        'drop_zone': (0, 0),
        'seed': 42
    }

    model = FactoryModel(params)
    results = model.run(steps=3000, display=False)

    delivered_series = results.variables['FactoryModel']['delivered']
    last_delivered = delivered_series.iloc[-1] if len(delivered_series) else 0
    steps = len(model.trace) - 1

    print("=== RESULTADOS ===")
    print(f"Cajas entregadas: {last_delivered}")
    print(f"Pasos ejecutados: {steps}")

    if args.animate:
        if plt is None or FuncAnimation is None:
            print("matplotlib no disponible. Instala con: pip install matplotlib")
            return
        animate_python(model)

def animate_python(model: FactoryModel):
    grid_w, grid_h = model.size
    frames = model.trace

    fig, ax = plt.subplots()
    ax.set_xlim(-0.5, grid_w - 0.5)
    ax.set_ylim(-0.5, grid_h - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.2)
    ax.set_title("Fábrica (animación ligera)")

    dz = model.drop_zone
    dz_scatter = ax.scatter([dz[0]], [dz[1]], s=200, marker='s')
    agents_scatter = ax.scatter([], [], s=120, marker='o')
    boxes_scatter = ax.scatter([], [], s=120, marker='s')

    def to_offsets(pairs):
        if not pairs:
            return np.empty((0, 2))
        return np.array(pairs, dtype=float)

    def update(i):
        fr = frames[i]
        ax.set_xlabel(f"Paso {i} | Entregadas: {fr['delivered']}")
        ax_agents = [(a["x"], a["y"]) for a in fr["agents"]]
        agents_scatter.set_offsets(to_offsets(ax_agents))
        bx = [(b["x"], b["y"]) for b in fr["boxes"]]
        boxes_scatter.set_offsets(to_offsets(bx))
        return agents_scatter, boxes_scatter, dz_scatter

    anim = FuncAnimation(fig, update, frames=len(frames), interval=250, blit=False, repeat=False)
    plt.show()


if __name__ == "__main__":
    main()
