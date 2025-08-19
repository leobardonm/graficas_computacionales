# factory_agents.py
# Modelo de fábrica con n-agentes y m-cajitas.
# Comunicación: Contract Net (anuncio -> puja -> adjudicación).
# Anticolisión: reserva de celdas por paso.
# Exporta un "replay" del modelo a JSON (factory_trace.json) para Unity.
# Opción de animación rápida en Python con matplotlib (--animate).

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

# ---------- Agentes ----------

class ManagerAgent(ap.Agent):
    """ Gerente: anuncia cajas, recibe pujas y adjudica. """
    def setup(self):
        self.pending_announcements: List[Tuple[int, Tuple[int, int]]] = []  # [(box_id, pos)]
        self.collected_bids: Dict[int, List[Tuple[int, int]]] = {}          # box_id -> [(agent_id, bid)]
        self.assignments: Dict[int, int] = {}                               # box_id -> agent_id

    def step(self):
        model = self.model
        # Anunciar nuevas cajas sin asignación
        for box_id, pos in list(model.box_positions.items()):
            if box_id not in self.assignments and (box_id, pos) not in self.pending_announcements:
                self.pending_announcements.append((box_id, pos))

        # Anunciar UNA caja por paso para evitar congestión
        if self.pending_announcements:
            box_id, pos = self.pending_announcements.pop(0)
            self.collected_bids[box_id] = []
            for w in model.workers:
                if w.available:
                    w.receive_task_announcement(box_id, pos)

        # Evaluar pujas y adjudicar
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
    """ Worker: se mueve, recoge su caja asignada y la entrega en drop_zone. """
    def setup(self):
        self.available: bool = True
        self.carrying: bool = False
        self.target_box_id: Optional[int] = None
        self.target_pos: Optional[Tuple[int, int]] = None

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

        if self.carrying:
            dest = drop_zone
        else:
            if self.target_pos is None:
                return  # sin tarea asignada
            dest = self.target_pos

        current = tuple(grid.positions[self])

        # Si ya estamos en destino
        if current == dest:
            if self.carrying:
                # Entregar
                self.carrying = False
                self.available = True
                self.target_box_id = None
                self.target_pos = None
                self.model.delivered += 1
            else:
                # Recoger caja si existe aún
                if (self.target_box_id in self.model.box_positions and
                        self.model.box_positions[self.target_box_id] == current):
                    del self.model.box_positions[self.target_box_id]
                    self.carrying = True
            return

        # Proponer siguiente celda
        proposed = step_towards(current, dest)

        # Limitar a la grilla
        max_x, max_y = self.model.size
        px = max(0, min(proposed[0], max_x - 1))
        py = max(0, min(proposed[1], max_y - 1))
        proposed = (px, py)

        # Anticolisión: reservar celda si está libre
        if proposed not in self.model.reservations:
            self.model.reservations.add(proposed)
            try:
                grid.move_to(self, proposed)
            except Exception:
                pass
        else:
            pass

# ---------- Modelo ----------

class FactoryModel(ap.Model):
    """
    Parámetros (dict p):
        grid_width, grid_height
        n_agents
        m_boxes
        drop_zone: (x, y)
        seed
    """

    def setup(self):
        p = self.p
        self.size = (p['grid_width'], p['grid_height'])
        self.drop_zone: Tuple[int, int] = tuple(p.get('drop_zone', (0, 0)))
        self.delivered: int = 0

        # Grilla
        self.grid = ap.Grid(self, self.size, track_empty=True)

        # Manager (no está en grilla)
        self.manager = ManagerAgent(self)
        self.manager.setup()

        # Workers
        self.workers = ap.AgentList(self, p['n_agents'], WorkerAgent)
        self.worker_by_id: Dict[int, WorkerAgent] = {}

        # Colocación inicial aleatoria (evita drop_zone)
        free_cells = [(x, y) for x in range(self.size[0]) for y in range(self.size[1])
                      if (x, y) != self.drop_zone]
        self.random.shuffle(free_cells)

        start_pos_agents = free_cells[:p['n_agents']]
        self.grid.add_agents(self.workers, start_pos_agents)

        for w in self.workers:
            w.setup()
            self.worker_by_id[w.id] = w

        # Cajas como posiciones simples (id -> (x, y))
        remaining = [cell for cell in free_cells[p['n_agents']:] if cell != self.drop_zone]
        self.random.shuffle(remaining)
        self.box_positions: Dict[int, Tuple[int, int]] = {}
        for i in range(p['m_boxes']):
            self.box_positions[i] = remaining[i]

        # Métricas & replay
        self.record('delivered', self.delivered)
        self.trace: List[Dict] = []  # aquí guardaremos cada "frame" para Unity

        # Frame inicial
        self._record_frame()

    def _record_frame(self):
        """ Guarda un snapshot del estado en self.trace """
        # Posiciones de agentes
        agents_payload = []
        for w in self.workers:
            x, y = map(int, self.grid.positions[w])
            agents_payload.append({
                "id": int(w.id),
                "x": x, "y": y,
                "carrying": bool(w.carrying),
                "available": bool(w.available)
            })

        # Posiciones de cajas
        boxes_payload = [{"id": int(bid), "x": int(pos[0]), "y": int(pos[1])}
                         for bid, pos in self.box_positions.items()]

        self.trace.append({
            "agents": agents_payload,
            "boxes": boxes_payload,
            "delivered": int(self.delivered)
        })

    def step(self):
        # Limpiar reservas por paso
        self.reservations: Set[Tuple[int, int]] = set()

        # Manager anuncia/ adjudica
        self.manager.step()

        # Workers actúan
        self.workers.step()

        # Registrar entregas
        self.record('delivered', self.delivered)

        # Grabar frame para replay
        self._record_frame()

        # Paro si ya no hay cajas y nadie está cargando
        if not self.box_positions and all(not w.carrying for w in self.workers):
            self.stop()

    def end(self):
        # Exportar replay a JSON
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
        'n_agents': 5,      # n-agentes
        'm_boxes': 12,      # m-cajitas
        'drop_zone': (0, 0),
        'seed': 42
    }

    model = FactoryModel(params)
    results = model.run(steps=300, display=False)

    delivered_series = results.variables['FactoryModel']['delivered']
    last_delivered = delivered_series.iloc[-1] if len(delivered_series) else 0

    # Si el reporter steps no está disponible, dedúcelo del replay:
    steps = len(model.trace) - 1  # frames-1 (porque incluimos frame inicial)
    print("=== RESULTADOS ===")
    print(f"Cajas entregadas: {last_delivered}")
    print(f"Pasos ejecutados: {steps}")

    if args.animate:
        if plt is None or FuncAnimation is None:
            print("matplotlib no disponible. Instala con: pip install matplotlib")
            return
        animate_python(model)

def animate_python(model: FactoryModel):
    """ Animación simple en 2D para depurar (sin recrear artistas por frame). """
    grid_w, grid_h = model.size
    frames = model.trace

    fig, ax = plt.subplots()
    ax.set_xlim(-0.5, grid_w - 0.5)
    ax.set_ylim(-0.5, grid_h - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.2)
    ax.set_title("Fábrica (animación ligera)")

    # Drop zone (un solo artista)
    dz = model.drop_zone
    dz_scatter = ax.scatter([dz[0]], [dz[1]], s=200, marker='s')

    # Artistas reutilizables para agentes y cajas
    agents_scatter = ax.scatter([], [], s=120, marker='o')
    boxes_scatter = ax.scatter([], [], s=120, marker='s')

    def to_offsets(pairs):
        """ pairs: list[tuple(x,y)] -> Nx2 array for set_offsets """
        if not pairs:
            return np.empty((0, 2))
        return np.array(pairs, dtype=float)

    def update(i):
        fr = frames[i]
        ax.set_xlabel(f"Paso {i} | Entregadas: {fr['delivered']}")

        # Agents
        ax_agents = [(a["x"], a["y"]) for a in fr["agents"]]
        agents_scatter.set_offsets(to_offsets(ax_agents))

        # Boxes
        bx = [(b["x"], b["y"]) for b in fr["boxes"]]
        boxes_scatter.set_offsets(to_offsets(bx))

        # No devolvemos una lista obligatoriamente (blit=False)
        return agents_scatter, boxes_scatter, dz_scatter

    anim = FuncAnimation(fig, update, frames=len(frames), interval=250, blit=False, repeat=False)
    plt.show()

if __name__ == "__main__":
    main()
