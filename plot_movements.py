"""
Mapa de calor de posiciones de los agentes:
Este gráfico muestra la frecuencia con la que los agentes visitan cada celda de la cuadrícula durante la simulación. Los colores más claros indican mayor número de visitas. Se utiliza una escala logarítmica para resaltar diferencias incluso en celdas poco visitadas.
"""
import json
import matplotlib.pyplot as plt

import numpy as np
from matplotlib.colors import LogNorm

# Load the trace data
with open('factory_trace.json') as f:
    data = json.load(f)

frames = data['frames']
grid_w = data.get('gridWidth', 15)
grid_h = data.get('gridHeight', 10)

# --- 1. Agent Movements (Path Plot) ---
agent_paths = {}
for frame in frames:
    for agent in frame['agents']:
        aid = agent['id']
        if aid not in agent_paths:
            agent_paths[aid] = {'x': [], 'y': []}
        agent_paths[aid]['x'].append(agent['x'])
        agent_paths[aid]['y'].append(agent['y'])

plt.figure(figsize=(10, 6))
for aid, path in agent_paths.items():
    plt.plot(path['x'], path['y'], marker='o', label=f'Agent {aid}')
plt.title('Agent Movements')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.gca().invert_yaxis()
plt.grid(True)
plt.tight_layout()
plt.savefig('agent_movements.png')
plt.close()

# --- 2. Heatmap of Agent Positions ---
heatmap = np.zeros((grid_h, grid_w), dtype=int)
for path in agent_paths.values():
    for x, y in zip(path['x'], path['y']):
        if 0 <= x < grid_w and 0 <= y < grid_h:
            heatmap[y, x] += 1

plt.figure(figsize=(10, 6))
# Use 'viridis' colormap and logarithmic normalization for better visibility
plt.imshow(heatmap, cmap='viridis', interpolation='nearest', origin='upper', norm=LogNorm(vmin=1, vmax=heatmap.max() if heatmap.max() > 0 else 1))
plt.colorbar(label='Visits')
plt.title('Agent Position Heatmap (Log Scale)')
plt.xlabel('X')
plt.ylabel('Y')
plt.xticks(range(grid_w))
plt.yticks(range(grid_h))
plt.tight_layout()
plt.savefig('agent_heatmap.png')
plt.close()

# --- 3. Scatter Plot of Final Positions ---
plt.figure(figsize=(10, 6))
for aid, path in agent_paths.items():
    plt.scatter(path['x'][-1], path['y'][-1], label=f'Agent {aid}', s=100)
plt.title('Final Agent Positions')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.gca().invert_yaxis()
plt.grid(True)
plt.tight_layout()
plt.savefig('agent_final_positions.png')
plt.close()

# --- 4. Box Delivery Progress ---
delivered = [frame.get('delivered', 0) for frame in frames]
plt.figure(figsize=(10, 6))
plt.plot(range(len(delivered)), delivered, marker='o')
plt.title('Box Delivery Progress')
plt.xlabel('Frame')
plt.ylabel('Delivered Boxes')
plt.grid(True)
plt.tight_layout()
plt.savefig('box_delivery_progress.png')
plt.close()

print("Graphs generated: agent_movements.png, agent_heatmap.png, agent_final_positions.png, box_delivery_progress.png")
