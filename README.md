🏭 Factory Agents – Multi-Agent Simulation

Este proyecto implementa un sistema multiagente en un escenario hipotético de fábrica, donde varios agentes colaboran para recoger y entregar cajas.

Se desarrolla en dos partes:

Simulación en Python (AgentPy) – Modela el comportamiento de los agentes y su comunicación.

Animación en Unity – Visualiza los resultados de la simulación a partir de un archivo JSON exportado.

📚 Descripción

Los agentes trabajadores (WorkerAgent) se encargan de mover las cajas desde su posición hasta la zona de entrega.

Un agente administrador (ManagerAgent) coordina las tareas usando el protocolo Contract Net (anuncio → puja → adjudicación).

El modelo (FactoryModel) controla el grid, la simulación y recopila estadísticas.

Los resultados (posiciones de agentes y cajas en cada paso) se exportan a un archivo factory_trace.json que sirve como entrada para la animación en Unity.

⚙️ Requisitos
Python

Python 3.10+

AgentPy

matplotlib

pandas

Instalar dependencias:

pip install agentpy matplotlib pandas

▶️ Ejecución en Python

Para correr la simulación:

python factory_agents.py


Salida esperada:

Número de cajas entregadas.

Número de pasos ejecutados.

Archivo factory_trace.json con el registro de la simulación.

Para correr con animación en Python:

python factory_agents.py --animate


Esto abrirá una animación en matplotlib mostrando la simulación paso a paso.

🎮 Visualización en Unity

Abrir el proyecto de Unity incluido en la carpeta /unity-visualization.

Dentro del editor, ubicar el script FactoryPlayback.cs (lee el JSON exportado).

Colocar el archivo factory_trace.json generado por Python en la carpeta Assets/StreamingAssets/.

Ejecutar la escena FactoryScene.unity.

La animación mostrará:

Agentes como esferas.

Cajas como cubos.

Zona de entrega marcada en la cuadrícula.

👉 Nota: En esta implementación, los agentes pueden pasar sobre cajas sin recogerlas para eficientar el camino. Esto hace que su movimiento se vea un poco “raro” en la animación, pero refleja la lógica simplificada del modelo.

📊 Resultados de la simulación

12 cajas entregadas.

58 pasos ejecutados.

Coordinación eficiente gracias a Contract Net.

👥 Autores

Curso TC2008B: Modelación de Sistemas Multiagentes
Equipo:
Angela Lizeth Aguirre Zúñiga	       A01286354
Bruno Fernando Zabala Peña 		A00838627
Emilio Salas Porras				A01178414
José Leobardo Navarro Márquez 	A01541324
Ricardo Bastida Rodríguez			A00839429
