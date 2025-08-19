import matplotlib.pyplot as plt

# Model design
import agentpy as ap
import networkx as nx

# Visualization
import seaborn as sns

class ButtonModel(ap.Model):

    def setup(self):

        # Create a graph with n agents
        self.buttons = ap.Network(self)
        self.agents = ap.AgentList(self, self.p.n)
        self.buttons.add_agents(self.agents)
        self.agents.node = self.buttons.nodes
        self.threads = 0

    def update(self):

        # Record size of the biggest cluster
        clusters = nx.connected_components(self.buttons.graph)
        max_cluster_size = max([len(g) for g in clusters]) / self.p.n
        self.record('max_cluster_size', max_cluster_size)

        # Record threads to button ratio
        self.record('threads_to_button', self.threads / self.p.n)

    def step(self):

        # Create random edges based on parameters
        for _ in range(int(self.p.n * self.p.speed)):
            self.buttons.graph.add_edge(*self.agents.random(2).node)
            self.threads += 1


# Define parameter ranges
parameter_ranges = {
    'steps': 100,  # Number of simulation steps
    'speed': 1,  # Speed of connections per step
    'n': ap.Values(100, 1000, 10000)  # Number of agents
}

# Create sample for different values of n
sample = ap.Sample(parameter_ranges)

# Keep dynamic variables
exp = ap.Experiment(ButtonModel, sample, iterations=25, record=True)

# Perform 75 separate simulations (3 parameter combinations * 25 repetitions)
results = exp.run()

# Plot averaged time-series for discrete parameter samples
sns.set_theme()
sns.lineplot(
    data=results.arrange_variables(),
    x='threads_to_button',
    y='max_cluster_size',
    hue='n'
);


plt.show(block=True) # Show the plot