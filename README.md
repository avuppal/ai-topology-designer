# AI Topology Designer

## Overview

This project is a Python tool for generating, evaluating, and visualizing network topologies optimized for AI infrastructure, such as GPU clusters for distributed training. It supports common topologies like ring, torus, hypercube, and fat-tree, with metrics for performance analysis. Ideal for simulating designs in enterprise AI environments where topology impacts latency, bandwidth, and scalability.

Key features:
- Generate topologies programmatically.
- Evaluate metrics like diameter, bandwidth, and average path length.
- Simulate basic traffic patterns.
- Visualize graphs for intuitive understanding.

## Tech Stack
- Python 3.x
- NetworkX for graph operations
- Matplotlib for visualization
- Argparse for CLI

## Setup Instructions

1. **Prerequisites**:
   ```
   pip install networkx matplotlib
   ```

2. **Clone the Repo**:
   ```
   git clone https://github.com/avuppal/ai-topology-designer.git
   cd ai-topology-designer
   ```

3. **Run the Script**:
   ```
   python topology_designer.py --topology ring --num_nodes 8
   ```
   See `--help` for options.

## Usage

- **Generate and Evaluate**:
  - Ring: `--topology ring --num_nodes 16`
  - Torus: `--topology torus --rows 4 --cols 4`
  - Hypercube: `--topology hypercube --dim 4`
  - Fat-tree: `--topology fat_tree --pods 4 --servers_per_rack 4`

- **Metrics**: Outputs node/edge counts, degree, diameter, avg shortest path, bisection bandwidth approx.
- **Simulation**: Runs random traffic simulation for avg hops.
- **Visualization**: Displays graph with appropriate layout.

## Potential Improvements
- Integrate real AI workload simulations (e.g., all-reduce patterns).
- Add optimization algorithms for custom topologies.
- Support for link capacities and fault injection.
- Web-based UI for interactive design.

## Contributing
Fork, make changes, and PR. Focus on adding new topologies or metrics.

## License
MIT License.
