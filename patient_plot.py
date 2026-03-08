import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import sys

# ── Config ────────────────────────────────────────────────────────────────────
SOLUTION_FILE = "solution_1.txt"
DATA_FILE     = "test_instance_1.json"
# ─────────────────────────────────────────────────────────────────────────────

def load_data(data_file):
    with open(data_file) as f:
        data = json.load(f)
    depot = (data["depot"]["x_coord"], data["depot"]["y_coord"])
    patients = {
        int(k): (v["x_coord"], v["y_coord"])
        for k, v in data["patients"].items()
    }
    return depot, patients

def parse_routes(solution_file):
    """Read the gene from the solution file and split into routes."""
    with open(solution_file) as f:
        for line in f:
            line = line.strip()
            # Find the line that looks like a gene (all integers)
            tokens = line.split()
            if all(t.lstrip('-').isdigit() for t in tokens) and len(tokens) > 5:
                gene = list(map(int, tokens))
                break

    # Split on zeros into routes, skip empty ones
    routes = []
    current = []
    for val in gene:
        if val == 0:
            if current:
                routes.append(current)
                current = []
        else:
            current.append(val)
    if current:
        routes.append(current)

    # Filter out truly empty routes
    routes = [r for r in routes if len(r) > 0]
    return routes

def plot_routes(routes, depot, patients):
    fig, ax = plt.subplots(figsize=(14, 12))

    # Generate distinct colors per route
    colors = cm.tab20(np.linspace(0, 1, max(len(routes), 1)))

    for route_idx, (route, color) in enumerate(zip(routes, colors)):
        # Build coordinate sequence: depot -> patients -> depot
        coords = [depot] + [patients[p] for p in route] + [depot]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]

        # Draw route line
        ax.plot(xs, ys, color=color, linewidth=1.5, alpha=0.7)

        # Draw arrows to show direction
        for i in range(len(coords) - 1):
            dx = coords[i+1][0] - coords[i][0]
            dy = coords[i+1][1] - coords[i][1]
            ax.annotate(
                "",
                xy=(coords[i+1][0], coords[i+1][1]),
                xytext=(coords[i][0], coords[i][1]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=1.2,
                    mutation_scale=10
                )
            )

        # Label each patient with their ID
        for p in route:
            x, y = patients[p]
            ax.text(x + 0.4, y + 0.4, str(p),
                    fontsize=6, color=color,
                    fontweight='bold', alpha=0.9)

        # Mark patient nodes
        px = [patients[p][0] for p in route]
        py = [patients[p][1] for p in route]
        ax.scatter(px, py, color=color, s=30, zorder=3)

    # Mark depot prominently
    ax.scatter(*depot, color='black', s=200, zorder=5, marker='*')
    ax.text(depot[0] + 0.5, depot[1] + 0.5, "DEPOT",
            fontsize=9, fontweight='bold', color='black')

    ax.set_title(f"Routes ({len(routes)} nurses)", fontsize=14)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("routes.png", dpi=150)
    print(f"Saved to routes.png ({len(routes)} routes plotted)")
    plt.show()

if __name__ == "__main__":
    sol_file  = sys.argv[1] if len(sys.argv) > 1 else SOLUTION_FILE
    data_file = sys.argv[2] if len(sys.argv) > 2 else DATA_FILE

    depot, patients = load_data(data_file)
    routes = parse_routes(sol_file)

    print(f"Found {len(routes)} routes")
    for i, r in enumerate(routes):
        print(f"  Route {i+1}: {r}")

    plot_routes(routes, depot, patients)