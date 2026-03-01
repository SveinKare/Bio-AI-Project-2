import json
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def parse_solution(filepath):
    routes = []
    metadata = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith("Nurse capacity:"):
                metadata["capacity"] = int(line.split(":")[1].strip())
            elif line.startswith("Depot return time:"):
                metadata["return_time"] = int(line.split(":")[1].strip())
            elif line.startswith("Objective value"):
                metadata["objective"] = float(line.split(":")[1].strip())
            elif line.startswith("Nurse "):
                parts = line.split("Patients:")
                patient_str = parts[1].strip()
                patients = [int(p) for p in patient_str.split()]

                info_part = parts[0]
                duration = float(info_part.split("Duration:")[1].split("Demand:")[0].strip())
                demand = int(info_part.split("Demand:")[1].strip())

                routes.append({
                    "patients": patients,
                    "duration": duration,
                    "demand": demand,
                })
    return routes, metadata


def load_coords(json_path):
    with open(json_path) as f:
        data = json.load(f)
    depot = (data["depot"]["x_coord"], data["depot"]["y_coord"])
    patient_coords = {}
    for pid, pdata in data["patients"].items():
        patient_coords[int(pid)] = (pdata["x_coord"], pdata["y_coord"])
    return depot, patient_coords


def plot(routes, depot, patient_coords, metadata, output_path="solution_plot.png"):
    palette = list(mcolors.TABLEAU_COLORS.values())

    fig, ax = plt.subplots(figsize=(12, 10))

    for i, route in enumerate(routes):
        color = palette[i % len(palette)]
        pts = route["patients"]
        xs = [depot[0]] + [patient_coords[p][0] for p in pts] + [depot[0]]
        ys = [depot[1]] + [patient_coords[p][1] for p in pts] + [depot[1]]

        ax.plot(xs, ys, '-', color=color, linewidth=1.2, alpha=0.8)
        ax.plot(xs[1:-1], ys[1:-1], 's', color=color, markersize=5)

        label = f"Nurse {i+1} (dur={route['duration']:.0f}, dem={route['demand']})"
        ax.plot([], [], 's-', color=color, label=label, markersize=5)

    ax.plot(depot[0], depot[1], 'o', color='black', markersize=14, zorder=10)
    ax.plot(depot[0], depot[1], 'o', color='white', markersize=8, zorder=11)

    obj = metadata.get("objective", "?")
    ax.set_title(f"Best Solution — Total Duration: {obj}", fontsize=14, fontweight='bold')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    solution_file = sys.argv[1] if len(sys.argv) > 1 else "solution.txt"
    data_file = sys.argv[2] if len(sys.argv) > 2 else "data/train_2.json"

    routes, metadata = parse_solution(solution_file)
    depot, patient_coords = load_coords(data_file)
    plot(routes, depot, patient_coords, metadata)
