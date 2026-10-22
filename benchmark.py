import time
import matplotlib.pyplot as plt
from game import create_board
import algorithms_sol as algorithms  

def benchmark(board, algorithm_name, **kwargs):
    """Exécute un algorithme et retourne ses performances."""
    start_time = time.perf_counter()
    
    if algorithm_name == "minimax":
        res = algorithms.minimax(board, **kwargs)
    elif algorithm_name == "alphabeta":
        res = algorithms.alphabeta(board, **kwargs)
    elif algorithm_name == "mcts":
        res = algorithms.mcts(board, **kwargs)
    else:
        raise ValueError("Algorithme inconnu")
        
    return {
        "algorithm": algorithm_name,
        "config": kwargs,
        "move": res["move"],
        "score": res["score"],
        "nodes": res["nodes_explored"],
        "time": res["time"]
    }

def run_comparison():
    board = create_board()
    
   
    depths = [1, 2, 3, 4, 5]
    mm_nodes, ab_nodes = [], []
    
    print("--- Minimax vs Alpha-Beta ---")
    for d in depths:
        res_mm = benchmark(board, "minimax", depth=d)
        res_ab = benchmark(board, "alphabeta", depth=d)
        
        mm_nodes.append(res_mm["nodes"])
        ab_nodes.append(res_ab["nodes"])
        
        print(f"Profondeur {d} -> Minimax: {res_mm['nodes']} nœuds ({res_mm['time']}s) | Alpha-Beta: {res_ab['nodes']} nœuds ({res_ab['time']}s)")

    simulations_list = [100, 500, 1000, 2000]
    mcts_times = []
    
    print("\n--- MCTS Simulations ---")
    for sim in simulations_list:
        res_mcts = benchmark(board, "mcts", num_simulations=sim, seed=42)
        mcts_times.append(res_mcts["time"])
        print(f"Simulations {sim} -> Temps: {res_mcts['time']}s | Nœuds: {res_mcts['nodes']}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(depths, mm_nodes, marker='o', label='Minimax', color='red')
    ax1.plot(depths, ab_nodes, marker='s', label='Alpha-Beta', color='blue')
    ax1.set_title("Comparaison des nœuds explorés")
    ax1.set_xlabel("Profondeur de recherche")
    ax1.set_ylabel("Nombre de nœuds explorés")
    ax1.legend()
    ax1.grid(True)

    # Graphique 2 : Temps d'exécution en fonction des simulations MCTS
    ax2.plot(simulations_list, mcts_times, marker='^', label='MCTS', color='green')
    ax2.set_title("Temps d'exécution MCTS")
    ax2.set_xlabel("Nombre de simulations")
    ax2.set_ylabel("Temps (secondes)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("benchmark_results.png")
    print("\nGraphiques sauvegardés avec succès dans 'benchmark_results.png' !")
    plt.show()

if __name__ == "__main__":
    run_comparison()