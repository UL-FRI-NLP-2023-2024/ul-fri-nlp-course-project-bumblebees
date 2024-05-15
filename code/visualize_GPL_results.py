import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def draw_graph(results, base_model_f1):
    base_model = "SloBERTa"
    T5 = "Boshko"

    fig = plt.figure(figsize=(10,5))
    plt.title(f"GPL_{T5}_{base_model}")
    plt.plot(results[0,:], np.ones(len(results[0,:])) * base_model_f1, label=base_model, color='r')
    plt.plot(results[0,:], results[1,:], label='GPL versions')
    
    plt.xlabel("Number of steps")
    x_names = [f"{int(r/1000)}k" for r in results[0,:]]
    plt.xticks(results[0,:], x_names)

    plt.ylabel("F1-score")
    # ys = np.linspace(min(results[1,:]), max(results[1,:]), 8)
    # y_names = [f"{y:.4f}" for y in ys]
    # plt.yticks(ys, y_names)
    for i in range(14):
        plt.annotate(f"{results[1,i]:.4f}", results[:,i])

    plt.annotate(f"{base_model_f1:.4f}", (results[0,0],base_model_f1+0.001), color='r')

    plt.legend(loc='lower right')
    plt.savefig(f"report/fig/GPL_{T5}_{base_model}", dpi=300)
    plt.show()


if __name__=="__main__":
    path = "data/results_gpl_boshko_sloberta.txt"
    base_model_f1 = 0.5867172613808097

    results = np.zeros((2, 14))

    with open(path) as f:
        line = f.read()[1:-1]
        i = 0
        for tup in line.split("), "):
            if i == 13:
                steps, f1 = tup[1:-1].split(", ")
            else:
                steps, f1 = tup[1:].split(", ")
            results[0, i] = float(steps)
            results[1, i] = float(f1)
            i+=1

    draw_graph(results, base_model_f1)