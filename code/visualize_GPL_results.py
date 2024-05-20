import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def draw_graph(results, base_model_f1, base_model, T5, set="test"):
    fig = plt.figure(figsize=(10,5))
    plt.title(f"GPL_{T5}{base_model} on {set} set")

    if(set=="test"):
        base_color = "r"
        model_color = "b"
    else:
        base_color = "tab:orange"
        model_color = "g"

    plt.plot(results[0,:], np.ones(len(results[0,:])) * base_model_f1, label=base_model, color=base_color)
    plt.plot(results[0,:], results[1,:], label='GPL versions', color=model_color)

    plt.xlabel("Number of steps")
    x_names = [f"{int(r/1000)}k" for r in results[0,:]]
    plt.xticks(results[0,:], x_names)

    plt.ylabel("F1-score")
    for i in range(14):
        plt.annotate(f"{results[1,i]:.4f}", results[:,i])

    plt.annotate(f"{base_model_f1:.4f}", (results[0,0],base_model_f1+0.0005), color=base_color)

    #plt.legend(loc='lower right')
    plt.legend(loc='upper right')
    plt.savefig(f"report/fig/GPL_{T5}_{base_model}_{set}", dpi=300)
    plt.show()


def read_data(line, n):
    results = np.zeros((2, n))
    i = 0
    for tup in line.split("), "):
        if i == 13:
            steps, f1 = tup[1:-1].split(", ")
        else:
            steps, f1 = tup[1:].split(", ")
        results[0, i] = float(steps)
        results[1, i] = float(f1)
        i+=1
    return results


if __name__=="__main__":
    #path = "data/results_gpl.txt"
    #base_model_f1 = 0.5637068742661127
    #base_model = "paraphrase"
    #T5 = ""
    # path = "data/results_gpl_boshko.txt"
    # base_model_f1 = 0.5637068742661127
    # base_model = "paraphrase"
    # T5 = "SLO_"
    # path = "data/results_gpl_boshko_sloberta.txt"
    # base_model_f1 = 0.5867172613808097
    # base_model = "SloBERTa"
    # T5 = "SLO_"


    # PARAPHRASE:
    path = "data/results_gpl_updated.txt"
    base_model_f1_train = 0.6547366815623796
    base_model_f1 = 0.5637068742661127
    base_model = "paraphrase"
    T5 = ""

    # path = "data/results_gpl_boshko_updated.txt"
    # base_model_f1_train = 0.6547366815623796
    # base_model_f1 = 0.5637068742661127
    # base_model = "paraphrase"
    # T5 = "SLO_"


    # SLOBERTA:

    # path = "data/results_gpl_boshko_sloberta_updated.txt"
    # base_model_f1_train = 0.6755744390188151
    # base_model_f1 = 0.5867172613808097
    # base_model = "SloBERTa"
    # T5 = "SLO_"

    # path = "data/results_gpl_sloberta_updated.txt"
    # base_model_f1_train = 0.6755744390188151
    # base_model_f1 = 0.5867172613808097
    # base_model = "SloBERTa"
    # T5 = ""

    results_train = np.zeros((2, 14))
    results_test = np.zeros((2, 14))

    with open(path) as f:
        line = f.readline()[1:-2]
        results_train = read_data(line, 14)

        line = f.readline()[1:-2]
        results_test = read_data(line, 14)

    draw_graph(results_train, base_model_f1_train, base_model, T5, set="train")
    draw_graph(results_test, base_model_f1, base_model, T5, set="test")
