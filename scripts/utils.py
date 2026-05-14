from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average="weighted"),
        "recall": recall_score(labels, preds, average="weighted"),
    }

def plotter(title,xlabel,ylabel,grid,file):
    def decorator(func):
        def wrapper(*args,**kwargs):
            plt.figure()
            plt.title(title)
            func(*args,**kwargs)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            if (handles:= plt.gca().get_legend_handles_labels()[0]):
                plt.legend()
            plt.grid(grid)
            result_folder = kwargs["result_folder"] 
            plt.savefig(str(result_folder / file), dpi=300, bbox_inches="tight")
            plt.close()
        return wrapper
    return decorator




@plotter(title="ROC_CURVE",xlabel="FPR",ylabel="TPR",grid=True,file="roc_curve.png")
def plot_roc_curve(fpr,tpr,roc_auc,result_folder):
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")


@plotter(title="Precision–Recall Curve",xlabel="recall",ylabel="precision",grid=True,file="pr_curve.png")
def plot_pr_curve(recall,precision,result_folder):
        plt.plot(recall, precision)




@plotter(title="Probability Distribution",xlabel="AI probability",ylabel="Desnity",grid=True,file="prob_distribution.png")
def plot_pb_distro(y_true, y_probs, labels, result_folder):

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    human_probs = y_probs[y_true == 0]
    ai_probs = y_probs[y_true == 1]

    sns.histplot(
        human_probs,
        bins=30,
        label=labels[0]
    )

    sns.histplot(
        ai_probs,
        bins=30,
        label=labels[1]
    )

@plotter(title="Confusion Matrix",xlabel="Predicted",ylabel="True",grid=False,file="confusion_matrix.png")
def plot_confusion_matrix(y_true,y_pred,result_folder):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")