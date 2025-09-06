import matplotlib.pyplot as plt
import seaborn as sns

def plot_losses(results_df):
    plt.figure(figsize=(6, 15))

    plt.subplot(3, 1, 1)
    plt.plot(results_df['epoch'], results_df['train/box_loss'], label='Train Box Loss', color='blue')
    plt.plot(results_df['epoch'], results_df['val/box_loss'], label='Val Box Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Box Loss')
    plt.title('Box Loss vs. Epochs')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(results_df['epoch'], results_df['train/dfl_loss'], label='Train DFL', color='blue')
    plt.plot(results_df['epoch'], results_df['val/dfl_loss'], label='Val DFL', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Distribution Focal Loss')
    plt.title('Distribution Focal Loss vs. Epochs')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(results_df['epoch'], results_df['train/cls_loss'], label='Train Classification Loss', color='blue')
    plt.plot(results_df['epoch'], results_df['val/cls_loss'], label='Val Classification Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Classification Loss')
    plt.title('Classification Loss vs. Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_map(results_df):
    plt.figure(figsize=(6, 5))
    plt.plot(results_df['epoch'], results_df['metrics/mAP50(B)'], label='mAP@0.5', color='green')
    plt.plot(results_df['epoch'], results_df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('mAP')
    plt.title('Mean Average Precision vs. Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_precision_recall(results_df):
    plt.figure(figsize=(6, 5))
    plt.plot(results_df['epoch'], results_df['metrics/precision(B)'], label='Precision', color='purple')
    plt.plot(results_df['epoch'], results_df['metrics/recall(B)'], label='Recall', color='brown')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.title('Precision and Recall vs. Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(metrics):
    conf_matrix = metrics.confusion_matrix.matrix
    print("Confusion Matrix:")
    print(conf_matrix)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='.0f', cmap='Blues',
                xticklabels=metrics.names.values(),
                yticklabels=metrics.names.values())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()