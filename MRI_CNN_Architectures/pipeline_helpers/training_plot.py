import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_metrics(history, save_path):
    metrics = history.history
    epochs = range(1, len(metrics['accuracy']) + 1)

    fig, axs = plt.subplots(2, 2, figsize=(20, 12))
    plt.style.use('fivethirtyeight')

    def annotate_best(ax, y, label):
        idx = np.argmax(y) if 'accuracy' in label else np.argmin(y)
        ax.scatter(epochs[idx], y[idx], s=150, c='blue', label=f'Best epoch = {idx + 1}')
        return idx

    axs[0, 0].plot(epochs, metrics['loss'], 'r', label='Training Loss')
    axs[0, 0].plot(epochs, metrics['val_loss'], 'g', label='Validation Loss')
    annotate_best(axs[0, 0], metrics['val_loss'], 'loss')
    axs[0, 0].set_title('Loss'), axs[0, 0].legend(), axs[0, 0].grid()

    axs[0, 1].plot(epochs, metrics['accuracy'], 'r', label='Training Accuracy')
    axs[0, 1].plot(epochs, metrics['val_accuracy'], 'g', label='Validation Accuracy')
    annotate_best(axs[0, 1], metrics['val_accuracy'], 'accuracy')
    axs[0, 1].set_title('Accuracy'), axs[0, 1].legend(), axs[0, 1].grid()

    precision_key = next(k for k in metrics if 'precision' in k and not k.startswith('val_'))
    recall_key = next(k for k in metrics if 'recall' in k and not k.startswith('val_'))

    axs[1, 0].plot(epochs, metrics[precision_key], 'r', label='Training Precision')
    axs[1, 0].plot(epochs, metrics['val_' + precision_key], 'g', label='Validation Precision')
    annotate_best(axs[1, 0], metrics['val_' + precision_key], 'precision')
    axs[1, 0].set_title('Precision'), axs[1, 0].legend(), axs[1, 0].grid()

    axs[1, 1].plot(epochs, metrics[recall_key], 'r', label='Training Recall')
    axs[1, 1].plot(epochs, metrics['val_' + recall_key], 'g', label='Validation Recall')
    annotate_best(axs[1, 1], metrics['val_' + recall_key], 'recall')
    axs[1, 1].set_title('Recall'), axs[1, 1].legend(), axs[1, 1].grid()

    plt.suptitle('Model Training Metrics Over Epochs', fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)
    plt.close()
