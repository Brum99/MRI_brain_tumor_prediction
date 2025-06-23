# evaluation.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import plot_model
from PIL import Image


def save_model_architecture(model, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    arch_path = os.path.join(save_dir, filename)
    plot_model(model, to_file=arch_path, show_shapes=True, show_layer_names=True)
    print(f"Model architecture image saved to: {arch_path}")


def save_model_weights(model, save_dir, base_filename):
    os.makedirs(save_dir, exist_ok=True)
    h5_path = os.path.join(save_dir, base_filename + ".h5")
    keras_path = os.path.join(save_dir, base_filename + ".keras")
    model.save(h5_path)
    model.save(keras_path)
    print(f"Model saved as .h5 and .keras in: {save_dir}")


def plot_training_metrics(history, save_path):
    acc = history.history['accuracy']
    loss = history.history['loss']
    prec_key = [k for k in history.history.keys() if 'precision' in k][0]
    recall_key = [k for k in history.history.keys() if 'recall' in k][0]

    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']
    val_prec = history.history[f'val_{prec_key}']
    val_recall = history.history[f'val_{recall_key}']

    tr_prec = history.history[prec_key]
    tr_recall = history.history[recall_key]

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(20, 12))
    plt.style.use('fivethirtyeight')

    plt.subplot(2, 2, 1)
    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'g', label='Validation Loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, acc, 'r', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'g', label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs, tr_prec, 'r', label='Training Precision')
    plt.plot(epochs, val_prec, 'g', label='Validation Precision')
    plt.title('Precision')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(epochs, tr_recall, 'r', label='Training Recall')
    plt.plot(epochs, val_recall, 'g', label='Validation Recall')
    plt.title('Recall')
    plt.legend()

    plt.suptitle("Training Metrics", fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training metrics plot saved to: {save_path}")


def evaluate_and_save_results(model, train_gen, valid_gen, test_gen, save_dir):
    results = []
    sets = {'Train': train_gen, 'Validation': valid_gen, 'Test': test_gen}

    for name, gen in sets.items():
        loss, acc, prec, recall = model.evaluate(gen, verbose=1)
        results.append(f"{name} Set")
        results.append(f"Loss      : {loss:.4f}")
        results.append(f"Accuracy  : {acc * 100:.2f}%")
        results.append(f"Precision : {prec * 100:.2f}%")
        results.append(f"Recall    : {recall * 100:.2f}%")
        results.append('-' * 40)

    for line in results:
        print(line)

    os.makedirs(save_dir, exist_ok=True)
    results_path = os.path.join(save_dir, "evaluation_results.txt")
    with open(results_path, 'w') as f:
        f.write('\n'.join(results))


def save_confusion_matrix(model, test_gen, class_labels, save_path):
    preds = model.predict(test_gen)
    y_pred = np.argmax(preds, axis=1)
    cm = confusion_matrix(test_gen.classes, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")

    return classification_report(test_gen.classes, y_pred, target_names=class_labels)


def predict_and_save_plot(model, img_path, class_labels, save_dir):
    img = Image.open(img_path).resize((299, 299))
    img_array = np.asarray(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    probs = list(predictions[0])

    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Input MRI", fontsize=14)

    plt.subplot(2, 1, 2)
    bars = plt.barh(class_labels, probs)
    plt.xlabel('Probability', fontsize=14)
    plt.title("Model Prediction", fontsize=14)
    plt.gca().bar_label(bars, fmt='%.2f')

    os.makedirs(save_dir, exist_ok=True)
    file_name = os.path.basename(img_path).split('.')[0] + "_prediction.png"
    save_path = os.path.join(save_dir, file_name)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Saved prediction image to: {save_path}")