import matplotlib.pyplot as plt
import seaborn as sns


def save_confusion_matrix(conf_matrix, save_path="confusion_matrix.png"):
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # 이미지 저장
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    # 메모리 누수 방지
    plt.close()

    print(f"Confusion matrix saved to {save_path}")
