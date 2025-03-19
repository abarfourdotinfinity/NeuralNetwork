import matplotlib.pyplot as plt

def plot(num_epochs, losses):
    # ===== Plot the loss =====
    plt.plot(range(1, num_epochs + 1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid(True)
    plt.tight_layout()
    plt.show()