# train.py
import numpy as np
import yaml
import wandb
import argparse
from tqdm import tqdm
from data.dataset import load_data, DataLoader
from models.feedforward_nn import FeedForwardNN
from utils.optimizer_utils import get_optimizer
from utils.losses import cross_entropy_grad, softmax


# ---------------------------------------------------------
# Utility to load YAML config
# ---------------------------------------------------------
def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------
# Compute accuracy
# ---------------------------------------------------------
def accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


# ---------------------------------------------------------
# Training loop
# ---------------------------------------------------------
def train(config=None):
    # Initialize wandb
    wandb.init(config=config, project=config["project"], entity=config["entity"])
    config = wandb.config

    # Load data
    X_train, y_train, X_val, y_val, _, _ = load_data(flatten=True)

    # Unpack config
    model_cfg = config["model"]
    train_cfg = config["training"]
    data_cfg = config["data"]

    # Model and optimizer
    model = FeedForwardNN(
        input_dim=model_cfg["input_dim"],
        hidden_sizes=model_cfg["hidden_sizes"],
        output_dim=model_cfg["output_dim"],
        activation=model_cfg["activation"],
        weight_init=model_cfg["weight_init"],
        weight_decay=model_cfg["weight_decay"],
        seed=train_cfg["seed"],
    )

    optimizer = get_optimizer(
        train_cfg["optimizer"],
        lr=train_cfg["learning_rate"]
    )

    # MODIFIED: Create a DataLoader instance for training data.
    train_loader = DataLoader(
        X_train,
        y_train,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        seed=train_cfg["seed"]
    )
    # Training loop
    for epoch in range(train_cfg["epochs"]):
        epoch_loss = 0
        epoch_acc = 0

        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg['epochs']}"):
            # Forward + Gradients
            dWs, dbs = model.compute_gradients(X_batch, y_batch, loss=train_cfg["loss"])
            params = model.weights + model.biases
            grads = dWs + dbs
            optimizer.update(params, grads)

            # Replace updated params
            model.weights = params[:len(model.weights)]
            model.biases = params[len(model.weights):]

            # Compute batch loss and accuracy
            probs = model.predict_proba(X_batch)
            preds = np.argmax(probs, axis=1)
            acc = accuracy(preds, y_batch)
            loss = -np.mean(np.log(probs[np.arange(len(y_batch)), y_batch] + 1e-8))

            epoch_loss += loss
            epoch_acc += acc

        # Validation
        val_probs = model.predict_proba(X_val)
        val_preds = np.argmax(val_probs, axis=1)
        val_acc = accuracy(val_preds, y_val)
        val_loss = -np.mean(np.log(val_probs[np.arange(len(y_val)), y_val] + 1e-8))

        #Use len(train_loader) instead of the old n_batches variable.
        train_loss_avg = epoch_loss / len(train_loader)
        train_acc_avg = epoch_acc / len(train_loader)

        # Log to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss_avg,
            "train_acc": train_acc_avg,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        })
        print(f"Epoch {epoch+1}: Train Acc={train_acc_avg:.3f}, Val Acc={val_acc:.3f}")

    wandb.finish()


# ---------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    train(base_cfg)
