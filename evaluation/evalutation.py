import torch
from sklearn.metrics import accuracy_score

def accuracy(
    X: torch.Tensor,
    y: torch.Tensor,
    w,
    g: torch.Tensor = None,
) -> float:
    if isinstance(w, torch.Tensor):
        # Logistic regression 
        if g is None:
            g = torch.ones_like(w) 
        preds = torch.sigmoid((X * g) @ w) > 0.5
        return accuracy_score(y.numpy(), preds.numpy())
    else:
        # Neural network
        model = w
        model.eval()
        with torch.no_grad(): 
            X_mod = X * g if g is not None else X
            outputs = model(X_mod)
            if outputs.shape[1] == 1:
                preds = torch.sigmoid(outputs) > 0.5
            else:
                preds = torch.argmax(outputs, dim=1)
            if y.ndim > 1 and y.shape[1] == 1:
                y = y.squeeze(1)
            return accuracy_score(y.cpu().numpy(), preds.cpu().numpy())