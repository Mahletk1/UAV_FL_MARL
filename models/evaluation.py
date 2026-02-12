import torch
from torch.utils.data import DataLoader

def test_model(model, dataset_test, args):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = torch.nn.CrossEntropyLoss()
    loader = DataLoader(dataset_test, batch_size=args.bs, shuffle=False)

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(args.device), y.to(args.device)
            out = model(x)
            loss = criterion(out, y)
            loss_sum += loss.item() * y.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = correct / total
    avg_loss = loss_sum / total
    model.train()
    return acc, avg_loss
