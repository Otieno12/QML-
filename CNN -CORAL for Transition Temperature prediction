import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CoralOrdinalLayer(nn.Module):
    def __init__(self, num_classes):
        super(CoralOrdinalLayer, self).__init__()
        self.num_classes = num_classes
        self.bias = nn.Parameter(torch.zeros(num_classes - 1))  

    def forward(self, x):
        logits = x.unsqueeze(2) + self.bias 
        logits = logits.squeeze(1)  
        probas = torch.sigmoid(logits)
        return logits, probas


class CoralCNN(nn.Module):
    def __init__(self, num_classes):
        super(CoralCNN, self).__init__()
        self.num_classes = num_classes

       
        base_model = models.resnet50(pretrained=True)
        modules = list(base_model.children())[:-1]  
        self.feature_extractor = nn.Sequential(*modules)
        self.fc = nn.Linear(base_model.fc.in_features, 1)  

        # CORAL ordinal layer
        self.coral = CoralOrdinalLayer(num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)  
        x = x.view(x.size(0), -1)      
        g = self.fc(x)                 
        logits, probas = self.coral(g)
        return logits, probas


def predict_rank(probas):
    """
    Predict the rank based on probas > 0.5 threshold
    """
    return torch.sum(probas > 0.5, dim=1)

num_classes = 3
model = CoralCNN(num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
epochs = 200

for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits, probas = model(images)
        loss = coral_loss(logits, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")



from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

model.eval()
true_labels, predicted_ranks = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        _, probas = model(images)
        preds = predict_rank(probas)

        true_labels.extend(labels.cpu().numpy())
        predicted_ranks.extend(preds.cpu().numpy())

mae = mean_absolute_error(true_labels, predicted_ranks)
rmse = np.sqrt(mean_squared_error(true_labels, predicted_ranks))

print(f"Test MAE: {mae:.4f}, RMSE: {rmse:.4f}")
