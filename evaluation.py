# --- At the top (after imports) ---
from sklearn.model_selection import train_test_split

# --- Data Preparation ---
# Split for future validation if needed
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Create dataset and loaders
train_dataset = TemperatureOrdinalDataset(train_df, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Evaluation (non-shuffled for consistent ordering)
eval_dataset = TemperatureOrdinalDataset(df, transform=transform)
eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False)

# --- Model Definition (same as yours) ---
class CoralResNet50(nn.Module):
    def __init__(self, num_classes=3):
        super(CoralResNet50, self).__init__()
        from torchvision.models import resnet50
        self.backbone = resnet50(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.coral_head = nn.Linear(in_features, num_classes - 1)
        self.reg_head = nn.Linear(in_features, 1)

    def forward(self, x):
        features = self.backbone(x)
        coral_logits = self.coral_head(features)
        reg_output = self.reg_head(features).squeeze(1)
        return coral_logits, reg_output

# --- Training (same as before) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CoralResNet50(num_classes=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

losses = train_model(model, train_loader, optimizer, device, num_epochs=20, alpha=0.6)

plt.plot(losses, marker='o')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# --- Save Model ---
torch.save(model.state_dict(), "coral_resnet50_temp.pth")
print("Model saved to coral_resnet50_temp.pth")

# --- Evaluate Model ---
plot_confusion_matrix(model, eval_loader, device)

def plot_residuals(model, dataloader, device):
    model.eval()
    true_temps, pred_temps = [], []
    with torch.no_grad():
        for images, _, real_temp in dataloader:
            images = images.to(device)
            _, reg_output = model(images)
            true_temps.extend(real_temp.numpy())
            pred_temps.extend(reg_output.cpu().numpy())

    residuals = np.array(true_temps) - np.array(pred_temps)
    mae = mean_absolute_error(true_temps, pred_temps)

    plt.figure(figsize=(8, 5))
    plt.scatter(true_temps, residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("True Temperature")
    plt.ylabel("Residuals (True - Predicted)")
    plt.title(f"Residual Plot (MAE: {mae:.2f})")
    plt.grid(True)
    plt.show()

plot_residuals(model, eval_loader, device)
