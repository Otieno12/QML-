import os
import torch
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224))
])


def load_images_from_folder(folder_path):
    image_files = [x for x in os.listdir(folder_path) if x.endswith('.jpeg') or x.endswith('.jpg')]
    images = []
    for file in image_files:
        img_path = os.path.join(folder_path, file)
        image = Image.open(img_path)
        image = transform(image)
        images.append(image)
    return torch.stack(images)

folder_1 = "lc_pictures/8cb(segunda ordem)/8CB_1/"



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

images_1 = load_images_from_folder(folder_1).to(device)



def load_images_and_temperatures(load_path, image_size=224):
    """
    Loads images and their corresponding temperatures from filenames.
    Returns preprocessed images (for ResNet) and temperatures as float labels.
    """
    image_paths = sorted(glob.glob(os.path.join(load_path, "*.jpeg")))
    cut_index = len(load_path) + 1

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    images = []
    temperatures = []
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        images.append(transform(img))
        temp = float(path[cut_index:-5])
        temperatures.append(temp)

    return images, temperatures






class CoralResNet50(nn.Module):
    def __init__(self, num_classes):
        super(CoralResNet50, self).__init__()
        self.backbone = resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()
        self.fc = nn.Linear(2048, num_classes - 1)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.fc(features)
        return logits





def coral_loss(logits, levels, importance_weights=None):
    """
    Computes CORAL loss.
    """
    val = - torch.sum(
        (levels * torch.log(torch.sigmoid(logits)) + (1 - levels) * torch.log(1 - torch.sigmoid(logits))),
        dim=1
    )
    if importance_weights is not None:
        val = val * importance_weights
    return torch.mean(val)





def temperatures_to_levels(temps, class_thresholds):
    """
    Converts temperature values to ordinal levels (binary vector of K-1).
    """
    levels = []
    for t in temps:
        level = [1 if t > thresh else 0 for thresh in class_thresholds[:-1]]
        levels.append(level)
    return torch.tensor(levels, dtype=torch.float)



def logits_to_levels(logits):
    """
    Converts logits into predicted ordinal levels.
    Each level is 1 if sigmoid(logit) > 0.5 else 0.
    """
    probas = torch.sigmoid(logits)
    return (probas > 0.5).int()





def levels_to_class_index(levels):
    """
    Converts binary levels into class index (temperature bin index).
    """
    return torch.sum(levels, dim=1)





def class_index_to_temperature(class_indices, class_thresholds):
    """
    Converts class indices to midpoints of corresponding temperature bins.
    """
    bin_centers = 0.5 * (class_thresholds[:-1] + class_thresholds[1:])
    bin_centers = torch.tensor(bin_centers)
    return bin_centers[class_indices]




def predict_temperature_from_logits(logits, class_thresholds):
    """
    Full pipeline: logits -> levels -> class index -> temperature.
    """
    levels = logits_to_levels(logits)
    class_indices = levels_to_class_index(levels)
    temperatures = class_index_to_temperature(class_indices, class_thresholds)
    return temperatures






num_classes = 3  
model = CoralResNet50(num_classes=num_classes)





folder_1 = "lc_pictures/8cb(segunda ordem)/8CB_1/"
image_paths = [f for f in os.listdir(folder_1) if f.endswith(".jpeg")]


df = pd.DataFrame({
    "image_path": [os.path.join(folder_1, f) for f in image_paths],
    "temperature": [float(f.replace(".jpeg", "")) for f in image_paths]
})

def assign_label(temp):
    if temp >= 35:
        return 0  
    elif 25 <= temp < 35:
        return 1  
    elif temp < 25:
        return 2 

df["label"] = df["temperature"].apply(assign_label)





class TemperatureOrdinalDataset(Dataset):
    def __init__(self, dataframe, transform=None, num_classes=None):
        self.df = dataframe
        self.transform = transform
        self.num_classes = num_classes or self.df['label'].nunique()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = int(row['label'])
        levels = torch.tensor([1]*label + [0]*(self.num_classes - 1 - label), dtype=torch.float32)

        return image, levels





transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataset = TemperatureOrdinalDataset(df, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)






def coral_loss(logits, levels, importance_weights=None):
    """
    logits: shape (batch_size, num_classes - 1)
    levels: same shape, binary 1 if class >= k
    """
    val = F.binary_cross_entropy_with_logits(logits, levels, weight=importance_weights, reduction='mean')
    return val







class CoralResNet50(nn.Module):
    def __init__(self, num_classes=3):
        super(CoralResNet50, self).__init__()
        self.backbone = resnet50(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.coral_head = nn.Linear(in_features, num_classes - 1)

    def forward(self, x):
        x = self.backbone(x)
        logits = self.coral_head(x)
        return logits






num_classes = df['label'].nunique()
model = CoralResNet50(num_classes=num_classes).to(device)






def coral_logits_to_label(logits):
    probas = torch.sigmoid(logits)
    return torch.sum(probas > 0.5, dim=1)






model = CoralResNet50(num_classes=num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for images, levels in dataloader:
    images = images.to(device)
    levels = levels.to(device)

    optimizer.zero_grad()
    logits = model(images)
    loss = coral_loss(logits, levels)
    loss.backward()
    optimizer.step()




import matplotlib.pyplot as plt

def train_model(model, dataloader, optimizer, num_epochs=10, device='cuda'):
    model.train()
    loss_history = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for images, levels in dataloader:
            images = images.to(device)
            levels = levels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = coral_loss(logits, levels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * images.size(0)

        avg_loss = epoch_loss / len(dataloader.dataset)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")

    return loss_history






model = CoralResNet50(num_classes=num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

losses = train_model(model, dataloader, optimizer, num_epochs=10, device=device)

plt.figure(figsize=(8,5))
plt.plot(losses, marker='o')
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()


from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])



class TemperatureOrdinalDataset(Dataset):
    def __init__(self, dataframe, transform=None, num_classes=None):
        self.df = dataframe
        self.transform = transform
        self.num_classes = num_classes or self.df['label'].nunique()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = int(row['label'])
        levels = torch.tensor([1]*label + [0]*(self.num_classes - 1 - label), dtype=torch.float32)
        temp = row['temperature']

        return image, levels, temp



train_dataset = TemperatureOrdinalDataset(train_df, transform=transform)
val_dataset = TemperatureOrdinalDataset(val_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)




def evaluate_real_temperatures(model, dataloader, class_thresholds, device='cuda'):
    model.eval()
    pred_temps = []
    true_temps = []

    with torch.no_grad():
        for images, levels, real_temp in dataloader:
            images = images.to(device)
            logits = model(images)

            predicted = predict_temperature_from_logits(logits, class_thresholds)
            pred_temps.extend(predicted.cpu().numpy())
            true_temps.extend(real_temp.numpy())

    mae = mean_absolute_error(true_temps, pred_temps)
    rae = np.sum(np.abs(np.array(true_temps) - np.array(pred_temps))) / \
          np.sum(np.abs(np.array(true_temps) - np.mean(true_temps)))

    print(f"\nReal Temp Evaluation:\nMAE: {mae:.4f}°C\nRAE: {rae:.4f}")
    return mae, rae



class_thresholds = [25.0, 35.0, 100.0]  


evaluate_real_temperatures(model, val_loader, class_thresholds, device=device)






