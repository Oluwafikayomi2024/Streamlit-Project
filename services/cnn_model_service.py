import torch
from torchvision import transforms
from PIL import Image

class CNNModel_Service:
    def __init__(self, model, labels):
        self.model = model
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def predict(self, image_path):
        img = Image.open(image_path).convert("RGB")
        img_t = self.transform(img).unsqueeze(0)

        with torch.no_grad():
            output = self.model(img_t)
            pred = torch.argmax(output, dim=1).item()

        return self.labels[pred]
