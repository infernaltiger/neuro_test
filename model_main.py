import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from PIL import Image

import numpy as np
from data_refactor import crop_center_square, TRANSFORM
from config import EPOCHS, LEARNING_RATE, PATIENCE, DEVISE

class DocumentOrientationModel(nn.Module):
    def __init__(self):
        super(DocumentOrientationModel, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.base_model(x))
class Main_Model_Functions:
    def __init__(self, model, device=DEVISE):
        self.model = model
        self.device = device
        self.model.to(self.device)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Модель сохранена в {path}")

    def load_model(self,path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        print(f"Модель загружена из {path}")

    def predict(self, image_path):
        self.model.eval()
        image = Image.open(image_path).convert('RGB')
        image = crop_center_square(image)
        image = TRANSFORM(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image)
            prediction = torch.round(output).item()
            res = "Перевернут" if prediction == 1 else "Нормально"
        return prediction, output, res

    def test_model(self, test_loader):
        criterion = nn.BCELoss()
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(DEVISE)
                labels = labels.float().to(DEVISE)

                outputs = self.model(images).squeeze()

                test_loss += criterion(outputs, labels).item()

                predicted = torch.round(outputs)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                test_loss /= len(test_loader)
                accuracy = 100 * correct / total
            print(f'Testing Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
            return test_loss, accuracy

    def train_model(self, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE, patience=PATIENCE):
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        best_val_loss = np.inf
        epochs_without_improvement = 0
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            for images, labels in train_loader:
                images = images.to(DEVISE)
                labels = labels.float().to(DEVISE)

                # Обнуляем градиенты
                optimizer.zero_grad()

                # Прямой проход
                outputs = self.model(images).squeeze()
                loss = criterion(outputs, labels)

                # Обратный проход и оптимизация
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {running_loss / len(train_loader):.4f}')
            # Валидация
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(DEVISE)
                    labels = labels.float().to(DEVISE)

                    # Прямой проход
                    outputs = self.model(images).squeeze()

                    # Вычисляем loss и точность
                    val_loss += criterion(outputs, labels).item()

                    predicted = torch.round(outputs)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_loss /= len(val_loader)
            accuracy = 100 * correct / total
            print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')

            # Ранняя остановка
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                self.save_model("doc_orient_model_central_square_best.pth")  # Сохраняем лучшую модель
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Ранняя остановка на эпохе {epoch + 1}")
                    break
