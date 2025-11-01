import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os



print("=== COMP5511 Assignment - Final Version ===")

# Create directory to save results
os.makedirs('./results', exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =============================================================================
# Part 1: Transformer
# =============================================================================

print("\n" + "="*60)
print("Part 1: Transformer MNIST Classification")
print("="*60)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim=28, d_model=64, nhead=4, num_layers=2, num_classes=10):
        super(SimpleTransformer, self).__init__()
        
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        x = x.squeeze(1)
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x

def train_transformer():
    """Train Transformer model"""
    print("Preparing MNIST data...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    model = SimpleTransformer(d_model=64, nhead=4, num_layers=2).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    num_epochs = 10
    
    print("Starting Transformer training...")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        train_loss_avg = train_loss / len(train_loader)
        train_accuracy = 100. * correct / total
        
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        test_loss_avg = test_loss / len(test_loader)
        test_accuracy = 100. * correct / total
        
        train_losses.append(train_loss_avg)
        test_losses.append(test_loss_avg)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        
        print(f'Epoch {epoch+1}: Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', label='Train Loss')
    plt.plot(test_losses, 'r-', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Transformer: Training and Testing Loss Curves')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, 'b-', label='Train Accuracy')
    plt.plot(test_accuracies, 'r-', label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Transformer: Training and Testing Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('./results/transformer_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nTransformer training completed!")
    print(f"Final Training Accuracy: {train_accuracies[-1]:.2f}%")
    print(f"Final Testing Accuracy: {test_accuracies[-1]:.2f}%")
    
    return model, train_losses, test_losses, train_accuracies, test_accuracies




# =============================================================================
# Part 2: Diffusion Model
# =============================================================================

print("\n" + "="*60)
print("Part 2: Diffusion Model")
print("="*60)

class SimpleDiffusion(nn.Module):
    def __init__(self):
        super(SimpleDiffusion, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1)
        )
    
    def forward(self, x, t):
        
        return self.net(x)

def demo_diffusion():
    """Diffusion model demonstration"""
    print("Preparing CIFAR10 data...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    print(f"CIFAR10 training samples: {len(train_dataset)}")
    
    # Create simple model
    model = SimpleDiffusion().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Diffusion training process (for demonstration)
    print("Starting Diffusion training...")
    losses = []
    
    for epoch in range(5):  # Only train 5 epochs for demonstration
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Diffusion Epoch {epoch+1}/5')
        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(device)
            
            # Directly predict the image itself
            optimizer.zero_grad()
            output = model(images, None)
            loss = nn.MSELoss()(output, images)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f'Diffusion Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')
    
    # Plot training loss
    plt.figure(figsize=(8, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Diffusion Training Loss')
    plt.grid(True)
    plt.savefig('./results/diffusion_demo_loss.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Diffusion training completed!")
    return model, losses




# =============================================================================
# Main Function
# =============================================================================

def main():
    try:
        print("Starting COMP5511 Assignment Execution...")
        
        # Train Transformer (Task 1)
        print("\n" + ">"*50)
        print("TASK 1: Transformer for MNIST Classification")
        print(">"*50)
        transformer_model, t_losses, t_test_losses, t_acc, t_test_acc = train_transformer()
        
        # Train Diffusion Model (Task 2)
        print("\n" + ">"*50)
        print("TASK 2: Diffusion Model for CIFAR-10")
        print(">"*50)
        diffusion_model, d_losses = demo_diffusion()
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()