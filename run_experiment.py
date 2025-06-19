import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

# Configuración básica
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 64
batch_size = 128
epochs = 20
save_interval = 4

# Crear directorio para guardar resultados
os.makedirs("results", exist_ok=True)

# DataFrame para guardar las pérdidas
losses_df = pd.DataFrame(columns=['epoch', 'discriminator_loss', 'generator_loss'])

# Transformación y carga de datos MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # escala de -1 a 1
])

dataloader = DataLoader(
    datasets.MNIST(root="./data", train=True, download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True
)

# Generador
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Tanh()  # salida en el rango [-1, 1]
        )

    def forward(self, z):
        return self.net(z)

# Discriminador
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Inicialización de modelos y optimizadores
G = Generator().to(device)
D = Discriminator().to(device)

loss_fn = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=0.0002)
optimizer_D = optim.Adam(D.parameters(), lr=0.0002)

# Función para generar y guardar muestras
def save_generated_samples(epoch, generator, latent_dim=64, device=device):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(16, latent_dim).to(device)
        samples = generator(z).reshape(-1, 1, 28, 28).cpu()
        
        # Crear una cuadrícula 4x4 de imágenes
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i in range(16):
            row, col = i // 4, i % 4
            axes[row, col].imshow(samples[i].squeeze(), cmap='gray')
            axes[row, col].axis('off')
        
        plt.suptitle(f'Muestras generadas - Época {epoch}')
        plt.tight_layout()
        
        # Guardar la imagen
        plt.savefig(f'results/generated_epoch_{epoch:03d}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Imágenes guardadas en results/generated_epoch_{epoch:03d}.png")

# Entrenamiento
for epoch in range(epochs):
    for real_imgs, _ in dataloader:
        real_imgs = real_imgs.view(-1, 784).to(device)
        batch_size = real_imgs.size(0)

        # 🏷️ Etiquetas reales y falsas
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # ---------------------
        # 1. Entrenar el Discriminador
        # ---------------------
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_imgs = G(z)

        real_loss = loss_fn(D(real_imgs), real_labels)
        fake_loss = loss_fn(D(fake_imgs.detach()), fake_labels)
        d_loss = real_loss + fake_loss

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # ---------------------
        # 2. Entrenar el Generador
        # ---------------------
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_imgs = G(z)
        g_loss = loss_fn(D(fake_imgs), real_labels)  # queremos que D piense que son reales

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

    print(f"Época {epoch+1}/{epochs} | Pérdida D: {d_loss.item():.4f} | Pérdida G: {g_loss.item():.4f}")
    
    # Generar y guardar muestras cada 10 épocas
    if (epoch + 1) % save_interval == 0:
        save_generated_samples(epoch + 1, G)

    # Guardar pérdidas en el DataFrame
    new_row = pd.DataFrame({'epoch': [epoch + 1], 'discriminator_loss': [d_loss.item()], 'generator_loss': [g_loss.item()]})
    losses_df = pd.concat([losses_df, new_row], ignore_index=True)

# Mostrar resultados finales
def show_generated():
    G.eval()
    with torch.no_grad():
        z = torch.randn(16, latent_dim).to(device)
        samples = G(z).reshape(-1, 1, 28, 28).cpu()
        grid = torch.cat([s.squeeze(0) for s in samples], dim=1)
        plt.imshow(grid, cmap='gray')
        plt.axis("off")
        plt.title("Muestras generadas finales")
        plt.show()

print("\nEntrenamiento completado! Mostrando resultados finales...")
show_generated()

# Guardar DataFrame de pérdidas
losses_df.to_csv('results/losses.csv', index=False)

# Función para generar gráficos de pérdidas
def plot_losses(losses_df):
    plt.figure(figsize=(15, 5))
    
    # Gráfico 1: Pérdidas individuales
    plt.subplot(1, 3, 1)
    plt.plot(losses_df['epoch'], losses_df['discriminator_loss'], label='Discriminador', color='blue')
    plt.plot(losses_df['epoch'], losses_df['generator_loss'], label='Generador', color='red')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.title('Evolución de las Pérdidas')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gráfico 2: Pérdida del Discriminador
    plt.subplot(1, 3, 2)
    plt.plot(losses_df['epoch'], losses_df['discriminator_loss'], color='blue', linewidth=2)
    plt.xlabel('Época')
    plt.ylabel('Pérdida del Discriminador')
    plt.title('Pérdida del Discriminador')
    plt.grid(True, alpha=0.3)
    
    # Gráfico 3: Pérdida del Generador
    plt.subplot(1, 3, 3)
    plt.plot(losses_df['epoch'], losses_df['generator_loss'], color='red', linewidth=2)
    plt.xlabel('Época')
    plt.ylabel('Pérdida del Generador')
    plt.title('Pérdida del Generador')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/loss_evolution.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Gráficos de pérdidas guardados en results/loss_evolution.png")

# Función para generar gráfico de convergencia
def plot_convergence(losses_df):
    plt.figure(figsize=(10, 6))
    
    # Calcular la diferencia entre pérdidas para ver convergencia
    loss_diff = np.abs(losses_df['discriminator_loss'] - losses_df['generator_loss'])
    
    plt.plot(losses_df['epoch'], loss_diff, color='green', linewidth=2)
    plt.xlabel('Época')
    plt.ylabel('|Pérdida D - Pérdida G|')
    plt.title('Convergencia del GAN\n(Diferencia entre pérdidas)')
    plt.grid(True, alpha=0.3)
    
    # Agregar línea horizontal en y=0.5 para referencia
    plt.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Umbral de convergencia')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/convergence.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Gráfico de convergencia guardado en results/convergence.png")

# Generar todos los gráficos
print("\nGenerando gráficos de análisis...")
plot_losses(losses_df)
plot_convergence(losses_df)

print(f"\nResumen del entrenamiento:")
print(f"Pérdida final del Discriminador: {losses_df['discriminator_loss'].iloc[-1]:.4f}")
print(f"Pérdida final del Generador: {losses_df['generator_loss'].iloc[-1]:.4f}")
print(f"Pérdidas guardadas en: results/losses.csv")
