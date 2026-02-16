import torch
from torch.utils.data import DataLoader
from dataset import LungDataset
from model import get_model
from monai.losses import DiceLoss
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = LungDataset("processed/images", "processed/masks")
loader = DataLoader(dataset, batch_size=2, shuffle=True)

model = get_model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = DiceLoss(sigmoid=True)

epochs = 50

for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for imgs, masks in tqdm(loader):
        imgs, masks = imgs.to(device), masks.to(device)

        outputs = model(imgs)
        loss = loss_fn(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(loader)}")

torch.save(model.state_dict(), "lung_unet.pth")

