from PIL import Image
from torch.utils.data import Dataset


class SMGD(Dataset):
    def __init__(self, images_path, x, y, transform=None):
        self.images_path = images_path
        self.x = x
        self.y = y
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.images_path + self.x.iloc[index]
        img = Image.open(img_path).convert("RGB")
        label = self.y.iloc[index]

        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.x)