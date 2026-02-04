from torch.utils.data import Dataset
import cv2

class SMGD(Dataset):
    def __init__(self, images_path, x, y, transform=None):
        self.images_path = images_path
        self.x = x
        self.y = y
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.images_path + self.x.iloc[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        label = self.y.iloc[index]

        if self.transform:
            img = self.transform(image= img)["image"]

        return img, label

    def __len__(self):
        return len(self.x)