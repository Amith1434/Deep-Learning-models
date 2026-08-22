import torch
import os
from torch.utils.data import Dataset

class IMDBDataset(Dataset):
    def __init__(self,root_dir):
        self.reviews = []
        self.labels = []
        pos_dir = os.path.join(root_dir,"pos")

        for filename in os.listdir(pos_dir):
            filepath = os.path.join(pos_dir,filename)
            with open(filepath, "r", encoding="utf-8") as file:
                review = file.read()

            self.reviews.append(review)
            self.labels.append(1)

        neg_dir = os.path.join(root_dir, "neg")

        for filename in os.listdir(neg_dir):
            filepath = os.path.join(neg_dir, filename)

            with open(filepath, "r", encoding="utf-8") as file:
                review = file.read()

            self.reviews.append(review)
            self.labels.append(0)

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, index):
        return self.reviews[index], self.labels[index]

if __name__ == "__main__":

    dataset = IMDBDataset("aclImdb/train")

    print("Dataset size:", len(dataset))

    review, label = dataset[0]

    print("Label:", label)
    print("Review:", review[:500])

    