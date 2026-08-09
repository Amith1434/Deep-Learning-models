import kagglehub
import os

from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Download latest version
path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")

# print("Path to dataset files:", path)
# print("Dataset path:", path)
# print("Contents : ", os.listdir(path))

train_path = os.path.join(path, "Training")
test_path = os.path.join(path, "Testing")

#Convert Image to Tensors

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(
    root = train_path,
    transform = transform
)

test_dataset = datasets.ImageFolder(
    root = test_path,
    transform = transform
 )
# image, label = train_dataset[10]
# print("Image shape:", image.shape)
# print("Label:",label)
# print("Class : ", train_dataset.classes[label])
# print(train_dataset.class_to_idx)
# plt.imshow(image.permute(1,2,0))
# plt.axis("off")
# plt.show()