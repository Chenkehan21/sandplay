from PIL import Image
import torchvision.transforms as T

img_path = "/raid/ckh/sandplay_homework/resource/homework_sand_label_datasets/20201109101336_142/BireView.png"
img = Image.open(img_path).convert('RGB')
img = T.ToTensor()(img)
print(img.shape) # torch.size([3, 540, 960])