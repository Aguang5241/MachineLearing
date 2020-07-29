from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

def show_whole(panda):
    plt.imshow(panda)
    plt.show()

def show_part1(panda_tensor):
    fig = plt.figure()
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)
    ax3 = plt.subplot(133)
    ax1.imshow(panda_tensor[:, :, 0].numpy())
    ax2.imshow(panda_tensor[:, :, 1].numpy())
    ax3.imshow(panda_tensor[:, :, 2].numpy())
    plt.show()

def show_part2(panda_tensor):
    plt.imshow(panda_tensor[25:175, 60:130, 0].numpy())
    plt.show()

def main():
    panda = np.array(Image.open(r'D:\Study\Coding\MachineLearing\DeepLearningWithPyTorch\res\panda.png').resize((224, 224)))
    panda_tensor = torch.from_numpy(panda)
    # print(panda_tensor)
    # print(panda_tensor.size()) # torch.Size([224, 224, 4])

    # show_whole(panda)

    # show_part1(panda_tensor)

    show_part2(panda_tensor)

if __name__ == '__main__':
    main()
