import argparse
import torch
from unet import UNet
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import copy


def plot_img_and_mask(img, mask):
    classes = mask.max()
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    ax[1].set_title('Mask')
    ax[1].imshow(mask == 0)
    plt.savefig("a.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='model.pth',
                        help='Specify the file in which the model is stored')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Loading model {args.model}')
    print(f'Using device {device}')

    model = UNet(in_channels=3, out_channels=1).to(device)
    # x=torch.rand((1,3,572,572)).to(device)
    # y=model(x)
    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict)

    print('Model loaded')

    img=Image.open("infer.jpg")
    img=torchvision.transforms.Resize((572, 572),interpolation=torchvision.transforms.InterpolationMode.BILINEAR)(img)
    img0=copy.deepcopy(img)
    img=torchvision.transforms.ToTensor()(img)
    # img=torch.tensor([img])
    img.unsqueeze_(0)
    img=img.to(device)
    mask=model(img)
    mask=mask.to("cpu")
    mask=mask.squeeze(0)
    mask = mask.squeeze(0)
    mask=torch.nn.Sigmoid()(mask)
    theshold=0.1
    mask=(mask>theshold).numpy()
    print(mask)
    plot_img_and_mask(img0, mask)