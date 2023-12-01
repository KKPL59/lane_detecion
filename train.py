import torch as T
from torch import nn
from netowrk import UNet
from datagen import DataGenerator
from torch.distributions import Categorical
import cv2
import numpy as np
import shutil
from torch.utils.tensorboard import SummaryWriter
from torch import optim



width, height = 256, 128
data_gen = DataGenerator(8, width, height)

paths_list1, lanes_list1 = data_gen.get_lists(
    r"C:\Users\lukas\PycharmProjects\line_detection\datasets\archive (10)\TUSimple\train_set\label_data_0313.json")
paths_list2, lanes_list2 = data_gen.get_lists(
    r"C:\Users\lukas\PycharmProjects\line_detection\datasets\archive (10)\TUSimple\train_set\label_data_0531.json")
paths_list3, lanes_list3 = data_gen.get_lists(
    r"C:\Users\lukas\PycharmProjects\line_detection\datasets\archive (10)\TUSimple\train_set\label_data_0601.json")

paths_list1.extend(paths_list2)
paths_list1.extend(paths_list3)

lanes_list1.extend(lanes_list2)
lanes_list1.extend(lanes_list3)

epochs = 1000
loss_fn = nn.BCEWithLogitsLoss(pos_weight=T.tensor([23.], device="cuda")) # 25
# loss_fn = nn.BCEWithLogitsLoss()
network = UNet(1).to("cuda")
network.load_state_dict(T.load(r"C:\Users\lukas\PycharmProjects\line_detection\lstm_unet_test\model"))
optimizer = optim.Adam(network.parameters(), 1e-4, weight_decay=1e-5) # 1e-4
# optimizer = optim.SGD(network.parameters(), 0.01)

loss_min = 1000
for epoch in range(epochs):
    # if epoch == 0:
    data, target = data_gen.get_data(paths_list1, lanes_list1)
    data = data.unsqueeze(2)
    data = data / 255.
    target = target / 255.
    target = target.unsqueeze(1)
    data = data.to("cuda")
    target = target.to("cuda")

    optimizer.zero_grad()
    pred = network(data).to("cuda")
    # print(pred.device)
    # print(target.device)
    loss = loss_fn(pred, target)
    # if epoch == 0:
    #     loss_min = loss.item()
    #
    # if loss.item() < loss_min:
    #     loss_min = loss.item()
    #     T.save(network.state_dict(), r"C:\Users\lukas\PycharmProjects\line_detection\lstm_unet_test\model")
    #     print("saving...")

    print(f"loss: {loss.item()}")
    loss.backward()
    optimizer.step()

    image = T.sigmoid(pred[0])
    image_o = data[0, 4] * 255
    cv2.imshow("image_o", image_o.detach().permute(1, 2, 0).cpu().numpy().astype("uint8"))
    print(image.max())
    image = (image > 0.8).float()
    image = image * 255
    image = image[0].detach().cpu().numpy().astype("uint8")
    cv2.imshow("image", image)
    cv2.waitKey(1)

