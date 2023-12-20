# lane_detecion
pytorch model that uses unet and lstm to predict traffic lanes masks



the model basicly takes the short video of road(5 frames) uses unest to resize(decode) every frame and than it passes the frames to 
lstm model that is able to extract features from sequence data. Than the output of lstm is encodet and the predicted mask are returned.

Trained on gpu using:
amd ryzen 5600 X, 
64 GB ram 
1x nvidia geforce gtx titan x
1x nvidia geforce gtx 1660 ti

using adam optimizer(1e-4 learning rate, weight_decay=1r-5) with 8 batch size
loss function: Binary cross entropy(used pos_weight=23 due to unbalanced amount of "true" and "false" pixels)

