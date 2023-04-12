from CAMELOT import CamelotModel
from datasets_dataloader_pytorch import *

epochs = 100

input_shape = (1, )

model = CamelotModel(input_shape)
model.initialize(train_data, val_data)

encoder_optim = torch.optim.Adam(model.Identifier.parameters(), lr=0.001)
identifier_optim = torch.optim.Adam(model.Identifier.parameters(), lr=0.001)
predictor_optim = torch.optim.Adam(model.Identifier.parameters(), lr=0.001)
cluster_optim = torch.optim.Adam(model.Identifier.parameters(), lr=0.001)

for i in range(epochs):
    epoch_loss = 0
    for step_, (x_batch, clus_batch) in enumerate(temp):
        encoder_optim.zero_grad()

        clus_pred = self.Identifier(self.Encoder(x_batch))
        loss = clus_pred_loss(clus_batch, clus_pred, self.weights)

        loss.backward()
        initialize_optim.step()

        epoch_loss += loss.item()

    with torch.no_grad():
        clus_pred_val = self.Identifier(self.Encoder(x_val))
        loss_val = clus_pred_loss(
            clus_val, clus_pred_val, self.weights)

    iden_loss[i] = loss_val.item()
    if torch.le(iden_loss[-50:], loss_val.item() + 0.001).any():
        break
