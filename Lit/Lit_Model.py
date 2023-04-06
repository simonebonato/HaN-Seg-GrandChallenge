import lightning.pytorch as Lit


class LitModel(Lit.LightningModule):
    def __init__(self):
        pass

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass
