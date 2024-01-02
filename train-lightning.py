import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from pytorch_lightning.loggers import WandbLogger

from gpt import GPTLanguageModel


class TinyShakespeareDataSet(Dataset):
    def __init__(self, raw_text, block_size=256):
        super().__init__()
        self.raw_text = raw_text
        self.xs = torch.stack([raw_text[i:i + block_size] for i in range(len(raw_text) - block_size)])
        self.ys = torch.stack([raw_text[i + 1:i + block_size + 1] for i in range(len(raw_text) - block_size)])

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]

class TinyShakespeareDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size=32,train_test_split=0.95):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_test_split = train_test_split

    def prepare_data(self):
        # runs only once when main process runs
        # this tokenises the data in one step

        with open(self.data_dir+'/input.txt', 'r', encoding='utf-8') as f:
            text = f.read()
            chars = sorted(list(set(text)))

            stoi = {ch: i for i, ch in enumerate(chars)} # integer tokens for each character - character level encoding representation
            itos = {i: ch for i, ch in enumerate(chars)}

            def encode(s):
                return [stoi[c] for c in s] # encoder - a simple dict mapping

            # def decode(l):
            #     return ''.join([itos[i] for i in l]) # decoder - a simple dict mapping

            data = torch.tensor(encode(text), dtype = torch.long)

            torch.save(data, self.data_dir + '/tokenised.pt')

    def setup(self, stage):
        # runs on each GPU
        # this splits the data into train and test sets

        data = torch.load(self.data_dir + '/tokenised.pt')

        n = int(self.train_test_split * len(data))
        # self.train_data = data[:n]
        # self.val_data = data[n:]
        self.train_data = TinyShakespeareDataSet(data[:n])
        self.val_data = TinyShakespeareDataSet(data[n:])

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=10, persistent_workers=True)

    def val_dataloader(self):
        # lightning will lazily evaluate dataloaders as needed
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=10, persistent_workers=True)


class LightningGPT(L.LightningModule):
    def __init__(self, GPTLanguageModel):
        super().__init__()
        self.decoder = GPTLanguageModel

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, loss = self.decoder(x, y)
        self.log('train_loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, loss = self.decoder(x, y)
        self.log('val_loss', loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)



if __name__ == '__main__':
    # torch.set_float32_matmul_precision('high')

    data_module = TinyShakespeareDataModule('data')

    data_module.prepare_data()
    data_module.setup('train')

    model = GPTLanguageModel()

    lightning_model = LightningGPT(model)

    # logging to WandB
    wandb_logger = WandbLogger(log_model="all", project="lightning-gpt")

    # train model
    num_devices = 1
    gpus_per_device = 1

    trainer = L.Trainer(accelerator="cpu", devices=num_devices, num_nodes=gpus_per_device, logger=wandb_logger, max_epochs=5, strategy='ddp')
    trainer.fit(model=lightning_model, datamodule=data_module)