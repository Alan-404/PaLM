import os
import torch
import torch.nn as nn
import torch.optim as optim
import torcheval.metrics as metrics
from palm import PaLM
from processor import PaLMProcessor
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm

class PaLMTrainer:
    def __init__(self,
                 processor: PaLMProcessor,
                 n: int = 12,
                 d_model: int = 768,
                 heads: int = 12,
                 eps: float = 0.02,
                 dropout_rate: float = 0.1,
                 bias: bool = False,
                 lr: float = 3e-5,
                 device: str= 'cpu',
                 checkpoint: str = None,
                 ignore_index: int = -100) -> None:
        
        self.model = PaLM(
            token_size=len(processor.dictionary),
            n=n,
            d_model=d_model,
            heads=heads,
            eps=eps,
            dropout_rate=dropout_rate,
            bias=bias
        )

        self.processor = processor

        self.optimizer = optim.AdamW(params=self.model.parameters(), lr=lr)
        self.loss_function = PaLMLoss(ignore_index)
        self.metric = PaLMMetric(ignore_index)

        self.epoch = 0
        self.loss = 0.0

        self.losses = []

        self.val_losses = []
        self.val_scores = []

        self.ignore_index = ignore_index

        self.device = device
        self.model.to(self.device)

        self.checkpoint = checkpoint

        if self.checkpoint is not None:
            self.load_model(self.checkpoint)

    def __load_model(self, path: str):
        checkpoint = torch.load(path)

        self.model.load_state_dict(checkpoint[ModelInfo.MODEL_STATE_DICT])
        self.optimizer.load_state_dict(checkpoint[ModelInfo.OPTIMIZER_STATE_DICT])
        self.epoch = checkpoint[ModelInfo.EPOCH]
        self.losses = checkpoint[ModelInfo.LOSS]
        self.val_losses = checkpoint[ModelInfo.VAL_LOSS]
        self.val_scores = checkpoint[ModelInfo.VAL_SCORE]

    def load_model(self, path: str):
        if os.path.exists(path):
            self.__load_model(path)

    def __save_model(self, path: str):
        torch.save({
            ModelInfo.MODEL_STATE_DICT: self.model.state_dict(),
            ModelInfo.OPTIMIZER_STATE_DICT: self.optimizer.state_dict(),
            ModelInfo.EPOCH: self.epoch,
            ModelInfo.LOSS: self.losses,
            ModelInfo.VAL_LOSS: self.val_losses,
            ModelInfo.VAL_SCORE: self.val_scores
        }, path)

    def save_model(self, path: str):
        try:
            self.__save_model(path)
        except Exception as e:
            print(str(e))
            self.__save_model("./palm.pt")

    def build_dataloader(self, dataset: Dataset, batch_size: int):
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: get_batch_with_padding(batch, self.processor))

    def train_step(self, inputs: torch.Tensor, labels: torch.Tensor):
        self.optimizer.zero_grad()

        outputs, mask = self.model(inputs)

        labels = labels.masked_fill((~mask), self.ignore_index)
        outputs = outputs.masked_fill((~mask).unsqueeze(-1), self.ignore_index)

        loss = self.loss_function(outputs, labels)
        loss.backward()
        self.optimizer.step()

        self.loss += loss.item()

    def fit(self, dataset: Dataset, epochs: int, batch_size: int):
        self.model.train()

        dataloader = self.build_dataloader(dataset, batch_size)
        num_batches = len(dataloader)

        for _ in range(epochs):
            for index, data in enumerate(tqdm(dataloader), 0):
                inputs = data[:-1].to(self.device)
                labels = data[1:].to(self.device)

                self.train_step(inputs, labels)

            loss = self.loss / num_batches
            print(f"Epoch {self.epoch + 1} Train Loss: {(loss):.3f}")

            self.losses.append(loss)

            self.loss = 0.0

            self.epoch += 1

        if self.checkpoint is not None:
            self.save_model(self.checkpoint)
        else:
            self.save_model("./palm.pt")

class PaLMDataset(Dataset):
    def __init__(self, manifest_path: str, processor: PaLMProcessor) -> None:
        super().__init__()
        self.prompts = pd.read_csv(manifest_path, sep="\t")
        self.processor = processor
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, index: int):
        prompt = self.prompts.iloc[index]
        text_input = prompt['input']
        text_output = prompt['output']
        digits = self.processor.text2sequence(text_input, text_output)
        return digits

class PaLMLoss(nn.Module):
    def __init__(self, ignore_index: int = -100) -> None:
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor):
        batch_size = labels.size(0)

        loss = 0.0

        for batch in range(batch_size):
            loss += self.criterion(outputs[batch], labels[batch])

        return loss / batch_size
    
class PaLMMetric(nn.Module):
    def __init__(self, ignore_index: int = -100) -> None:
        super().__init__()
        self.metric = metrics.Perplexity(ignore_index=ignore_index)

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor):
        return self.metric.update(outputs, labels).compute().item()

def get_batch_with_padding(batch, processor: PaLMProcessor):
    max_length = np.max([len(item) for item in batch])
    data = []
    for item in batch:
        data.append(np.pad(item, (0, max_length-len(item)), constant_values=processor.padding_token))
    return torch.tensor(np.array(data))

class ModelInfo:
    MODEL_STATE_DICT = 'model_state_dict'
    OPTIMIZER_STATE_DICT = 'optimizer_state_dict'
    EPOCH = 'epoch'
    LOSS = 'loss'
    VAL_LOSS = 'val_loss'
    VAL_SCORE = 'val_score'