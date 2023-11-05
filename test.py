#%%
import torch
from palm import PaLM
from trainer import PaLMTrainer
from processor import PaLMProcessor
# %%
tokenizer = PaLMProcessor("./tokenizers/dictionary.pkl")
# %%
tokenizer.dictionary
#%%

# %%
trainer = PaLMTrainer(
    processor=tokenizer
)
# %%

# %%
a = torch.tensor([[1,2,3,4,5, 0, 0]])
#%%
trainer.train_step(a[:, :-1], a[:, 1:])
# %%
b = trainer.model(a)
# %%
b
# %%
