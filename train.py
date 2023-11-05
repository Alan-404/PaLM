import torch
import numpy as np
from trainer import PaLMTrainer, PaLMDataset
from processor import PaLMProcessor

def train(manifest_path: str,
          checkpoint: str,
          tokenizer_path: str,
          n: int,
          d_model: int,
          heads: int,
          eps: float,
          dropout_rate: float,
          bias: bool,
          lr: float,
          device: str,
          epochs: int,
          batch_size: int,
          ignore_index: int = -100):
    
    processor = PaLMProcessor(
        tokenizer_path
    )

    dataset = PaLMDataset(manifest_path, processor=processor)

    trainer = PaLMTrainer(
        processor=processor,
        n=n,
        d_model=d_model,
        heads=heads,
        eps=eps,
        dropout_rate=dropout_rate,
        bias=bias,
        lr=lr,
        device=device,
        checkpoint=checkpoint,
        ignore_index=ignore_index
    )

    trainer.fit(
        dataset=dataset,
        epochs=epochs,
        batch_size=batch_size
    )

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("--manifest_path", type=str)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--checkpoint", type=str)

    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--eps", type=float, default=0.02)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--bias", type=bool, default=False)

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--device", type=str ,default='cpu')
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--ignore_index", type=int, default=-100)

    args = parser.parse_args()


    train(
        manifest_path=args.manifest_path,
        checkpoint=args.checkpoint,
        tokenizer_path=args.tokenizer_path,
        n=args.n,
        d_model=args.d_model,
        heads=args.heads,
        eps=args.eps,
        dropout_rate=args.dropout_rate,
        bias=args.bias,
        lr=args.lr,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        ignore_index=args.ignore_index
    )
