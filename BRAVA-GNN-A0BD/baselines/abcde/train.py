import argparse
import resource
import shutil
import tempfile
from pathlib import Path

import torch

# Raise RLIMIT_NOFILE soft → hard. With 4 DDP ranks × (cpu_count-1) DataLoader
# workers × file_system tensor sharing, the OAR-node default (1024) is blown
# during the first val iteration. Children inherit the new soft limit.
_soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (_hard, _hard))
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

from abcde.data import GraphDataModule
from abcde.models import ABCDE
from abcde.util import fix_random_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt_out", type=Path, default=Path("baselines/abcde/best.ckpt"),
                        help="Final checkpoint path. Best-by-val_kendal model is copied here at end of training.")
    parser.add_argument("--cache_dir", type=Path, default=Path("datasets/cache"),
                        help="Persistent cache for generated training graphs (shared across seeds).")
    parser.add_argument("--max_epochs", type=int, default=50)
    args = parser.parse_args()

    fix_random_seed(args.seed)
    torch.multiprocessing.set_sharing_strategy("file_system")
    args.ckpt_out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="abcde_train_") as work_dir:
        work_dir = Path(work_dir)
        ckpt_dir = work_dir / "checkpoints"

        model = ABCDE(nb_gcn_cycles=(4, 4, 6, 6, 8, 8),
                      conv_sizes=(48, 48, 32, 32, 24, 24),
                      drops=(0.3, 0.3, 0.2, 0.2, 0.1, 0.1),
                      lr_reduce_patience=2, dropout=0.1)
        data = GraphDataModule(min_nodes=4000, max_nodes=5000, nb_train_graphs=160, nb_valid_graphs=240,
                               batch_size=16, graph_type="powerlaw", repeats=8, regenerate_epoch_interval=10,
                               cache_dir=args.cache_dir)

        ckpt_cb = ModelCheckpoint(dirpath=ckpt_dir, filename="model-{epoch:02d}-{val_kendal:.2f}",
                                  monitor="val_kendal", save_top_k=1, mode="max", verbose=True)
        trainer = Trainer(
            logger=CSVLogger(save_dir=work_dir, name="logs"),
            gradient_clip_val=1,
            accelerator="gpu", devices=-1 if torch.cuda.is_available() else None,
            max_epochs=args.max_epochs,
            callbacks=[
                EarlyStopping(monitor="val_kendal", patience=5, verbose=True, mode="max"),
                ckpt_cb,
                LearningRateMonitor(logging_interval="epoch"),
            ],
        )
        trainer.fit(model, datamodule=data)
        print(trainer.callback_metrics)

        best = ckpt_cb.best_model_path
        if not best:
            raise RuntimeError("Training finished without a best checkpoint")
        shutil.copyfile(best, args.ckpt_out)
        print(f"Saved best checkpoint to {args.ckpt_out}")


if __name__ == "__main__":
    main()
