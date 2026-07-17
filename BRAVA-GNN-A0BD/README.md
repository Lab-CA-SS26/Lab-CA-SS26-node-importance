# XXXX-13
<a href="XXXX-11"><img src="XXXX-12"></a>

Official code implementation for BRAVA-GNN, a parameter-efficient GNN for approximating betweenness centrality with a 9.7% average improvement in Kendall–Tau correlation over the strongest baseline, while using 56× fewer parameters than the lightest competing GNN baseline and achieving up to a 44× speedup in inference time over the state-of-the-art.

---

**Setup & Requirements**

To install dependencies and activate the virtual environment:
```bash
source setup.sh
```

**Dataset Details**

All datasets are sourced from [SNAP](https://snap.stanford.edu/data/) or [Network Repository](https://networkrepository.com/).

To download the 14 paper test graphs:
```bash
python datasets/download.py
```
Add `--calibration` to also fetch the parameter-tuning graphs used in the appendix.

To generate the synthetic training graphs and build the training splits used by the our model:
```bash
python datasets/generate_graph.py --datasets SF_10_Dir SF_10_Sym HY_10_Dir
python datasets/create_dataset.py --datasets SF_10_Dir SF_10_Sym HY_10_Dir
```

**Running the model code**

```bash
# Final model hyperparameters
python betweenness.py --init_type degree_mix_mass_6 --num_layers 2 --nhid 12 --dropout 0.3 --train_type SF_10_Dir+SF_10_Sym+HY_10_Dir --seed 1 --run_all_tests
```
*   `--run_all_tests` will test on the entire test suite, instead of a small subset.
*   `--train_type` combines training sets with `+`. Each SF/HY name carries an explicit regime: `_Dir` (directed) or `_Sym` (symmetric DiGraph). The headline mix is 10 directed Scale-Free + 10 symmetric Scale-Free + 10 directed Hyperbolic graphs.


### ABCDE / DrBC Baseline

To import/convert datasets for [ABCDE](https://github.com/MartinXPN/abcde/tree/main) format:
```bash
wget https://github.com/MartinXPN/abcde/releases/download/v1.0.0/real.zip
unzip real.zip && rm real.zip
python datasets/import_abcde_datasets.py
```

To run the ABCDE/DrBC experiment (checkpoints ship in `baselines/abcde/`):
```bash
python baselines/eval_abcde.py --model abcde --seed 1
python baselines/eval_abcde.py --model drbc  --seed 1
```

### GNN-Bet Baseline

```bash
python baselines/GNN-Bet/betweenness.py --g SF
```

### Sampling-based Baselines (Bavarian, SILVAN, KADABRA)

Compile the C++ binaries (see each baseline's README under `baselines/`), then:
```bash
python baselines/eval_bavarian.py
python baselines/eval_silvan.py
python baselines/eval_kadabra.py
```

**Results**

Results are automatically appended to CSV files in `results/betweenness/`:
*   `all_results.csv`: Kendall Tau correlation scores.
*   `all_results_topk.csv`: Detailed Top-K accuracy metrics.
*   `all_results_wallclock.csv`: Inference time measurements.

To generate the LaTeX tables used in the paper:

```bash
python scripts/parse_results.py
```

## Citation

If you find BRAVA-GNN useful in your research, please cite our paper:

```bibtex
@misc{XXXX-10,
      title={XXXX-13},
      author={XXXX-9},
      year={2026},
      eprint={XXXX-12},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={XXXX-11}
}
```
