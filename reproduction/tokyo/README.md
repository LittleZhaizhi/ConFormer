# Tokyo reproduction tools

Run these commands from the repository root. The standard model and its only
training entry point live under `model/`.

## Check attention axes and run a real batch

```bash
python reproduction/tokyo/verify.py axes --device cpu
python reproduction/tokyo/verify.py smoke --device cuda:0 --data-dir data/TKY --output-dir logs/tky-smoke-01
```

The first check uses synthetic inputs to verify the spatial and historical axes
and direct graph conditioning of K/V. The second runs one training update with
the released Tokyo data, reloads the saved weights, and compares the predictions
on a validation batch.

## Evaluate the selected checkpoint

```bash
python reproduction/tokyo/evaluate.py --device cuda:0 --data-dir data/TKY --checkpoint logs/tky-train-01/best.pt --output-dir logs/tky-test-01
```

Evaluation reports MAE, RMSE and MAPE for the full test set and the three
local scenarios. It also compares 10, 30 and 60 minute forecasts against
paper Table 2. `test_predictions.npz` includes window indices, original-scale
truth and predictions, and shared masks.

To check the refactor against the accepted `tky-dual-full-02` run, set
`--checkpoint` to that run's `best.pt` and add
`--reference-dir logs/tky-dual-full-02`. Use a new output directory and the same
data and runtime as the reference. A pass confirms the same checkpoint hash,
evaluation positions, predictions and metrics within numeric tolerance. See
`reference_check.json` for the largest absolute prediction difference.

## Optional Linux launch and monitoring

```bash
python reproduction/tokyo/launch_tky_detached.py --job-dir logs/tky-train-01-launch -- --device cuda:0 --data-dir data/TKY --output-dir logs/tky-train-01
python reproduction/tokyo/watch_tky_detached.py --job-dir logs/tky-train-01-launch
```

The first command launches training in a separate Linux session. The second reads
the saved task state and logs and can be started again after reconnecting. Use
separate, new directories for the launcher and trainer. A completed run writes
`train_report.json`; evaluating the checkpoint is a separate step.

## Historical source

The first dual-axis training and evaluation entry point is saved in the local
`archive/tky-dual-full-02` source snapshot. The earlier single-path reproduction
is recorded at commit `f9112ca`. The data, checkpoints and individual prediction
files are server artifacts.
