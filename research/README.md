# 研究工作区说明

## 代码来源与基线

- 上游仓库：<https://github.com/Dreamzz5/ConFormer>
- 当前 fork 的 `main` 保存已验证的东京复现版本。
- fork 时的上游基线提交：`743603fc350771d78cbf8a219146ef9c717775a4`。
- XTraffic 的 M0 和 M1 从同一复现版本出发，分别放在 `research/m0` 和 `research/m1`。

## 目录用途

本仓库保留作者原有结构，并新增本目录作为研究记录入口：

- `model/`：作者模型、配置、训练与评价代码；实验修改在研究分支进行。
- `lib/`：数据读取、指标和辅助工具。
- `data/`：运行时挂载或准备数据的位置。数据集本体不提交到 GitHub。
- `logs/`、`saved_models/`：训练日志与模型权重；通常只保存在 Kaggle 输出或其他存储中，不提交到代码仓库。
- `research/`：记录复现协议、数据适配说明、实验配置和结果索引。尚未运行的实验不填写为已验证结论。

## 建议的版本使用方式

1. 查看或复跑东京基线时使用 `main`。
2. 核对 XTraffic 时切到相应的 M0 或 M1 分支。
3. 每完成一项可解释的小改动就提交一次，并在实验记录中写明提交号、数据版本、配置和运行结果。
4. 在含数据集的远程环境运行实验，代码改动保存在相应的研究分支。大数据文件、密钥、日志和权重不要推送到公开代码仓库。

## Alameda 数据检查与正式训练

在项目根目录运行，数据目录指向可信的 IGSTGNN Alameda 发布包。

    python -m model.prepare_xtraffic --data-dir data/xtraffic/Alameda --output-dir logs/alameda-prepare

默认读取当前分支的 XTraffic 配置，生成划分清单、训练尺度和数据核验报告。

在项目根目录启动本分支的正式训练：

    python model/train_xtraffic.py --data-dir data/xtraffic/Alameda --device cuda:0 --expect-model m0 --seed 2024

每次训练用本分支的事件编码及事件表，独立保存日志、初始主干、最佳权重和测试预测。相同环境和种子下，事件表初始化与后续主干初始化使用隔离的 RNG 状态。复跑归档实验的随机起点时，两组可选用同一份训练前的初始主干文件，通过 --initial-backbone 指定；不要用最佳权重 best.pt。配对结果在两组训练后比较，核对报告中的主干指纹、训练设置及逐轮样本顺序。

配对核验及可选的逐样本误差细分由同一工具完成：

    python model/compare_xtraffic_runs.py --m0-summary logs/<m0-run>/run_summary.json --m1-summary logs/<m1-run>/run_summary.json

当两份报告引用的 `test_predictions.npz` 可读取时，工具还会核对样本、真值、有效位置、事件节点掩码和类别，并汇总全区域、事件节点、其余节点、类别及预测时距的 MAE/RMSE 与覆盖数。报告里的预测文件路径不可用时，原有报告级配对核验仍可单独使用；也可显式传入 `--m0-predictions` 和 `--m1-predictions`。
