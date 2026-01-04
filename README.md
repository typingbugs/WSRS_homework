# 推荐系统课程大作业

作者: 柯劲帆  学号: 25120323

本项目为北京交通大学计算机学院研究生课程《Web搜索与推荐系统导论》的大作业实现，包含数据预处理、模型实现（若干基于SASRec的变体）、训练脚本与推理流程。仓库整合了自定义的模型实现（位于 `transformers/src/transformers/models/` 下）、数据处理管道（`data_preprocess/`）、训练/推理脚本（`scripts/`）以及示例配置（`model_configs/` 与 `train_configs/`）。

**目录结构（关键信息）**
- `data/`：训练、推理所需的数据与嵌入结果
- `data_preprocess/`：数据抽取、tokenizer 与预训练嵌入构建脚本
- `model_configs/`：模型配置文件夹（示例模型放在 `model_1/`、`model_2/`）
- `train_configs/`：训练参数配置（SFT / DPO 等实验设置）
- `scripts/`：运行训练与推理的一键脚本（`sft.sh`、`dpo.sh`、`inference.sh`）
- `transformers/`：包含自定义模型实现的本地 transformers 修改版（用于实验）

**注意**：本 README 假定你在 Linux 环境（或具有相应 CUDA 支持的服务器）中使用 conda/venv 管理 Python 环境。

---

**1. 环境与安装**

推荐先创建并激活 conda 环境（可选）：

```bash
conda create -n recsys python=3.11 -y
conda activate recsys
```

安装依赖（按顺序执行）：

- 安装 PyTorch（要求版本）：`pytorch:2.7.1-cuda11.8-cudnn9`（请参考官方安装命令以匹配 CUDA 驱动与显卡）

- 可选：安装 flash-attn（若使用且需加速）：

```bash
pip install ninja
MAX_JOBS=4 pip install flash-attn==2.3.5
```

- 安装 Python 依赖：

```bash
pip install -r requirements.txt
```

- 安装 Faiss GPU（采用 conda）：

```bash
conda install -c pytorch -c nvidia faiss-gpu -y
```

- 安装项目中修改/使用的 `transformers`（本地源码安装）：

```bash
cd transformers
pip install '.[torch]'
cd -
```

---

**2. 数据准备**

2.1 原始数据

- 将作业提供的 `test1.csv` / `test2.csv` 放到 `data/raw_data/` 下。

2.2 从 CSV 提取训练数据

- 运行数据抽取脚本：

```bash
python data_preprocess/load_csv.py
```

- 将生成的数据拷贝到训练数据目录：

```bash
cp data_preprocess/outputs/*-dataset.jsonl data/train_data/
```

2.3 构建 tokenizer

- 运行 tokenizer 构建脚本：

```bash
python data_preprocess/build_tokenizer.py
```

- 将生成的 tokenizer 文件复制到模型配置目录（示例）：

```bash
cp -r data_preprocess/outputs/hf_item_tokenizer/* model_configs/model_1/
```

2.4 构建 DPO 数据（可选）

- 若需要 DPO 训练数据：

```bash
python data_preprocess/build_dpo_dataset.py
cp data_preprocess/outputs/dpo_*_dataset.jsonl data/train_data/
```

2.5 预训练嵌入（可选，用于冷启动或增强 item 特征）

- 仅使用商品标题提取嵌入：

```bash
python data_preprocess/build_pretrained_embeddings/item_to_embedding.py
```

对于 Faiss 索引与近似检索，确保 Faiss 的 CUDA 版本与系统 CUDA 驱动兼容。

- 使用外部数据（Wikipedia / Steam 等）增强信息的流程（可选）：

流程如图：

![data_aug](https://github.com/typingbugs/WSRS_homework/blob/main/figs/data_augmentation.png)

1. 提取 Steam 数据（若有）：

```bash
python data_preprocess/build_pretrained_embeddings/retrieve_external_data/steam.py
```

2. 使用 Wikipedia 搜索商品信息（需提供 email）：

```bash
EMAIL_FOR_WIKI_BOT="your_email@example.com" python data_preprocess/build_pretrained_embeddings/retrieve_external_data/wikipedia.py
```

3. 使用 LLM 抽取 Wikipedia 关键信息（需配置 OPENAI）：

```bash
OPENAI_API_KEY="your_api_key" BASE_URL="your_api_base_url" python data_preprocess/build_pretrained_embeddings/extract_wiki.py
```

4. 合并 Steam 与 Wiki 信息：

```bash
python data_preprocess/build_pretrained_embeddings/merge_info.py
```

5. 若要在 `item_to_embedding.py` 中使用“add-info”流程，请取消注释相应段落然后运行：

```bash
# 修改 data_preprocess/build_pretrained_embeddings/item_to_embedding.py 中的 add-info 选项为启用
python data_preprocess/build_pretrained_embeddings/item_to_embedding.py
```

- 预训练得到的商品向量将保存到 `data/item_embeddings/` 下（`.npy` 文件）。

---

**3. 训练与推理**

- 准备模型配置：将 `model_configs/model_1/`（或其它 model_* 文件夹）下的配置文件准备好，包含 tokenizer、`config.json`、`item2id.json` 与 `id2item.json` 等。

- 准备训练配置：在 `train_configs/` 下选择或修改相应的 YAML 配置（例如 `train_configs/sft/settings_1.yaml` 或 `train_configs/dpo/settings_1.yaml`）。

- 推荐做的实验集：
  - `sasrec_rope`：基准模型（效果最好的一版变体）
  - 消融与对比变体：`sasrec`、`sasrec_gate`、`sasrec_moe`

- 运行 SFT 训练（示例脚本）：

```bash
bash scripts/sft.sh
```

  输出（默认）：`data/outputs/sft/`

  若显存受限，请在训练配置中调小 `batch_size` 或 `seq_len`。
  
  如果使用 `flash-attn` 时遇到问题，先用纯 PyTorch 后端验证训练脚本是否能顺利运行，再逐步启用加速库。

  训练分为两阶段，如图：

  ![train](https://github.com/typingbugs/WSRS_homework/blob/main/figs/train_stages.png)

- 运行 DPO 训练：

```bash
bash scripts/dpo.sh
```

  输出（默认）：`data/outputs/dpo/`

- 使用训练好的 checkpoint 进行推理：

```bash
bash scripts/inference.sh
```

  推理输出默认存放：`data/infer/`

---

**4. 模型结构与实现位置**

项目在本地 `transformers` 源码中加入了若干自定义模型实现，用于对比不同设计：

- `transformers/src/transformers/models/sasrec`
  - 说明：参考并实现了效果最好的 SASRec 参考实现。
  - 主要修改：将 `layer_norm` 替换为 `RMS_norm`，其余部分参考SASRec原论文与实现（可学习位置编码）。

- `transformers/src/transformers/models/sasrec_gate`
  - 说明：在自定义 SASRec 的 MLP 中使用了 Qwen3 的 gate MLP 实现以验证 gate 结构效果。

- `transformers/src/transformers/models/sasrec_rope`
  - 说明：将SASRec原本的可学习位置编码替换为 RoPE（rotary position embeddings），用于研究位置编码对性能的影响。（实验证明在本任务中更稳定/更优）。

- `transformers/src/transformers/models/sasrec_moe`
  - 说明：将 SASRec 改造为混合专家（MoE）结构，用于测试模型容量与专家路由对推荐性能的影响。

每个模型目录下包含模型定义、配置与必要的帮助函数，参照 `transformers` 的模型组织方式，便于加载与训练。

---
