<h1 align="center"> S²R: Teaching LLMs to Self-verify and Self-correct via Reinforcement Learning </a></h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

<h5 align="center">

![GitHub stars](https://img.shields.io/github/stars/NineAbyss/S2R.svg) ![](https://img.shields.io/badge/license-MIT-blue)

</h5>

This is the official implementation of the following paper:

> **S²R: Teaching LLMs to Self-verify and Self-correct via Reinforcement Learning** [[Paper](https://www.arxiv.org/abs/2502.12853)]

<p align="center"><img width="90%" src="figs/main.png" /></p>
<p align="center"><em>The overview of S²R.</em></p>

## 1. Environment Setup 🔧

```
pip install -r requirements.txt
```
**Note:** Different models require specific transformers versions:
- Qwen2-7B-Instruct & Qwen2.5-Math-7B SFT: transformers 4.39.3
- Llama-3.1-8B-Instruct SFT: transformers 4.44.3
- RL: transformers 4.46.3

## 2. Data Collection 📚
Download the original [[MATH500](https://github.com/openai/prm800k/tree/main/prm800k/math_splits)] and [[GSM8K](https://github.com/openai/grade-school-math)] dataset.
#### Serve the model with vLLM
```
python -m vllm.entrypoints.api_server  --model Qwen/Qwen2.5-Math-7B --port 8081  --tensor-parallel-size 4
```
Configure the model and data path in `./scripts/collect_data.sh`
```
sh ./scripts/collect_data.sh
```
We also provide the SFT and RL data in `data/train_data/sft_qwen2.5_math_7B.json` and `data/train_data/rl_data_qwen2.5.jsonl` for reproducing the result of Qwen2.5-Math-7B in our paper.

We also support using our datasets via [Hugging Face](https://huggingface.co/datasets/S2R-data/S2R-dataset) now! 


## 3. SFT Training 🔥
cd ./code/scripts
Configure your data and model path, then run:

```
sh ./code/scripts/train_sft.sh
```

## 4. RL Training 🚀 

Configure your data and model path, then run:
```
sh ./code/scripts/train_rl.sh
```

Use the following config for outcome-level training:
```
--use_instance_level True
--kl_coef 0.01   # for Qwen2.5-Math_7B
--rl_data_path ./data/train_data/rl_data_qwen2.5.jsonl  # for Qwen2.5-Math_7B
```

Use the following config for process-level training:
```
--use_instance_level False
--kl_coef 0.05
--rl_data_path ./data/train_data/rl_data_qwen2.5.jsonl  # for Qwen2.5-Math_7B
```

## 5. Evaluation

Please refer to 
`./tools/qwen_eval/eval/README.md`

## 🌟 Cite

```tex
@article{ma2025s,
  title={S $\^{} 2$ R: Teaching LLMs to Self-verify and Self-correct via Reinforcement Learning},
  author={Ma, Ruotian and Wang, Peisong and Liu, Cheng and Liu, Xingyan and Chen, Jiaqi and Zhang, Bang and Zhou, Xin and Du, Nan and Li, Jia},
  journal={arXiv preprint arXiv:2502.12853},
  year={2025}
}
```


## Acknowledgement
The code refer to [huggingface/trl](https://github.com/huggingface/trl).
The evaluation toolkit is built on [QwenLM/Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math).
Thanks for their wonderful work.
