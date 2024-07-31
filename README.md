# MMRA
This is the repo for the paper: '[MMRA: A Benchmark for Multi-granularity Multi-image Relational Association](https://arxiv.org/pdf/2407.17379)'.

Our benchmark dataset is released: [Huggingface Dataset: m-a-p/MMRA](https://huggingface.co/datasets/m-a-p/MMRA), [Google Drive](https://drive.google.com/file/d/1XhyCfCM6McC_umSEQJ4NvCZGMUnMdzj2/view?usp=sharing), and [Baidu Netdisk](https://pan.baidu.com/s/1deQOSzpX_-Y6-IjlSiN1OA?pwd=zb3s).

The MMRA.zip in Google Drive and Baidu Netdisk contains a metadata.json file, which includes all the sample information. We can input the relevant questions, options, and image pairs to LVMLs through it.

---

# Introduction

We define a multi-image relation association task, and meticulously curate **MMRA** benchmark, a **M**ulti-granularity **M**ulti-image **R**elational **A**ssociation benchmark, consisted of **1,024** samples.
In order to systematically and comprehensively evaluate mainstream LVLMs, we establish an associational relation system among images that contain **11 subtasks** (e.g, UsageSimilarity, SubEvent, etc.) at two granularity levels (i.e., "**image**" and "**entity**") according to the relations in ConceptNet.
Our experiments reveal that on the MMRA benchmark, current multi-image LVLMs exhibit distinct advantages and disadvantages across various subtasks. Notably, fine-grained, entity-level multi-image perception tasks pose a greater challenge for LVLMs compared to image-level tasks. Tasks that involve spatial perception are especially difficult for LVLMs to handle.
Additionally, our findings indicate that while LVLMs demonstrate a strong capability to perceive image details, enhancing their ability to associate information across multiple images hinges on improving the reasoning capabilities of their language model component.
Moreover, we explored the ability of LVLMs to perceive image sequences within the context of our multi-image association task. Our experiments indicate that the majority of current LVLMs do not adequately model image sequences during the pre-training process.
<div align="center">
<img src=./imgs/framework.png width=80% />
</div>

<div align="center">
<img src=./imgs/main_result.png width=80% />
</div>

---
# Using Datasets

You can load our datasets by following codes:

```python
MMRA_data = datasets.load_dataset('m-a-p/MMRA')['train']
print(MMRA_data[0])
```
---
# Evaluation LVMLs on Our MMRA benchmark

We provide a demo to rerun the result of the LVLMs (i.e., Idefics2, Mantis, Phi3, Qwen-VL-Chat, GPT4o and GPT4v).
You can run the following codes for the open-source multi-image LVMLs:
```
python ./evaluate/evaluate_demo.py --model_name Idefics2 --model_id HuggingFaceM4/idefics2-8b
python ./evaluate/evaluate_demo.py --model_name Qwen-VL-Chat --model_id Qwen/Qwen-VL-Chat
python ./evaluate/evaluate_demo.py --model_name Mantis_Idefics2 --model_id TIGER-Lab/Mantis-8B-Idefics2
python ./evaluate/evaluate_demo.py --model_name Phi3v --model_id microsoft/Phi-3-vision-128k-instruct
```
If you want to use the OpenAI's API, please change the function "gpt_4_api" for a different model (e.g., GPT4o and GPT4v) and update your api and base_url in the "--API_KEY" and "--BASE_URL".

If you want to evaluate your model, you need to change the function "gpt_message" and "process_sample" to cater to your model, and it is also sample.

---
# Notice

Our original results are conducted by Lmms-eval framework. To make it easier for others to use our code and benchmark to evaluate the performance of their models, we replicate the result of Lmms-eval and provide the demo code. 
The results of our reproduced demo code are generally consistent with those of Lmms-eval, but there are some slight differences.

---
# Citation

BibTeX:
```
@inproceedings{Wu2024MMRAAB,
  title={MMRA: A Benchmark for Multi-granularity Multi-image Relational Association},
  author={Siwei Wu and Kang Zhu and Yu Bai and Yiming Liang and Yizhi Li and Haoning Wu and Jiaheng Liu and Ruibo Liu and Xingwei Qu and Xuxin Cheng and Ge Zhang and Wenhao Huang and Chenghua Lin},
  year={2024},
  url={https://api.semanticscholar.org/CorpusID:271404179}
}
```
