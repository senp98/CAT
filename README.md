# CAT: Contrastive Adversarial Training for Evaluating the Robustness of Protective Perturbations in Latent Diffusion Models

This repository contains the official implementation of the paper: [CAT: Contrastive Adversarial Training for Evaluating the Robustness of Protective Perturbations in Latent Diffusion Models]().

![demo](figures/vggface2_dreambooth_demo.pdf)

## 1. Dependencies
To install the dependencies, run the following:
```bash
mamba create -n cat python=3.12.4
mamba activate cat
pip install -r requirements.txt
cd src/diffusers
pip install -e .
cd ../peft
pip install -e .
```
You can also use ```conda``` instead of ```mamba``` if it is not installed.

## 2. Dataset
Download the dataset from the following links: [sdv2](https://drive.google.com/file/d/1Vl3QFGceeD1-MsWVk9IegubKet_cZtSE/view?usp=drive_link) and unzip it into the path ``dataset/``.
The following 9 protective perturbations are evaluated: 
1. AdvDM(+): ``--adv_type=advdm+``
2. AdvDM(-): ``--adv_type=advdm-``
3. Mist: ``--adv_type=mist``
4. SDS(+): ``--adv_type=sds+``
5. SDS(-): ``--adv_type=sds-``
6. SDST: ``--adv_type=sdsT5``
7. Glaze: ``--adv_type=glaze2``
8. Anti-DreamBooth: ``--adv_type=anti-dreambooth``
9. Metacloak: ``--adv_type=metacloak``

for 3 datasets: CelebA-HQ, VGGFace2 and WikiArt.

## 3. Customization
We use the [Stable Diffusion v2.1](https://github.com/Stability-AI/stablediffusion) ([HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-2-1)) as the base pre-trained model.

#### 3.1. Using DreamBooth
To fine-tune a model using DreamBooth, we first give the following example:
```bash
python code/finetune_dreambooth_cat/finetune_dreambooth_cat.py \
--exp_type=cat \
--dataset=celebahq \
--adv_type=advdm+ \
--target_module=both \
--cat_adapter_rank=128 \
--cat_adapter_train_steps=10000 \
--class_data_dir=dataset/db-reference \
--subfolder=5
```
This example fine-tunes a model on the CelebA-HQ dataset (subfolder equals 5) protecetd with AdvDM(+) using the CAT adapter with the rank of 128 (target moddule is both). 
The CAT adapter is fine-tuned for 10,000 steps and the model is fine-tuned for 2,000 steps by default.
We also provide the clean and baseline (without CAT adapter) configs as follows:
```bash
python code/finetune_dreambooth_cat/finetune_dreambooth_cat.py \
--exp_type=clean \
--dataset=celebahq \
--adv_type=advdm+ \
--class_data_dir=dataset/db-reference \
--subfolder=5
```
and
```bash
python code/finetune_dreambooth_cat/finetune_dreambooth_cat.py \
--exp_type=baseline \
--dataset=celebahq \
--adv_type=advdm+ \
--class_data_dir=dataset/db-reference \
--subfolder=5
```

#### 3.2. Using LoRA
The customization dataset for LoRA needs to be prepared:
```bash
cd dataset
cp -r sdv2 sdv2_lora
bash add_metadata.sh
```

To fine-tune a model using LoRA, we first give the following example:
```bash
python code/finetune_lora_cat/finetune_lora_cat.py \
--exp_type=cat \
--dataset=celebahq \
--adv_type=advdm+ \
--target_module=both \
--cat_adapter_rank=128 \
--cat_adapter_train_steps=10000 \
--subfolder=5
```
also the baseline and clean can be tested by replacing the configs.

For customization for style mimicry, use the ``--dataset=wikiart`` and choose the subfolder accordingly.


## 4. Sampling
To sample the images using DreamBooth customization, we first give the following example:
```bash
python code/sample/sample_dreambooth.py \
--exp_type=cat \
--dataset=celebahq \
--adv_type=advdm+ \
--target_module=both \
--cat_adapter_rank=128 \
--cat_adapter_train_steps=10000 \
--subfolder=5
```
We also give the example for LoRA customization:
```bash
python code/sample/sample_lora.py \
--exp_type=cat \
--dataset=celebahq \
--adv_type=advdm+ \
--target_module=both \
--cat_adapter_rank=128 \
--cat_adapter_train_steps=10000 \
--subfolder=5
```

## 5. Evaluation
Due to the package conflicts, we first prepare the evaluation environment for the metric ``Retina-FDR`` and ``ISM``:
```bash
mamba create --clone cat -n cat-ism
mamba activate cat-ism
cd retinaface
pip install -e .
cd ../deepface
pip install -e .
pip install tf-keras
pip install tensorflow[and-cuda]
```
Then we can evaluate the sampling results by:
```bash
python code/evaluate/eval_ism.py \
--exp_type=cat \
--dataset=celebahq \
--adv_type=advdm+ \
--target_module=both \
--cat_adapter_rank=128 \
--cat_adapter_train_steps=10000 \
--subfolder=5
```
To evaluate the metric ``TOPIQ-FDR`` and ``FIQ`` implemented in [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch), we also needs to first prepare the environment:
```bash
mamba create --clone cat -n cat-iqa
mamba activate cat-iqa
cd src/IQA-PyTorch
pip install -e .
```
Then we can evaluate the sampling results by:
```bash
python code/evaluate/eval_iqa.py \
--metric=topiq_nr_swin-face \
--exp_type=cat \
--dataset=celebahq \
--adv_type=advdm+ \
--target_module=both \
--cat_adapter_rank=128 \
--cat_adapter_train_steps=10000 \
--subfolder=5
```

## 6. Citation