import pyiqa
import torch
import os
import argparse
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

METRIC_REF = {
    "fid": True,
    "ssim": True,
    "psnr": True,
    "lpips": True,
    "clipiqa": False,
    "topiq_nr_swin-face": False,
    "qalign_8bit": False,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # default config
    parser.add_argument("--device", type=int, default=3)
    # exp related config
    parser.add_argument(
        "--exp_type",
        type=str,
        default="cat",
        choices=["baseline", "clean", "cat"],
    )
    parser.add_argument(
        "--sample_dir",
        type=str,
        default="dreambooth_cat",
        choices=["dreambooth_cat", "lora_cat"],
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        choices=[
            "fid",
            "ssim",
            "psnr",
            "lpips",
            "clipiqa",
            "topiq_nr_swin-face",
            "qalign_8bit",
        ],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["vggface2", "celebahq", "wikiart"],
    )
    parser.add_argument(
        "--adv_type",
        type=str,
        required=True,
        choices=[
            "advdm+",
            "advdm-",
            "mist",
            "sds+",
            "sds-",
            "sdsT5",
            "glaze2",
            "anti-dreambooth",
            "metacloak",
        ],
    )
    # CAT adapter related config
    parser.add_argument("--cat_adapter_rank", type=int, default=None)
    parser.add_argument(
        "--target_module",
        type=str,
        choices=["both", "encoder", "decoder"],
    )
    parser.add_argument(
        "--cat_adapter_train_steps",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default=None,
        required=True,
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    ref_flag = METRIC_REF[args.metric]
    iqa_metric = pyiqa.create_metric(args.metric, device="cuda")
    print(f"Metric: {args.metric}, Lower better: {iqa_metric.lower_better}")

    sp_sample = args.subfolder

    if args.cat_adapter_rank is None:
        cat_adapter_rank = 128 if args.target_module == "both" else 256
    else:
        cat_adapter_rank = args.cat_adapter_rank

    if ref_flag:
        raise NotImplementedError("Reference based IQA metric is not supported yet")
    else:
        if args.exp_type == "clean":
            sample_data_dir = (
                f"outputs/samples/{args.sample_dir}/clean/{args.dataset}/{sp_sample}"
            )
        elif args.exp_type == "baseline":
            sample_data_dir = f"outputs/samples/{args.sample_dir}/baseline/{args.adv_type}/{args.dataset}/{sp_sample}"
        else:
            if (
                args.cat_adapter_train_steps is not None
                and args.cat_adapter_train_steps != 10000
            ):
                sample_data_dir = f"outputs/samples/{args.sample_dir}/{args.target_module}/{args.adv_type}/{args.dataset}/{sp_sample}/r{cat_adapter_rank}/CATstep{args.cat_adapter_train_steps}/checkpoint-{args.cat_adapter_train_steps}"
            else:
                sample_data_dir = f"outputs/samples/{args.sample_dir}/{args.target_module}/{args.adv_type}/{args.dataset}/{sp_sample}/r{cat_adapter_rank}/checkpoint-10000"

    iqa_scores = []
    total_c, nf_c = 0, 0
    for img_file in tqdm(os.listdir(sample_data_dir)):
        if img_file.endswith((".png", ".jpg", ".jpeg")):
            total_c += 1
            img_path = os.path.join(sample_data_dir, img_file)
            try:
                score_fr = iqa_metric(img_path).detach().cpu().numpy()
                iqa_scores.append(score_fr)
            except AssertionError:
                print("No face detected")
                nf_c += 1
    metric_value = sum(iqa_scores) / len(iqa_scores)
    print(
        f"{args.dataset}, {args.adv_type}, Mean {args.metric} score: {metric_value[0][0]}, Face detection rate: {1-nf_c/total_c}"
    )
