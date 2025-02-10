import subprocess
import argparse
import os

SHORT_NAME = {
    "both": "both",
    "encoder": "en",
    "decoder": "de",
}


def run_command_ft_dreambooth(
    instance_data_dir,
    class_data_dir,
    instance_prompt,
    class_prompt,
    out_dir,
    validation_prompt,
    tracker_project_name,
    run_name,
    run_tags,
    cat_adapter,
    cat_adapter_path=None,
    rank=None,
    target_module=None,
    cuda=0,
    max_train_steps=2000,
    pretrain_name="stabilityai/stable-diffusion-2-1",
    prior_loss_weight=1,
    resolution=512,
    train_batch_size=2,
    checkpointing_steps=2000,
    num_processes=1,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{cuda}"
    command = f"""
    accelerate launch --num_processes={num_processes} code/finetune_dreambooth_cat/train_dreambooth_cat.py \
    --pretrained_model_name_or_path="{pretrain_name}" \
    --instance_data_dir="{instance_data_dir}" \
    --class_data_dir="{class_data_dir}" \
    --instance_prompt="{instance_prompt}" \
    --class_prompt="{class_prompt}" \
    --with_prior_preservation \
    --prior_loss_weight={prior_loss_weight} \
    --num_class_images=200 \
    --output_dir="{out_dir}" \
    --seed=0 \
    --resolution={resolution} \
    --center_crop \
    --train_text_encoder \
    --train_batch_size={train_batch_size} \
    --sample_batch_size=40 \
    --max_train_steps={max_train_steps} \
    --checkpointing_steps={checkpointing_steps} \
    --gradient_accumulation_steps=1 \
    --learning_rate=5e-7 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --report_to="wandb" \
    --validation_prompt="{validation_prompt}" \
    --num_validation_images=4 \
    --mixed_precision=bf16 \
    --prior_generation_precision=bf16 \
    --tracker_project_name={tracker_project_name} \
    --run_name={run_name} \
    --run_tags={run_tags} \
    """
    if cat_adapter:
        command += f"""--cat_adapter \
            --cat_adapter_path={cat_adapter_path} \
            --target_module={target_module} \
            --rank={rank} """
    subprocess.run(command, shell=True)


def run_command_ft_cat_adapter(
    instance_data_dir,
    dataset,
    adv_type,
    target_module,
    cat_adapter_rank,
    max_train_steps,
    device=0,
):
    # default gradient_checkpointing, mixed_precision=bf16
    command = f"""python ./code/finetune_cat_adapter/finetune_cat_adapter.py \
        --device={device} \
        --dataset={dataset} \
        --adv_type={adv_type} \
        --target_module={target_module} \
        --cat_adapter_rank={cat_adapter_rank} \
        --max_train_steps={max_train_steps} \
        --instance_dir={instance_data_dir}
    """
    subprocess.run(command, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # default config
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--device", type=int, default=0)
    # exp related config
    parser.add_argument(
        "--exp_type",
        type=str,
        default="baseline",
        choices=["baseline", "clean", "cat"],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="vggface2",
        choices=["vggface2", "celebahq", "wikiart"],
    )
    parser.add_argument(
        "--adv_type",
        type=str,
        default=None,
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
    parser.add_argument(
        "--target_module",
        type=str,
        default=None,
        choices=["both", "encoder", "decoder"],
    )
    parser.add_argument("--cat_adapter_rank", type=int, default=128)
    parser.add_argument(
        "--cat_adapter_train_steps",
        type=int,
        default=10000,
    )
    # dreambooth related config
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
    )
    parser.add_argument("--class_prompt", type=str, default="a photo of person")
    parser.add_argument("--instance_prompt", type=str, default="a photo of sks person")
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default="a dslr portrait of sks person",
    )
    parser.add_argument(
        "--max_db_train_steps",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default=None,
        required=True,
    )
    args = parser.parse_args()
    
    if args.dataset == "wikiart":
        subfolders = [args.subfolder]
        args.class_prompt = "a drawing by artist"
        artist = " ".join(args.subfolder.split("_"))
        args.instance_prompt = f"a drawing by {artist} artist"
        args.validation_prompt = f"a painting by {artist} artist"
    else:
        subfolders = [args.subfolder]
        
    if args.exp_type == "clean":
        assert args.adv_type is None
        instance_dir = f"./dataset/sdv2/{args.dataset}/clean"
    else:
        instance_dir = f"./dataset/sdv2/{args.dataset}/{args.adv_type}"

    for subfolder in subfolders:
        instance_data_dir = f"{instance_dir}/{subfolder}"

        if args.exp_type == "baseline":
            output_dir = f"./outputs/models/dreambooth_cat/baseline/{args.adv_type}/{args.dataset}/{subfolder}"
            run_name = f"baseline_{args.adv_type}_{args.dataset}_{subfolder}"
            run_tags = f"baseline,{args.adv_type},{args.dataset},{subfolder}"
        elif args.exp_type == "clean":
            output_dir = f"./outputs/models/dreambooth_cat/clean/{args.dataset}/{subfolder}"
            run_name = f"clean_{args.dataset}_{subfolder}"
            run_tags = f"clean,{args.dataset},{subfolder}"
        elif args.exp_type == "cat":
            cat_adapter_path = f"./outputs/models/cat_adapters/{args.target_module}/{args.adv_type}/{args.dataset}/{subfolder}/r{args.cat_adapter_rank}/checkpoint-{args.cat_adapter_train_steps}/pytorch_lora_weights.safetensors"

            if not os.path.isfile(cat_adapter_path):
                print(f"CAT adapter not found. Fine-tuning CAT adapter...")
                run_command_ft_cat_adapter(
                    instance_data_dir=instance_data_dir,
                    dataset=args.dataset,
                    adv_type=args.adv_type,
                    target_module=args.target_module,
                    cat_adapter_rank=args.cat_adapter_rank,
                    max_train_steps=args.cat_adapter_train_steps,
                    device=args.device,
                )

            if args.cat_adapter_train_steps != 10000:
                run_name = f"{SHORT_NAME[args.target_module]}_{args.adv_type}_{args.dataset}_{subfolder}_r{args.cat_adapter_rank}_CATstep{args.cat_adapter_train_steps}"
                run_tags = f"{args.target_module},{args.adv_type},{args.dataset},{subfolder},r{args.cat_adapter_rank},CATstep{args.cat_adapter_train_steps}"
                output_dir = f"./outputs/models/dreambooth_cat/{args.target_module}/{args.adv_type}/{args.dataset}/{subfolder}/r{args.cat_adapter_rank}/CATstep{args.cat_adapter_train_steps}"
            else:
                run_name = f"{SHORT_NAME[args.target_module]}_{args.adv_type}_{args.dataset}_{subfolder}_r{args.cat_adapter_rank}"
                run_tags = f"{args.target_module},{args.adv_type},{args.dataset},{subfolder},r{args.cat_adapter_rank}"
                output_dir = f"./outputs/models/dreambooth_cat/{args.target_module}/{args.adv_type}/{args.dataset}/{subfolder}/r{args.cat_adapter_rank}"

        tracker_project_name = "finetuning-dreambooth-cat"
        run_command_ft_dreambooth(
            instance_data_dir=instance_data_dir,
            class_data_dir=args.class_data_dir,
            instance_prompt=args.instance_prompt,
            class_prompt=args.class_prompt,
            out_dir=output_dir,
            validation_prompt=args.validation_prompt,
            tracker_project_name=tracker_project_name,
            run_name=run_name,
            run_tags=run_tags,
            cat_adapter=True if args.exp_type == "cat" else False,
            cat_adapter_path=cat_adapter_path if args.exp_type == "cat" else None,
            rank=args.cat_adapter_rank if args.exp_type == "cat" else None,
            max_train_steps=args.max_db_train_steps,
            target_module=args.target_module if args.exp_type == "cat" else None,
            cuda=args.device,
        )
