import subprocess
import threading
import argparse
import shutil
import os
import tempfile


SHORT_NAME = {
    "both": "both",
    "encoder": "en",
    "decoder": "de",
}


def create_tmp_train_dir(in_dir):
    # Create a temporary directory
    tmp_dir = tempfile.mkdtemp()

    # Create subdirectories 'train' and 'val' inside the temporary directory
    train_dir = os.path.join(tmp_dir, "train")
    val_dir = os.path.join(tmp_dir, "val")
    os.makedirs(train_dir)
    os.makedirs(val_dir)

    # Copy all files from in_dir to the 'train' subdirectory
    for filename in os.listdir(in_dir):
        src_file = os.path.join(in_dir, filename)
        dst_file = os.path.join(train_dir, filename)
        if os.path.isfile(src_file):
            shutil.copy(src_file, dst_file)

    return tmp_dir


def process_directories(root, directories, method):
    while directories:
        # Start two threads if there are at least two directories left
        threads = []
        for i in args.gpu_ids:
            if not directories:
                break
            dir = directories.pop()
            print(f"Processing {dir}", flush=True)
            cuda_device = i
            thread = threading.Thread(target=method, args=(cuda_device, dir))
            thread.start()
            threads.append(thread)

        # Wait for both threads to complete
        for thread in threads:
            thread.join()


def run_command_sd(
    instance_data_dir,
    output_dir,
    max_train_steps,
    target_module,
    rank,
    tracker_project_name,
    run_name,
    run_tags,
    loss_type="mse",
    resolution=512,
    cuda=0,
    pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1",
    train_batch_size=4,
    checkpointing_steps=10000,
    random_flip=False,
    num_processes=1,
):
    # default gradient_checkpointing, mixed_precision=bf16
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{cuda}"
    command = f"""
    accelerate launch --num_processes={num_processes} code/finetune_cat_adapter/train_cat_adapter.py \
    --pretrained_model_name_or_path="{pretrained_model_name_or_path}" \
    --train_data_dir="{instance_data_dir}" \
    --resolution={resolution} \
    --center_crop \
    --train_batch_size={train_batch_size} \
    --max_train_steps={max_train_steps} \
    --learning_rate=1e-04 \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=0 \
    --output_dir="{output_dir}" \
    --report_to="wandb" \
    --validation_epochs=100 \
    --checkpointing_steps={checkpointing_steps} \
    --seed=0 \
    --target_module={target_module} \
    --rank={rank} \
    --loss_type={loss_type} \
    --tracker_project_name={tracker_project_name} \
    --run_name={run_name} \
    --run_tags={run_tags} \
    """
    if random_flip:
        command += "--random_flip"
    subprocess.run(command, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # default config
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--device", type=int, default=0)
    # exp related config
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["vggface2", "celebahq","wikiart"],
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
    parser.add_argument(
        "--target_module",
        type=str,
        required=True,
        choices=["both", "encoder", "decoder"],
    )
    parser.add_argument(
        "--cat_adapter_rank",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--instance_dir",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    tracker_project_name = "finetuning-cat-adapter"

    instance_name = args.instance_dir.split("/")[-1]
    output_dir = f"./outputs/models/cat_adapters/{args.target_module}/{args.adv_type}/{args.dataset}/{instance_name}/r{args.cat_adapter_rank}"
    run_name = (
        f"{SHORT_NAME[args.target_module]}_{args.adv_type}_{args.dataset}_sub{instance_name}_r{args.cat_adapter_rank}"
    )
    run_tags = f"{args.target_module},{args.adv_type},{args.dataset},sub{instance_name},r{args.cat_adapter_rank}"

    tmp_train_dir = create_tmp_train_dir(args.instance_dir)
    run_command_sd(
        instance_data_dir=tmp_train_dir,
        output_dir=output_dir,
        max_train_steps=args.max_train_steps,
        target_module=args.target_module,
        rank=args.cat_adapter_rank,
        tracker_project_name=tracker_project_name,
        run_name=run_name,
        run_tags=run_tags,
        resolution=args.resolution,
        cuda=args.device,
    )
    shutil.rmtree(tmp_train_dir)
