import os
import torch
import argparse
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel
from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
from safetensors.torch import load_file

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
    # sampling related config
    parser.add_argument("--sample_size", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument(
        "--sample_prompts",
        type=str,
        default="a photo of sks person,a dslr portrait of sks person",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    if args.dataset == "wikiart":
        sp_sample = args.subfolder
        artist=" ".join(args.subfolder.split("-"))
        args.sample_prompts = f"a painting of shoe with a plant growing inside by {artist} artist"
    else:
        sp_sample = args.subfolder

    model_path = "dreambooth_cat"
    if args.exp_type == "baseline":
        db_path = f"./outputs/models/{model_path}/{args.exp_type}/{args.adv_type}/{args.dataset}/{sp_sample}/checkpoint-{args.max_db_train_steps}"
    elif args.exp_type == "clean":
        db_path = f"./outputs/models/{model_path}/{args.exp_type}/{args.dataset}/{sp_sample}/checkpoint-{args.max_db_train_steps}"
    else:
        if args.cat_adapter_train_steps!=10000:
            db_path = f"./outputs/models/{model_path}/{args.target_module}/{args.adv_type}/{args.dataset}/{sp_sample}/r{args.cat_adapter_rank}/CATstep{args.cat_adapter_train_steps}/checkpoint-{args.max_db_train_steps}"
        else:
            db_path = f"./outputs/models/{model_path}/{args.target_module}/{args.adv_type}/{args.dataset}/{sp_sample}/r{args.cat_adapter_rank}/checkpoint-{args.max_db_train_steps}"

    unet = UNet2DConditionModel.from_pretrained(db_path, subfolder="unet")
    text_encoder = CLIPTextModel.from_pretrained(db_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-2-1", subfolder="vae"
    )

    # load cat adapter
    if args.exp_type == "baseline":
        output_path = f"outputs/samples/{model_path}/{args.exp_type}/{args.adv_type}/{args.dataset}/{sp_sample}"
    elif args.exp_type == "clean":
        output_path = (
            f"outputs/samples/{model_path}/{args.exp_type}/{args.dataset}/{sp_sample}"
        )
    else:
        cat_adapter_path = f"outputs/models/cat_adapters/{args.target_module}/{args.adv_type}/{args.dataset}/{sp_sample}/r{args.cat_adapter_rank}/checkpoint-{args.cat_adapter_train_steps}/pytorch_lora_weights.safetensors"

        if args.cat_adapter_train_steps!=10000:
            output_path = f"outputs/samples/{model_path}/{args.target_module}/{args.adv_type}/{args.dataset}/{sp_sample}/r{args.cat_adapter_rank}/CATstep{args.cat_adapter_train_steps}/checkpoint-{args.cat_adapter_train_steps}"
        else:
            output_path = f"outputs/samples/{model_path}/{args.target_module}/{args.adv_type}/{args.dataset}/{sp_sample}/r{args.cat_adapter_rank}/checkpoint-{args.cat_adapter_train_steps}"

        if args.target_module == "encoder":
            cat_adapter_config = LoraConfig(
                r=args.cat_adapter_rank,
                lora_alpha=args.cat_adapter_rank,
                init_lora_weights="gaussian",
                target_modules=".*encoder.*(conv|conv1|conv2|to_q|to_k|to_v|to_out\\.0)$",
            )
        elif args.target_module == "decoder":
            cat_adapter_config = LoraConfig(
                r=args.cat_adapter_rank,
                lora_alpha=args.cat_adapter_rank,
                init_lora_weights="gaussian",
                target_modules=".*decoder.*(conv|conv1|conv2|to_q|to_k|to_v|to_out\\.0)$",
            )
        elif args.target_module == "both":
            cat_adapter_config = LoraConfig(
                r=args.cat_adapter_rank,
                lora_alpha=args.cat_adapter_rank,
                init_lora_weights="gaussian",
                target_modules=[
                    "conv",
                    "conv1",
                    "conv2",
                    "to_q",
                    "to_k",
                    "to_v",
                    "to_out.0",
                ],
            )
        else:
            raise ValueError("Invalid target module")
        vae = inject_adapter_in_model(cat_adapter_config, vae)

        # load adapter weights
        peft_state_dict = load_file(cat_adapter_path)
        peft_state_dict = {
            k.replace("vae.", ""): v
            for k, v in peft_state_dict.items()
            if k.startswith("vae.")
        }
        peft_state_dict = {
            k.replace("lora.up", "lora_B") if "lora.up" in k else k: v
            for k, v in peft_state_dict.items()
        }
        peft_state_dict = {
            k.replace("lora.down", "lora_A") if "lora.down" in k else k: v
            for k, v in peft_state_dict.items()
        }

        outcome = set_peft_model_state_dict(vae, peft_state_dict)
        assert (
            outcome.unexpected_keys == []
        ), f"Unexpected keys: {outcome.unexpected_keys}"
        print(f"Successfully loaded the CAT adapter from {cat_adapter_path}")

    pipeline = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        safety_checker=None,
    ).to("cuda")

    prompts = args.sample_prompts.split(",")

    for prompt in prompts:
        print(">>>>>>", prompt)
        norm_prompt = prompt.lower().replace(",", "").replace(" ", "_")
        os.makedirs(output_path, exist_ok=True)

        batch_nums = args.sample_size // args.batch_size

        for batch_num in range(batch_nums):
            images = pipeline(
                [prompt] * args.batch_size,
                num_inference_steps=100,
                guidance_scale=7.5,
                height=args.resolution,
                width=args.resolution,
            ).images
            for idx, image in enumerate(images):
                image.save(
                    f"{output_path}/{norm_prompt}_{idx+batch_num*args.batch_size}.png"
                )
                del image
    del pipeline
    torch.cuda.empty_cache()
