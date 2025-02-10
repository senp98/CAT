from deepface import DeepFace
import numpy as np
import os
import torch
import torch.nn.functional as F
import argparse
from compute_idx_emb import compute_idx_embedding
import warnings
import tensorflow as tf

warnings.filterwarnings("ignore")


def compute_face_embedding(img_path):
    """Extract face embedding vector of given image
    Args:
        img_path (str): path to image
    Returns:
        None: no face found
        vector: return the embedding of biggest face among the all found faces
    """
    try:
        resps = DeepFace.represent(
            img_path=os.path.join(img_path),
            model_name="ArcFace",
            enforce_detection=True,
            detector_backend="retinaface",
            align=True,
        )
        if resps == 1:
            # detect only 1 face
            return np.array(resps[0]["embedding"])
        else:
            # detect more than 1 faces, choose the biggest one
            resps = list(resps)
            resps.sort(key=lambda resp: resp["facial_area"]["h"] * resp["facial_area"]["w"], reverse=True)
            return np.array(resps[0]["embedding"])
    except Exception:
        # no face found
        return None


def get_precomputed_embedding(path):
    """Get face embedding by loading the path to numpy file
    Args:
        path (str): path to numpy file
    Returns:
        vector: face embedding
    """
    return np.load(path)


def matching_score_id(image_path, avg_embedding):
    """getting the matching score between face image and precomputed embedding

    Args:
        img (2D images): images
        emb (vector): face embedding

    Returns:
        None: cannot detect face from img
        int: identity score matching
    """
    image_emb = compute_face_embedding(image_path)
    id_emb = avg_embedding
    if image_emb is None:
        return None
    image_emb, id_emb = torch.Tensor(image_emb), torch.Tensor(id_emb)
    ism = F.cosine_similarity(image_emb, id_emb, dim=0)
    return ism


def matching_score_genimage_id(images_path, list_id_path):
    image_list = os.listdir(images_path)
    fail_detection_count = 0
    ave_ism = 0
    avg_embedding = compute_idx_embedding(list_id_path)

    for image_name in image_list:
        image_path = os.path.join(images_path, image_name)
        ism = matching_score_id(image_path, avg_embedding)
        if ism is None:
            fail_detection_count += 1
        else:
            ave_ism += ism
    if fail_detection_count != len(image_list):
        return ave_ism / (len(image_list) - fail_detection_count), fail_detection_count / len(image_list)
    return None, 1



def main():
    parser = argparse.ArgumentParser()
    # default config
    parser.add_argument("--device", type=int, default=2)
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
        choices=["vggface2", "celebahq"],
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
        choices=["both", "encoder", "decoder"],
    )
    parser.add_argument(
        "--cat_adapter_train_steps",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument("--cat_adapter_rank", type=int, default=None)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    tf.random.set_seed(0)
    
    sp_sample = args.subfolder

    if args.cat_adapter_rank is None:
        cat_adapter_rank = 128 if args.target_module == "both" else 256
    else:
        cat_adapter_rank = args.cat_adapter_rank
    
    if args.exp_type == "clean":
        sample_data_dir = f"outputs/samples/{args.sample_dir}/clean/{args.dataset}/{sp_sample}"
    elif args.exp_type == "baseline":
        sample_data_dir = f"outputs/samples/{args.sample_dir}/baseline/{args.adv_type}/{args.dataset}/{sp_sample}"
    else:
        if args.cat_adapter_train_steps is not None and args.cat_adapter_train_steps != 10000:
            sample_data_dir = f"outputs/samples/{args.sample_dir}/{args.target_module}/{args.adv_type}/{args.dataset}/{sp_sample}/r{cat_adapter_rank}/CATstep{args.cat_adapter_train_steps}/checkpoint-{args.cat_adapter_train_steps}"
        else:
            sample_data_dir = f"outputs/samples/{args.sample_dir}/{args.target_module}/{args.adv_type}/{args.dataset}/{sp_sample}/r{cat_adapter_rank}/checkpoint-10000"
    
    ref_data_dir = f"dataset/sdv2/{args.dataset}/clean/{sp_sample}"
    print(f"Sample data dir: {sample_data_dir}")

        
    ism, fdr = matching_score_genimage_id(sample_data_dir, ref_data_dir)
    print(f"{args.dataset}, {args.adv_type}, {args.target_module}, ISM and FDR are {ism} and {1-fdr}")


if __name__ == "__main__":
    main()
