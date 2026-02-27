## Model Card

> [!NOTE] 
> The model weights are currently undergoing internal legal review and will be released on 🤗 Huggingface as soon as they are cleared.

| dataset | 🤗 hf model repo | params | encoder size | decoder size | 
|:---:|:---:|:---:|:---:|:---:|
| Animal-Faces | `sphere-l-af` | 642M | large | large |
| Oxford-Flowers | `sphere-l-of` | 948M | large | large |
| ImageNet | `sphere-l-imagenet` | 950M | large | large |
| ImageNet | `sphere-xl-imagenet` | 1.3B | xlarge | xlarge |

## Evaluation

Evaluate **ImageNet** models with `CFG = 1.4`:

```bash
# --job_dir can be
#   sphere-l-imagenet, or sphere-xl-imagenet

./run.sh eval.py \
  --job_dir sphere-xl-imagenet \
  --forward_steps 1 4 \
  --report_fid rfid gfid \
  --use_cfg True \
  --cfg_min 1.4 \
  --cfg_max 1.4 \
  --cfg_position combo \
  --rm_folder_after_eval True
```

| dataset | model | steps | rFID &darr; | gFID &darr; | IS &uarr; |
|:--:|:--|:--:|:--:|--:|:--:|
ImageNet 256x256 | Sphere-L | 1 | 0.62 | 15.69 | 274.5 |
|| Sphere-L | 4 | - | 4.78 | 259.1 |
|| Sphere-XL | 1 | 0.62 | 14.52 | 299.3 |
|| Sphere-XL | 4 | - | 4.05 | 266.0 |

Evaluate unconditional **Animal-Faces** model:

```bash
./run.sh eval.py \
  --job_dir sphere-l-af \
  --forward_steps 1 4 \
  --report_fid gfid \
  --rm_folder_after_eval True
```

| dataset | model | steps | rFID &darr; | gFID &darr; | IS &uarr; |
|:--:|:--|:--:|:--:|:--:|:--:|
Animal-Faces 256x256 | Sphere-L | 1 | - | 21.56 | 8.3 |
|| Sphere-L | 4 | - | 18.73 | 9.8 |

Evaluate **Oxford-Flowers** model with `CFG = 1.4`:

```bash
./run.sh eval.py \
  --job_dir sphere-l-of \
  --forward_steps 1 4 \
  --report_fid gfid \
  --use_cfg True \
  --cfg_min 1.6 \
  --cfg_max 1.6 \
  --cfg_position combo \
  --num_eval_samples 51000 \
  --rm_folder_after_eval True \
  --cache_sampling_noise False \
```

`--num_eval_samples = 51000` are set for 102 classes such that each class has 500 samples for evaluation on 8 gpus. 
Adjust them accordingly if you have different number of gpus or want to evaluate on different number of samples.

| dataset | model | steps | rFID &darr; | gFID &darr; | IS &uarr; |
|:--:|:--|:--:|:--:|:--:|:--:|
| Oxford-Flowers 256x256 | Sphere-L | 1 | - | 25.10 | 3.4 |
|| Sphere-L | 4 | - | 11.27 | 3.2 |