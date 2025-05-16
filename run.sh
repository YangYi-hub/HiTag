#!/usr/bin/env bash
set -euo pipefail

eval_ew_tes_res="eval_ew_tes_res"

if [ $# -ne 3 ]; then
  echo "usage: $0 <config_path> <output_pth_file_path> <eval_res_path>"
  exit 1
fi

config_path="$1"
output_pth_file_path="$2"
eval_res_path="$3"

python -m torch.distributed.run --nproc_per_node=8 train.py \
  --model-type hitag \
  --config "${config_path}" \
  --output-dir "${output_pth_file_path}"

for ckpt in 09; do
  python batch_inference_openimage.py \
    --model-type hitag \
    --checkpoint "${output_pth_file_path}/checkpoint_${ckpt}.pth" \
    --dataset openimages_common_214 \
    --input-size 224 \
    --output-dir "${eval_res_path}/open_common/${ckpt}"
  
  python batch_inference_openimage.py \
    --model-type hitag \
    --checkpoint "${output_pth_file_path}/checkpoint_${ckpt}.pth" \
    --open-set True \
    --dataset openimages_rare_200 \
    --input-size 224 \
    --output-dir "${eval_res_path}/open_rare/${ckpt}"

  python batch_inference_imagenet.py \
    --model-type hitag \
    --checkpoint "${output_pth_file_path}/checkpoint_${ckpt}.pth" \
    --open-set True \
    --dataset imagenet_multi_1000 \
    --input-size 224 \
    --output-dir "${eval_res_path}/imagenet1000/${ckpt}"

  python batch_inference_tag_hire.py \
  --model-type hitag \
  --checkpoint "${output_pth_file_path}/checkpoint_${ckpt}.pth" \
  --open-set True \
  --dataset hier_openimage_common \
  --input-size 224 \
  --output-dir "${eval_res_path}/hier_openimage_common/${ckpt}"
done


filename=$(basename "${config_path}")        
filename_noext="${filename%.*}"             
json_save_path="${eval_ew_tes_res}/${filename_noext}.json"

python parse_eval_summaries.py \
  --eval-res-path "${eval_res_path}" \
  --ckpts 09 \
  --json-save-path "${json_save_path}"