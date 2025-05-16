import os
import json
import argparse

def parse_summary_file(summary_file_path):
    if not os.path.exists(summary_file_path):
        return {}

    result = {}
    with open(summary_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("mAP:"):
                val_str = line.split(":", 1)[1].strip()
                result["mAP"] = float(val_str)
            elif line.startswith("CP:"):
                val_str = line.split(":", 1)[1].strip()
                result["CP"] = float(val_str)
            elif line.startswith("CR:"):
                val_str = line.split(":", 1)[1].strip()
                result["CR"] = float(val_str)
    return result

def parse_hier_summary_file(summary_file_path):
    if not os.path.exists(summary_file_path):
        return {}

    result = {}
    with open(summary_file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            if i == 1:
                line = line.strip()
                items = line.strip("[]").split(",")
                
                for item in items:
                    key, val = item.split(":")
                    result[key.strip()] = float(val.strip())

    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-res-path", type=str, required=True,
                        help="Path to the directory containing evaluation results.")
    parser.add_argument("--ckpts", type=str, nargs='+', required=True,
                        help="List of checkpoint IDs (e.g. 04 05 06 ...).")
    parser.add_argument("--json-save-path", type=str, required=True,
                        help="Path to save the final JSON.")
    args = parser.parse_args()

    data = {
        "open_rare": {},
        "open_common": {},
        "imagenet1000": {},
        "hier_openimage_common": {}
    }

    for ckpt in args.ckpts:
        # open_rare
        open_rare_summary_path = os.path.join(args.eval_res_path, "open_rare", ckpt, "summary.txt")
        data["open_rare"][ckpt] = parse_summary_file(open_rare_summary_path)
        
        #open_common
        open_common_summary_path = os.path.join(args.eval_res_path, "open_common", ckpt, "summary.txt")
        data["open_common"][ckpt] = parse_summary_file(open_common_summary_path)
        
        # imagenet_multi_1000
        imagenet1000_summary_path = os.path.join(args.eval_res_path, "imagenet1000", ckpt, "summary.txt")
        data["imagenet1000"][ckpt] = parse_summary_file(imagenet1000_summary_path)
        
        # hier_openimage_common
        hier_openimage_common_summary_path = os.path.join(args.eval_res_path, "hier_openimage_common", ckpt, "summary.txt")
        data["hier_openimage_common"][ckpt] = parse_hier_summary_file(hier_openimage_common_summary_path)

    os.makedirs(os.path.dirname(args.json_save_path), exist_ok=True)
    with open(args.json_save_path, "w", encoding="utf-8") as jf:
        json.dump(data, jf, ensure_ascii=False, indent=2)

    print(f"Summary results saved to: {args.json_save_path}")

if __name__ == "__main__":
    main()