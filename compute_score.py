import json
import args

from utils.grader import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default="",
                        help="Path to the inference results file (jsonl)")
    parser.add_argument("--k", type=int, default=1,
                        help="Value of k for pass@k calculation")
    return parser.parse_args()

def computer_score(args):
    out_file = args.file_path
    correct_cnt = 0
    pass_at_k_list = []
    k = args.k
    examples = []
    for line in open(out_file, 'r'):
        example = json.loads(line)
        examples.append(example)
        generated_answers = example["generated_answers"]
        gt_ans = example["gold_answer"]
        is_correct_list = [check_is_correct(generated_answer, gt_ans) for generated_answer in generated_answers]
        is_correct = any(is_correct_list)
        if is_correct:
            correct_cnt += 1
        if len(is_correct_list) > 1:
            correct_answers = sum(is_correct_list)
            n = len(generated_answers)
            if correct_answers > 0:
                if n - correct_answers < k:
                    pass_at_k = 1
                else:
                    pass_at_k = 1 - (comb(n - correct_answers, k) / comb(n, k))
                pass_at_k_list.append(pass_at_k)
            else:
                pass_at_k_list.append(0)
                
    metrics = []
    
    print(f"correct cnt / total cnt: {correct_cnt}/{len(examples)}")
    print(f"Acc: {correct_cnt / len(examples):.4f}")
    metrics.append({'Acc': correct_cnt / len(examples)})

    if pass_at_k_list:
        average_pass_at_k = sum(pass_at_k_list) / len(pass_at_k_list)
        print(f"Pass@{k}: {sum(pass_at_k_list)}/{len(pass_at_k_list)} = {average_pass_at_k:.4f}")
        metrics.append({'Pass@'+str(k): average_pass_at_k})
    else:
        print(f"Pass@1: {correct_cnt}/{len(examples)} = {correct_cnt / len(examples):.4f}")
        metrics.append({'Pass@1': correct_cnt / len(examples)})

    with open(out_file+'_metrics.json', 'w') as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    args = parse_args()
    infer(args)