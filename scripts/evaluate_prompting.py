import json
from pathlib import Path
from cs336_alignment.drgrpo_grader import question_only_reward_fn,r1_zero_reward_fn 
with open("data/gsm8k/test.jsonl", "r", encoding="utf-8") as file:
    first_line = file.readline()
record = json.loads(first_line)

question = record["question"]
ground_truth = record["answer"].split("####")[-1].strip()

print("Question: ",question)
print("Ground truth: ",ground_truth)

prompt_paths = {
    "question_only": "cs336_alignment/prompts/question_only.prompt",
    "r1_zero": "cs336_alignment/prompts/r1_zero.prompt",
    "r1_zero_three_shot":"cs336_alignment/prompts/r1_zero_three_shot_gsm8k.prompt"
}

for name,path in prompt_paths.items():
    template = Path(path).read_text(encoding="utf-8")
    prompt = template.format(question=question)

    print(f"\n======== {name} ========")
    print(prompt)

    sampling_params = {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 512,
        "n": 1,
        "seed": 0
    }

    if name in ("r1_zero", "r1_zero_three_shot"):
        sampling_params["stop"] = ["</answer>"]
        sampling_params["include_stop_str_in_output"] = True

    print(f"==== {name} sampling params ====")
    print(sampling_params)

    # 为三种提示词选择对应的评分函数
    if name == "question_only":
        reward_fn = question_only_reward_fn
    else:
        reward_fn = r1_zero_reward_fn
    print(f"{name}: {reward_fn.__name__}")