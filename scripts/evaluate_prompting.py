import json
from pathlib import Path

from cs336_alignment.drgrpo_grader import (
    question_only_reward_fn,
    r1_zero_reward_fn,
)
from cs336_alignment.vllm_utils import VLLMServer


MODEL_ID = "allenai/OLMo-2-0425-1B"
BATCH_SIZE = 16
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data/gsm8k/test.jsonl"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = RESULTS_DIR / "prompting_baselines.jsonl"
SUMMARY_PATH = RESULTS_DIR / "prompting_baselines_summary.json"

PROMPT_CONFIGS = {
    "question_only": (
        "question_only.prompt",
        question_only_reward_fn,
    ),
    "r1_zero": (
        "r1_zero.prompt",
        r1_zero_reward_fn,
    ),
    "r1_zero_three_shot": (
        "r1_zero_three_shot_gsm8k.prompt",
        r1_zero_reward_fn,
    ),
}


def main():
    examples = []

    with DATA_PATH.open("r", encoding="utf-8") as file:
        for question_id, line in enumerate(file):
            record = json.loads(line)

            if "####" not in record["answer"]:
                raise ValueError(
                    f"Question {question_id} has no answer delimiter"
                )

            examples.append({
                "question_id": question_id,
                "question": record["question"],
                "ground_truth": record["answer"].split("####")[-1].strip(),
            })

    if not examples:
        raise ValueError("The evaluation dataset is empty")

    server = VLLMServer(
        model_id=MODEL_ID,
        host="127.0.0.1",
        port=8000,
        launch_server=False,
    )
    server.start()

    summary = {}

    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        for name, (filename, reward_fn) in PROMPT_CONFIGS.items():
            template_path = PROJECT_ROOT / "cs336_alignment/prompts" / filename
            template = template_path.read_text(encoding="utf-8")

            sampling_params = {
                "temperature": 1.0,
                "top_p": 1.0,
                "max_tokens": 512,
                "n": 1,
                "seed": 0,
            }

            if name != "question_only":
                sampling_params["stop"] = ["</answer>"]
                sampling_params["include_stop_str_in_output"] = True

            counts = {
                "category_1": 0,
                "category_2": 0,
                "category_3": 0,
                "other": 0,
            }
            answer_total = 0.0
            length_total = 0

            for start in range(0, len(examples), BATCH_SIZE):
                batch = examples[start:start + BATCH_SIZE]
                prompts = [
                    template.format(question=example["question"])
                    for example in batch
                ]

                completions = server.generate_completions(
                    prompts=prompts,
                    sampling_params=sampling_params,
                )

                if len(completions) != len(batch):
                    raise RuntimeError("Response count does not match batch size")

                for example, prompt, completion in zip(
                    batch, prompts, completions
                ):
                    scores = reward_fn(
                        response=completion.text,
                        ground_truth=example["ground_truth"],
                    )

                    category = {
                        (1.0, 1.0): "category_1",
                        (1.0, 0.0): "category_2",
                        (0.0, 0.0): "category_3",
                    }.get(
                        (scores["format_reward"], scores["answer_reward"]),
                        "other",
                    )

                    response_length = len(completion.token_ids)
                    counts[category] += 1
                    answer_total += scores["answer_reward"]
                    length_total += response_length

                    result = {
                        **example,
                        "model_id": MODEL_ID,
                        "prompt_name": name,
                        "prompt": prompt,
                        "response": completion.text,
                        "sampling_params": sampling_params.copy(),
                        "scores": scores,
                        "category": category,
                        "finish_reason": completion.finish_reason,
                        "response_length": response_length,
                    }
                    output_file.write(
                        json.dumps(result, ensure_ascii=False) + "\n"
                    )

                output_file.flush()
                completed = start + len(batch)
                print(
                    f"{name}: {completed}/{len(examples)}",
                    flush=True,
                )

            summary[name] = {
                "num_examples": len(examples),
                "accuracy": answer_total / len(examples),
                "mean_response_length": length_total / len(examples),
                **counts,
            }

            print(
                json.dumps({name: summary[name]}, ensure_ascii=False, indent=2),
                flush=True,
            )

    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Responses saved to: {OUTPUT_PATH}")
    print(f"Summary saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()