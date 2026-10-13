from typing import Callable,Literal
import torch



def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    参数 (Args):
        reward_fn: Callable[[str, str], dict[str, float]], 
            将采样回答与标准答案（ground truths）进行比对评分，
            生成一个包含 "reward"（总奖励）、"format_reward"（格式奖励）
            和 "answer_reward"（答案准确奖励）键的字典。
        rollout_responses: list[str], 策略模型生成的采样回答列表。
            该列表的长度为 `rollout_batch_size = n_prompts_per_rollout_batch * group_size`。
        repeated_ground_truths: list[str], 样本的标准答案列表。
            该列表的长度为 `rollout_batch_size`，因为每个样本的标准答案
            都被重复了 `group_size` 次。
    """
    if len(rollout_responses) != len(repeated_ground_truths):
        raise ValueError("Responses and ground truths must have equal lengths.")
    if not rollout_responses:
        raise ValueError("The rollout batch must not be empty.")
    
    rewards =[]
    format_rewards = []

    # 从模型生成的回答以及标准答案中计算得分
    for response, ground_truth in zip(
        rollout_responses, repeated_ground_truths
    ):
        scores = reward_fn(response, ground_truth)
        rewards.append(scores["reward"])
        format_rewards.append(scores["format_reward"])

    raw_rewards = torch.tensor(rewards, dtype=torch.float32)

    metadata = {
        "mean_reward": float(sum(rewards) / len(rewards)),
        "mean_format_reward": float(
            sum(format_rewards) / len(format_rewards)
        ),
    }

    return raw_rewards,metadata

def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std"
):
    if baseline != "mean" or advantage_normalizer != "std":
        raise NotImplementedError("Only standard GRPO is suportted.")

    if group_size < 2:
        raise ValueError("Sample standard deviation requires group_size >= 2.")

    # 计算组内均值，沿着每行求均值
    grouped_rewards = raw_rewards.reshape(-1, group_size)

    group_means = grouped_rewards.mean(dim=1, keepdim=True)
    group_stds = grouped_rewards.std(dim=1, keepdim=True)

    advantages = (
        (grouped_rewards - group_means) / 
        (group_stds + advantage_eps)
    )

    metadata = {
        "mean_group_std": group_stds.mean().item(),
    }

    return advantages.reshape(-1), metadata