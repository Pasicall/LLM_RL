from typing import Callable,Literal
import torch
from transformers import PreTrainedTokenizerBase
from cs336_alignment.tokenization import tokenize_prompt_and_output


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
    #计算每道题生成回答的优势a 

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

def compute_policy_gradient_loss_on_policy(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    raw_rewards_or_advantages: 计算出来的优势A
    policy_log_probs: 每个token的对数概率
    """

    if importance_reweighting_method != "none":
        raise NotImplementedError("Only on-policy loss is supported!")
    
    weights = raw_rewards_or_advantages.reshape(-1,1)
    # 计算每个token的损失
    per_token_loss = - weights * policy_log_probs

    metadata = {}
    return per_token_loss,metadata

def aggregate_loss_across_microbatch_sequence(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:

    if loss_normalization != "sequence":
        raise NotImplementedError(f"only suport sequence mean!")
    
    # 计算mask之后的损失
    masked_losses = per_token_policy_gradient_loss * mask

    # 计算每个回答的loss总和、有效长度、平均损失
    per_response_loss = masked_losses.sum(dim=1)
    per_response_len =  mask.sum(dim=1)
    if (per_response_len < 1).any():
        raise ValueError(f"Each response should contain at lest one token!")
    else:
        per_response_avg_loss = per_response_loss / per_response_len

    # 计算批次平均损失
    batch_avg_loss = per_response_avg_loss.mean()

    return batch_avg_loss

def grpo_train_step(
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        optimizer: torch.optim.Optimizer,
        gradient_accumulation_steps: int,
        max_grad_norm: float | None,
        reward_fn: Callable[[str, str], dict[str, float]],
        repeated_prompts: list[str],
        rollout_responses: list[str],
        repeated_ground_truths: list[str],
        group_size: int,
        baseline: Literal["mean", "none"] = "mean",
        advantage_eps: float = 1e-6,
        advantage_normalizer: Literal["std", "none", "mean"] = "std",
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
        old_log_probs: torch.Tensor | None = None,
        cliprange: float | None = None,
        loss_normalization: Literal["sequence", "constant"] = "sequence",
        normalization_constant: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:

    # 1. 检查输入
    if (
        baseline != "mean"
        or advantage_normalizer != "std"
        or importance_reweighting_method != "none"
        or loss_normalization != "sequence"
    ):
        raise NotImplementedError(
            "Only standard on-policy GRPO with sequence normalization is supported."
        )

    batch_size = len(rollout_responses)
    if batch_size == 0:
        raise ValueError("The rollout batch must not be empty.")

    if not (
        len(repeated_prompts)
        == batch_size
        == len(repeated_ground_truths)
    ):
        raise ValueError(
            "Prompts, responses, and ground truths must have equal lengths."
        )

    if group_size < 2:
        raise ValueError("Standard GRPO requires group_size >= 2.")

    if batch_size % group_size != 0:
        raise ValueError("Batch size must be divisible by group_size.")

    if gradient_accumulation_steps < 1:
        raise ValueError("Gradient accumulation steps must be positive.")

    if not advantage_eps > 0 :
        raise ValueError("Advantage_eps must > 0")

    # 2. 准备模型输入
    tokenized = tokenize_prompt_and_output(
        prompt_strs = repeated_prompts,
        output_strs = rollout_responses,
        tokenizer = tokenizer
    )
    input_ids = tokenized["input_ids"]
    labels = tokenized["labels"]
    response_mask = tokenized["response_mask"]

    # 3.在完整的批次上计算奖励与优势
    raw_rewards, reward_metadata = compute_rollout_rewards(
        reward_fn = reward_fn,
        rollout_responses = rollout_responses,
        repeated_ground_truths = repeated_ground_truths
    )

    advantages, advantage_metadata = compute_group_normalized_rewards(
        raw_rewards = raw_rewards,
        group_size = group_size,
        baseline = baseline,
        advantage_eps = advantage_eps,
        advantage_normalizer = advantage_normalizer
    )

    # 4.将计算需要的张量移动到模型所在的设备
    device = next(model.parameters()).device

    input_ids = input_ids.to(device)
    labels = labels.to(device)
    response_mask = response_mask.to(device)
    advantages = advantages.to(device)

    # 5. 将数据切分为微批次
    K = gradient_accumulation_steps
    B = batch_size

    if K > B:
        raise ValueError(" Gradient accumulation steps must not exceed batch size.")

    # 将四个张量按照同样规格划分
    input_batches = input_ids.tensor_split(K, dim=0)
    label_batches = labels.tensor_split(K, dim=0)
    mask_batches = response_mask.tensor_split(K, dim=0)
    advantage_batches = advantages.tensor_split(K, dim=0)

    for cur_input_ids, cur_labels, cur_response_mask, cur_advantages in zip(
        input_batches, label_batches, mask_batches, advantage_batches
        ):
        microbatch_size = len(cur_input_ids)
        weight = microbatch_size / B

        assert (
        cur_input_ids.shape[0]
        == cur_labels.shape[0]
        == cur_response_mask.shape[0]
        == cur_advantages.shape[0]
)

