from transformers import PreTrainedTokenizerBase
import torch
from torch import Tensor

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    prompt_ids = []
    output_ids = []
    masks = []
    for prompt, output in zip(prompt_strs, output_strs):
        prompt_id = torch.tensor(tokenizer.encode(prompt, add_special_tokens=False))
        output_id = torch.tensor(tokenizer.encode(output, add_special_tokens=False))
        mask = torch.cat((torch.zeros_like(prompt_id), torch.ones_like(output_id)), dim=0)
        masks.append(mask)
        prompt_ids.append(prompt_id)
        output_ids.append(output_id)
    # 将二者拼接
    full_prompt_output_ids = []
    full_len = []
    for i_id, o_id in zip(prompt_ids,output_ids):
        full_prompt_output_id = torch.cat((i_id, o_id), dim=0)
        full_prompt_output_ids.append(full_prompt_output_id)
        full_len.append(len(full_prompt_output_id))
    max_len = max(full_len)

    # 进行padding操作
    input_ids = []
    labels = []
    response_mask = []
    for full_prompt_output_id, mask in zip(full_prompt_output_ids, masks):
        pad_len = max_len - full_prompt_output_id.size(0)
        full_prompt_output_id = torch.nn.functional.pad(full_prompt_output_id,(0,pad_len),value=tokenizer.pad_token_id)
        mask = torch.nn.functional.pad(mask, (0,pad_len), value=0)
        input_id = full_prompt_output_id[:-1]
        label = full_prompt_output_id[1:]
        response_m = mask[1:]
        input_ids.append(input_id)
        labels.append(label)
        response_mask.append(response_m)

    input_ids = torch.stack(input_ids,dim=0)
    labels = torch.stack(labels,dim=0)
    response_mask = torch.stack(response_mask,dim=0)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask
    }

def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> dict[str, torch.Tensor | None]:
    """
        model: 用于打分的模型
        input_ids: torch.Tensor, 形状为 (batch_size, sequence_length):
            分词后的提示和输出。
        labels: torch.Tensor, 形状为 (batch_size, sequence_length):
            移位后的 input_ids。
        return_token_entropy: bool, 是否返回下一个 token 预测的熵。

    """
    outputs = model(input_ids = input_ids)

    all_log_probs = torch.nn.functional.log_softmax(
        outputs.logits, dim=-1
    )

    log_probs = all_log_probs.gather(
        dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)

    # 熵计算
    result = {"log_probs": log_probs}
    if(return_token_entropy):
        probs = all_log_probs.exp()
        result["token_entropy"] = - (probs * all_log_probs).sum(dim=-1)

    return result
