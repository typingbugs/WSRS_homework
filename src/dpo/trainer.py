from trl import DPOTrainer
import torch


class DPORankingTrainer(DPOTrainer):
    """
    自定义 eval loop，返回最后 token logits 以计算 MRR
    """
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        model.eval()
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["prompt_input_ids"],
                attention_mask=inputs["prompt_attention_mask"],
                return_dict=True,
            )
        # 取 prompt 中最后一个非 pad 位置的 logits（注意 prompt 是左 padding 的）
        attn_mask = inputs["prompt_attention_mask"]
        last_indices = attn_mask.size(1) - attn_mask.flip(dims=[1]).argmax(dim=1) - 1
        batch_indices = torch.arange(last_indices.size(0), device=outputs.logits.device)
        logits = outputs.logits[batch_indices, last_indices, :].contiguous()

        # 只用正例序列的首个 token 作为目标 item id
        labels = inputs["chosen_input_ids"][:, 0].to(logits.device)
        loss = torch.tensor(0.0, device=logits.device)
        return (loss, logits, labels)