"""用 2 条故事完整演示 MiniStoryGPT 的一次端到端训练。

本文件故意复用 model.py 中的 GPTLanguageModel，而不是另外实现一套模型。
为使 CPU 上的教学演示几秒内完成，只在本进程中把模型缩小；不会修改正式训练参数。
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import tiktoken
import torch
from torch.nn import functional as F

import model as model_module


def section(number: int, title: str, purpose: str) -> None:
    print(f"\n{'=' * 88}\n[{number}] {title}\n算法作用：{purpose}\n{'=' * 88}")


def tensor_snapshot(name: str, tensor: torch.Tensor, rows: int = 2, cols: int = 8) -> None:
    detached = tensor.detach().cpu()
    memory_bytes = detached.numel() * detached.element_size()
    print(
        f"{name}: shape={tuple(detached.shape)}, dtype={detached.dtype}, "
        f"device={tensor.device}, memory={memory_bytes} bytes"
    )
    if detached.ndim == 0:
        print(f"  value={detached.item():.6f}")
    elif detached.ndim == 1:
        print(f"  values={detached[:cols].tolist()}")
    elif detached.ndim == 2:
        print(f"  values[:{rows}, :{cols}]=\n{detached[:rows, :cols]}")
    else:
        print(f"  values[0, :{rows}, :{cols}]=\n{detached[0, :rows, :cols]}")
    if detached.is_floating_point():
        print(
            f"  stats: min={detached.min():.5f}, max={detached.max():.5f}, "
            f"mean={detached.mean():.5f}, std={detached.std():.5f}"
        )


def decoded_tokens(ids: list[int], new_to_old: dict[int, int], enc) -> list[str]:
    return [repr(enc.decode([new_to_old[token_id]])) for token_id in ids]


def configure_small_model(block_size: int) -> None:
    """覆盖 model.py 的模块级参数，只影响当前教学进程。"""
    model_module.block_size = block_size
    model_module.n_embd = 32
    model_module.n_head = 4
    model_module.n_layer = 2
    model_module.dropout = 0.0
    model_module.use_sinusoidal = False
    model_module.device = "cpu"


def load_two_stories(path: Path) -> list[str]:
    records = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                raise ValueError(f"{path}:{line_number} 必须是包含字符串 text 字段的 JSON 对象")
            records.append(record["text"])
    if len(records) != 2:
        raise ValueError(f"{path} 必须恰好包含 2 条非空数据，实际为 {len(records)} 条")
    return records


def capture_forward(model: torch.nn.Module, xb: torch.Tensor, yb: torch.Tensor):
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def save_output(name):
        def hook(_module, _inputs, output):
            captured[name] = output.detach()

        return hook

    handles.append(model.token_embedding_table.register_forward_hook(save_output("token_embedding")))
    handles.append(model.position_embedding_table.register_forward_hook(save_output("position_embedding")))
    handles.append(model.blocks[0].ln1.register_forward_hook(save_output("block0_ln1")))
    handles.append(model.blocks[0].sa.register_forward_hook(save_output("block0_attention_output")))
    handles.append(model.blocks[0].ffwd.register_forward_hook(save_output("block0_ffn_output")))
    handles.append(model.ln_f.register_forward_hook(save_output("final_norm")))
    handles.append(model.lm_head.register_forward_hook(save_output("logits_3d")))

    logits, loss = model(xb, yb)
    for handle in handles:
        handle.remove()
    return logits, loss, captured


def main() -> None:
    parser = argparse.ArgumentParser(description="用两条数据追踪 MiniStoryGPT 完整训练流程")
    parser.add_argument("--steps", type=int, default=20, help="训练更新次数，默认 20")
    parser.add_argument("--block-size", type=int, default=16, help="每条样本使用的 token 数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).parent / "demo_two_stories.jsonl",
        help="恰好包含两条 {\"text\": ...} 记录的 JSONL 文件",
    )
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps 必须大于 0")
    if args.block_size < 4:
        parser.error("--block-size 必须至少为 4")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_printoptions(precision=4, sci_mode=False, linewidth=140)
    enc = tiktoken.get_encoding("gpt2")

    section(1, "加载 2 条原始数据", "模型最终学习的知识，最初只是内存中的两个字符串。")
    stories = load_two_stories(args.data)
    print(f"数据文件 = {args.data.resolve()}")
    print(f"内存对象 stories: type={type(stories).__name__}, len={len(stories)}")
    for index, story in enumerate(stories):
        print(f"story[{index}] ({len(story)} chars) = {story}")

    section(2, "GPT-2 BPE 分词", "把文本切成 token；神经网络只能计算数字，不能直接计算字符串。")
    original_token_rows = [enc.encode(story) for story in stories]
    for index, ids in enumerate(original_token_rows):
        print(f"story[{index}] GPT-2 ids ({len(ids)} tokens) = {ids}")
        print(f"story[{index}] token text = {[repr(enc.decode([token])) for token in ids]}")

    section(3, "建立小词表并重映射", "把稀疏的 GPT-2 id 压缩成从 0 开始的连续 id，以缩小 embedding 和输出层。")
    counts = Counter(token for row in original_token_rows for token in row)
    original_vocab = sorted(counts)
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(original_vocab)}
    new_to_old = {new_id: old_id for old_id, new_id in old_to_new.items()}
    remapped_rows = [[old_to_new[token] for token in row] for row in original_token_rows]
    print(f"vocab_size={len(original_vocab)}")
    print("映射前 12 项 (GPT-2 id -> 新 id -> token 文本):")
    for old_id in original_vocab[:12]:
        new_id = old_to_new[old_id]
        print(f"  {old_id:5d} -> {new_id:2d} -> {enc.decode([old_id])!r}")
    for index, row in enumerate(remapped_rows):
        print(f"story[{index}] remapped ids = {row}")

    if any(len(row) <= args.block_size for row in remapped_rows):
        raise ValueError("故事 token 数必须大于 block size，才能构造下一个 token 标签")

    section(4, "构造 batch：输入 x 与标签 y 错开一位", "位置 t 读取前文 x[t]，目标是预测真实的下一个 token y[t]。")
    xb = torch.tensor([row[: args.block_size] for row in remapped_rows], dtype=torch.long)
    yb = torch.tensor([row[1 : args.block_size + 1] for row in remapped_rows], dtype=torch.long)
    tensor_snapshot("xb", xb, rows=2, cols=args.block_size)
    tensor_snapshot("yb", yb, rows=2, cols=args.block_size)
    for batch_index in range(2):
        x_tokens = decoded_tokens(xb[batch_index].tolist(), new_to_old, enc)
        y_tokens = decoded_tokens(yb[batch_index].tolist(), new_to_old, enc)
        print(f"\nsample {batch_index} 的逐位置监督信号:")
        for position, (x_token, y_token) in enumerate(zip(x_tokens, y_tokens)):
            print(f"  pos {position:02d}: 看见 {x_token:<12} -> 应预测 {y_token}")

    section(5, "创建教学尺寸的原项目模型", "使用相同的 GPTLanguageModel 算法，小尺寸只为让两条数据在 CPU 上快速跑通。")
    configure_small_model(args.block_size)
    model = model_module.GPTLanguageModel(len(original_vocab))
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        "配置: block_size=%d, n_embd=%d, n_head=%d, n_layer=%d, parameters=%d"
        % (
            model_module.block_size,
            model_module.n_embd,
            model_module.n_head,
            model_module.n_layer,
            parameter_count,
        )
    )
    print("正式 model.py 默认配置未写回文件；教学脚本只在当前 Python 进程中临时缩小它。")

    section(6, "第一次前向传播：查看关键中间张量", "embedding 表示词与位置，Transformer 混合上下文，输出层给每个候选 token 打分。")
    model.train()
    logits, initial_loss, captured = capture_forward(model, xb, yb)
    tensor_snapshot("token embedding", captured["token_embedding"])
    tensor_snapshot("position embedding", captured["position_embedding"])
    tensor_snapshot("第 1 个 Block 的 LayerNorm 输出", captured["block0_ln1"])
    tensor_snapshot("第 1 个 Block 的多头注意力输出", captured["block0_attention_output"])
    tensor_snapshot("第 1 个 Block 的前馈网络输出", captured["block0_ffn_output"])
    tensor_snapshot("最终 LayerNorm 输出", captured["final_norm"])
    tensor_snapshot("logits（未归一化候选分数）", captured["logits_3d"])

    section(7, "查看第一个注意力头的因果注意力", "query 与 key 的相似度决定关注谁；上三角 mask 禁止偷看未来 token。")
    first_head = model.blocks[0].sa.heads[0]
    head_input = captured["block0_ln1"]
    keys = first_head.key(head_input)
    queries = first_head.query(head_input)
    scores = queries @ keys.transpose(-2, -1) * keys.shape[-1] ** -0.5
    masked_scores = scores.masked_fill(first_head.tril[: args.block_size, : args.block_size] == 0, float("-inf"))
    attention = F.softmax(masked_scores, dim=-1)
    print("sample 0 / head 0 / 前 8 个位置的注意力矩阵（每行和为 1）:")
    print(attention[0, :8, :8].detach())
    print("注意：对角线右侧全为 0，因此位置 t 只能读取位置 <= t 的信息。")

    section(8, "logits -> 概率 -> 交叉熵 loss", "softmax 把分数变成概率；loss 惩罚真实下一个 token 概率过低。")
    logits_3d = captured["logits_3d"]
    first_position_probs = F.softmax(logits_3d[0, 0], dim=-1)
    top_probs, top_ids = torch.topk(first_position_probs, k=min(5, len(original_vocab)))
    target_id = yb[0, 0].item()
    first_input_text = enc.decode([new_to_old[xb[0, 0].item()]])
    print(f"第一个位置看见 {first_input_text!r}，真实答案是 {enc.decode([new_to_old[target_id]])!r} (new_id={target_id})")
    print("随机初始化模型的 top-5 预测:")
    for probability, token_id in zip(top_probs.tolist(), top_ids.tolist()):
        print(f"  token={enc.decode([new_to_old[token_id]])!r:<12} new_id={token_id:2d} probability={probability:.4f}")
    print(f"真实答案概率={first_position_probs[target_id].item():.4f}")
    print(f"整个 batch 的 initial cross_entropy={initial_loss.item():.6f}")

    section(9, "反向传播与第一次参数更新", "autograd 计算每个参数怎样改变才能降低 loss，AdamW 据此更新权重。")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    tracked_parameter = model.token_embedding_table.weight
    before = tracked_parameter.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    initial_loss.backward()
    tensor_snapshot("token embedding 的 gradient", tracked_parameter.grad)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    delta = tracked_parameter.detach() - before
    print(f"全模型更新前 gradient norm={float(grad_norm):.6f}")
    tensor_snapshot("token embedding 参数变化 delta", delta)

    section(10, f"继续训练到 {args.steps} 步", "重复前向、算 loss、反向和更新，让两条训练数据的下一个 token 预测逐步变准。")
    losses = [initial_loss.item()]
    print(f"step 01/{args.steps}: loss={losses[-1]:.6f}")
    report_steps = {2, 3, 5, 10, args.steps}
    for step in range(2, args.steps + 1):
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())
        if step in report_steps:
            print(f"step {step:02d}/{args.steps}: loss={loss.item():.6f}")
    print(f"loss change: {losses[0]:.6f} -> {losses[-1]:.6f} ({losses[-1] - losses[0]:+.6f})")

    section(11, "训练后再预测并自回归生成", "训练用并行的 next-token 监督；生成时则一次采样一个 token，再把它接回上下文。")
    model.eval()
    with torch.no_grad():
        final_logits, final_loss = model(xb, yb)
        final_logits = final_logits.view(2, args.block_size, len(original_vocab))
        final_probs = F.softmax(final_logits[0, 0], dim=-1)
        top_probs, top_ids = torch.topk(final_probs, k=min(5, len(original_vocab)))
        print(f"训练后 batch loss={final_loss.item():.6f}")
        print(f"第一个位置真实答案概率: {first_position_probs[target_id].item():.4f} -> {final_probs[target_id].item():.4f}")
        print("训练后 top-5 预测:")
        for probability, token_id in zip(top_probs.tolist(), top_ids.tolist()):
            print(f"  token={enc.decode([new_to_old[token_id]])!r:<12} new_id={token_id:2d} probability={probability:.4f}")

        prompt = xb[:1, :4]
        generated = model.generate(prompt, max_new_tokens=12)[0].tolist()
        generated_text = enc.decode([new_to_old[token_id] for token_id in generated])
        print(f"生成起点 ids={prompt[0].tolist()}, text={enc.decode([new_to_old[i] for i in prompt[0].tolist()])!r}")
        print(f"生成结果 ids={generated}")
        print(f"生成结果 text={generated_text!r}")

    section(12, "流程结束", "两条字符串已完整经过数据准备、前向计算、损失、反向传播、更新和生成。")
    print("完整链路: text -> GPT-2 ids -> remapped ids -> (x,y) -> embeddings -> attention/FFN")
    print("          -> logits -> probabilities/loss -> gradients -> updated weights -> autoregressive generation")


if __name__ == "__main__":
    main()
