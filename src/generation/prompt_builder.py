from __future__ import annotations

from src.generation.answer_generator import AnswerGenerationInput


SYSTEM_PROMPT = "你是银行内部制度问答助手。只能依据给定证据回答；证据不足时必须说明无法确定。"


def build_grounded_answer_messages(generation_input: AnswerGenerationInput) -> list[dict[str, str]]:
    evidence_lines = []
    for index, context in enumerate(generation_input.contexts, start=1):
        evidence_lines.append(
            f"[{index}] {context['source_file']} 第 {context['page_number']} 页\n"
            f"{context['content']}"
        )

    evidence_text = "\n\n".join(evidence_lines) if evidence_lines else "无可用证据。"
    user_prompt = (
        f"问题：{generation_input.question}\n\n"
        f"证据：\n{evidence_text}\n\n"
        "请用中文简洁回答，并只依据上方证据。不要编造未给出的制度条款。"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]