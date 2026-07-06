from src.generation.answer_generator import AnswerGenerationInput
from src.generation.prompt_builder import build_grounded_answer_messages


def test_build_grounded_answer_messages_includes_question_and_contexts():
    messages = build_grounded_answer_messages(
        AnswerGenerationInput(
            question="客户经理投诉会影响评聘吗？",
            contexts=[
                {
                    "chunk_id": "chunk-1",
                    "content": "客户经理被投诉一次会影响评聘。",
                    "source_file": "policy-a.pdf",
                    "page_number": 2,
                    "score": 0.6,
                }
            ],
        )
    )

    assert messages[0] == {
        "role": "system",
        "content": "你是银行内部制度问答助手。只能依据给定证据回答；证据不足时必须说明无法确定。",
    }
    assert "问题：客户经理投诉会影响评聘吗？" in messages[1]["content"]
    assert "[1] policy-a.pdf 第 2 页" in messages[1]["content"]
    assert "客户经理被投诉一次会影响评聘。" in messages[1]["content"]


def test_build_grounded_answer_messages_asks_for_concise_cited_answer():
    messages = build_grounded_answer_messages(
        AnswerGenerationInput(question="怎么处理投诉？", contexts=[])
    )

    assert "请用中文简洁回答" in messages[1]["content"]
    assert "不要编造未给出的制度条款" in messages[1]["content"]