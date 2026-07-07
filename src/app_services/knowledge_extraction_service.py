from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate


class KnowledgeExtractionStatus(StrEnum):
    READY = "ready"
    EMPTY = "empty"
    FAILED = "failed"


@dataclass(frozen=True)
class KnowledgeExtractionResult:
    status: KnowledgeExtractionStatus
    extracted_knowledge: list[dict]
    conversation_summary: str
    user_intent: str
    checked_turns: int
    message: str


# ── LCEL chain ────────────────────────────────────────────────────────────────

_EXTRACTION_PROMPT = PromptTemplate.from_template(
    "你是专业的知识提取专家。请从以下对话中提取有价值的知识点。\n\n"
    "对话内容：\n{conversation}\n\n"
    "请严格按以下 JSON 格式返回，不要包含任何额外文字：\n"
    "{{\n"
    '  "extracted_knowledge": [\n'
    "    {{\n"
    '      "knowledge_type": "<事实|需求|流程|注意>",\n'
    '      "content": "<知识内容>",\n'
    '      "confidence": <0到1之间的小数>,\n'
    '      "keywords": ["<关键词1>", "<关键词2>"],\n'
    '      "category": "<分类>"\n'
    "    }}\n"
    "  ],\n"
    '  "conversation_summary": "<对话摘要>",\n'
    '  "user_intent": "<用户意图>"\n'
    "}}"
)


def build_extraction_chain(llm: Any):
    """LCEL chain: {conversation} → parsed knowledge dict."""
    return _EXTRACTION_PROMPT | llm | JsonOutputParser()


# ── Service function ──────────────────────────────────────────────────────────

def extract_knowledge_from_conversation(
    conversation: list[dict],
    llm: Any,
) -> KnowledgeExtractionResult:
    if not conversation:
        return KnowledgeExtractionResult(
            status=KnowledgeExtractionStatus.EMPTY,
            extracted_knowledge=[],
            conversation_summary="",
            user_intent="",
            checked_turns=0,
            message="对话为空，无法提取知识。",
        )

    conversation_text = "\n".join(
        f"{'用户' if m['role'] == 'user' else 'AI'}：{m['content']}"
        for m in conversation
    )

    try:
        chain = build_extraction_chain(llm)
        parsed = chain.invoke({"conversation": conversation_text})

        return KnowledgeExtractionResult(
            status=KnowledgeExtractionStatus.READY,
            extracted_knowledge=parsed.get("extracted_knowledge", []),
            conversation_summary=parsed.get("conversation_summary", ""),
            user_intent=parsed.get("user_intent", ""),
            checked_turns=len(conversation),
            message=f"提取完成，共发现 {len(parsed.get('extracted_knowledge', []))} 个知识点。",
        )
    except Exception as exc:
        return KnowledgeExtractionResult(
            status=KnowledgeExtractionStatus.FAILED,
            extracted_knowledge=[],
            conversation_summary="",
            user_intent="",
            checked_turns=len(conversation),
            message=f"知识提取失败：{type(exc).__name__}",
        )
