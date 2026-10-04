"""
游戏内NPC记忆与对话历史系统 - 核心模块

功能：
- 记忆存储：重要性评分 + 时间衰减 + 遗忘
- 记忆检索：上下文相关 + 标签关联 + 时间衰减排序
- 对话历史：完整记录 + 上下文 + 长历史摘要
- 情感关系：好感度/信任度/恐惧度多维度 + 行为驱动变化
- 性格演变：经历驱动 + 稳定性约束
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math
import re


# 情感关键词，用于自动重要性评分
_IMPORTANT_KEYWORDS = {
    "救命": 0.4, "救": 0.3, "背叛": 0.4, "礼物": 0.3, "攻击": 0.35,
    "威胁": 0.3, "朋友": 0.2, "敌人": 0.3, "帮助": 0.25, "偷": 0.3,
    "死": 0.35, "爱": 0.3, "恨": 0.3, "秘密": 0.3, "承诺": 0.25,
}


@dataclass
class Memory:
    """一条记忆。"""
    id: int
    content: str
    importance: float
    tags: frozenset = frozenset()
    created_at: float = 0.0
    access_count: int = 0

    def strength(self, now: float, decay_rate: float) -> float:
        """当前强度 = 重要性 * 时间衰减，访问次数提供少量强化。"""
        age = max(0.0, now - self.created_at)
        reinforcement = 1.0 + 0.1 * min(self.access_count, 10)
        return self.importance * reinforcement * math.exp(-decay_rate * age)


@dataclass
class DialogueTurn:
    """一轮对话。"""
    speaker: str
    text: str
    time: float


class NPCMemory:
    """NPC记忆与对话历史系统。"""

    DEFAULT_TRAITS = ("friendly", "brave", "cautious", "curious")

    # 互动类型 -> (好感, 信任, 恐惧) 增量
    _RELATION_EFFECTS = {
        "help": (0.10, 0.08, 0.0),
        "gift": (0.12, 0.05, 0.0),
        "kind_word": (0.05, 0.03, 0.0),
        "betray": (-0.20, -0.30, 0.10),
        "attack": (-0.25, -0.20, 0.30),
        "threat": (-0.10, -0.10, 0.25),
        "steal": (-0.15, -0.25, 0.05),
    }

    # 经历类型 -> 性格增量
    _EXPERIENCE_EFFECTS = {
        "combat_victory": {"brave": 0.10, "cautious": -0.05},
        "combat_defeat": {"brave": -0.10, "cautious": 0.10},
        "betrayed": {"friendly": -0.10, "cautious": 0.08},
        "helped_by_other": {"friendly": 0.08},
        "explored_new_place": {"curious": 0.08},
        "danger": {"cautious": 0.08, "brave": -0.04},
    }

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.decay_rate = float(self.config.get("decay_rate", 0.05))
        self.forget_threshold = float(self.config.get("forget_threshold", 0.05))
        self.summary_threshold = int(self.config.get("summary_threshold", 10))
        self.stability = float(self.config.get("stability", 0.8))
        self.reset()

    # ------------------------------------------------------------------
    # 基础
    # ------------------------------------------------------------------
    def update(self, dt: float):
        """推进内部时间，并遗忘强度低于阈值的记忆。"""
        self._time += max(0.0, dt)
        self._memories = {
            mid: m for mid, m in self._memories.items()
            if m.strength(self._time, self.decay_rate) >= self.forget_threshold
        }

    def reset(self):
        self._time = 0.0
        self._next_id = 1
        self._memories: Dict[int, Memory] = {}
        self._history: List[DialogueTurn] = []
        self._relationship = {"affection": 0.0, "trust": 0.0, "fear": 0.0}
        self._traits = {t: 0.5 for t in self.DEFAULT_TRAITS}
        self._experiences: List[str] = []

    # ------------------------------------------------------------------
    # 记忆存储：重要性评分 + 衰减
    # ------------------------------------------------------------------
    def score_importance(self, content: str) -> float:
        """根据内容自动评估重要性（0~1）。"""
        score = 0.1
        for kw, weight in _IMPORTANT_KEYWORDS.items():
            if kw in content:
                score += weight
        score += min(0.2, len(content) / 500.0)
        return max(0.0, min(1.0, score))

    def store_memory(self, content: str, importance: Optional[float] = None,
                     tags: Optional[List[str]] = None) -> int:
        """存储一条记忆，返回记忆ID。未指定重要性时自动评分。"""
        if importance is None:
            importance = self.score_importance(content)
        importance = max(0.0, min(1.0, float(importance)))
        mid = self._next_id
        self._next_id += 1
        self._memories[mid] = Memory(
            id=mid, content=content, importance=importance,
            tags=frozenset(tags or ()), created_at=self._time,
        )
        return mid

    # ------------------------------------------------------------------
    # 记忆检索：上下文相关 + 关联 + 时间衰减
    # ------------------------------------------------------------------
    @staticmethod
    def _tokens(text: str) -> set:
        return set(re.findall(r"[A-Za-z0-9_]+|[一-鿿]", text.lower()))

    def _relevance(self, memory: Memory, query_tokens: set) -> float:
        if not query_tokens:
            return 1.0
        content_tokens = self._tokens(memory.content) | set(memory.tags)
        overlap = len(query_tokens & content_tokens)
        return overlap / math.sqrt(len(query_tokens) * max(1, len(content_tokens)))

    def retrieve_memory(self, query: str = "", limit: int = 5) -> List[Memory]:
        """按 相关性 × 时间衰减强度 检索记忆，并做标签关联扩展。"""
        query_tokens = self._tokens(query)
        scored = []
        for m in self._memories.values():
            rel = self._relevance(m, query_tokens)
            # 关联：与高分记忆共享标签的记忆获得加成
            score = rel * m.strength(self._time, self.decay_rate)
            scored.append((score, rel, m))
        scored.sort(key=lambda x: (-x[0], -x[1], x[2].id))
        # 关联扩展：与已命中记忆共享标签的记忆提升得分
        top_tags = set()
        for score, rel, m in scored[:limit]:
            if score > 0:
                top_tags |= set(m.tags)
        if top_tags:
            boosted = []
            for score, rel, m in scored:
                if set(m.tags) & top_tags:
                    score = max(score * 1.5, 0.5 * m.strength(self._time, self.decay_rate))
                boosted.append((score, rel, m))
            boosted.sort(key=lambda x: (-x[0], -x[1], x[2].id))
            scored = boosted
        results = [m for score, rel, m in scored if score > 0 or not query_tokens][:limit]
        for m in results:
            m.access_count += 1
        return results

    # ------------------------------------------------------------------
    # 对话历史：完整记录 + 上下文 + 摘要
    # ------------------------------------------------------------------
    def add_dialogue(self, speaker: str, text: str):
        self._history.append(DialogueTurn(speaker=speaker, text=text, time=self._time))

    def get_dialogue_history(self, limit: Optional[int] = None) -> List[DialogueTurn]:
        """返回对话历史（默认全部，不只是最后一条）。"""
        if limit is None:
            return list(self._history)
        return list(self._history[-limit:])

    def summarize_history(self, turns: Optional[List[DialogueTurn]] = None) -> str:
        """将对话历史压缩为摘要：说话者、轮次、关键话题。"""
        turns = self._history if turns is None else turns
        if not turns:
            return ""
        speakers: Dict[str, int] = {}
        keywords: Dict[str, int] = {}
        for t in turns:
            speakers[t.speaker] = speakers.get(t.speaker, 0) + 1
            for kw in _IMPORTANT_KEYWORDS:
                if kw in t.text:
                    keywords[kw] = keywords.get(kw, 0) + 1
        speaker_part = "、".join(f"{s}({n}次)" for s, n in speakers.items())
        topic_part = "、".join(sorted(keywords, key=keywords.get, reverse=True)[:5])
        summary = f"共{len(turns)}轮对话，参与者：{speaker_part}"
        if topic_part:
            summary += f"，话题涉及：{topic_part}"
        return summary

    def get_context(self, recent: int = 5) -> Dict:
        """返回对话上下文：长历史自动摘要 + 最近几轮原文。"""
        if len(self._history) > self.summary_threshold:
            old = self._history[:-recent]
            return {
                "summary": self.summarize_history(old),
                "recent": list(self._history[-recent:]),
            }
        return {"summary": "", "recent": list(self._history)}

    # ------------------------------------------------------------------
    # 情感关系：多维度 + 行为驱动变化
    # ------------------------------------------------------------------
    def get_relationship(self) -> Dict[str, float]:
        return dict(self._relationship)

    def record_interaction(self, kind: str, intensity: float = 1.0):
        """记录一次玩家互动，更新好感/信任/恐惧三个维度。"""
        effects = self._RELATION_EFFECTS.get(kind)
        if effects is None:
            return
        intensity = max(0.0, float(intensity))
        for dim, delta in zip(("affection", "trust", "fear"), effects):
            value = self._relationship[dim] + delta * intensity
            self._relationship[dim] = max(-1.0, min(1.0, value))

    # ------------------------------------------------------------------
    # 性格演变：经历驱动 + 稳定性约束
    # ------------------------------------------------------------------
    def get_personality(self) -> Dict[str, float]:
        return dict(self._traits)

    def record_experience(self, kind: str, intensity: float = 1.0):
        """记录一次经历，性格按 (1-稳定性) 的速率缓慢演变。"""
        effects = self._EXPERIENCE_EFFECTS.get(kind)
        if effects is None:
            return
        self._experiences.append(kind)
        rate = (1.0 - self.stability) * max(0.0, float(intensity))
        for trait, delta in effects.items():
            if trait in self._traits:
                value = self._traits[trait] + delta * rate
                self._traits[trait] = max(0.0, min(1.0, value))
