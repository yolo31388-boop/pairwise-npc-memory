"""
游戏内NPC记忆与对话历史系统 - 核心模块

功能：
- 记忆存储：重要性评分 + 时间衰减 + 旧记忆遗忘
- 记忆检索：上下文相关 + 标签关联扩展 + 时间衰减排序
- 对话历史：完整记录 + 上下文窗口 + 长历史摘要
- 情感关系：好感度/信任度/恐惧度多维度 + 行为驱动变化
- 性格演变：经历驱动 + 稳定性约束（缓慢变化）
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

_RELATION_DIMENSIONS = ("affection", "trust", "fear")
_DEFAULT_TRAITS = ("friendly", "brave", "cautious", "curious")


@dataclass
class Config:
    """系统配置（也可直接传 dict 给 NPCMemory）。"""
    decay_rate: float = 0.05
    forget_threshold: float = 0.05
    summary_threshold: int = 10
    stability: float = 0.8


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
        """当前强度 = 重要性 * 时间衰减，反复回忆会小幅强化记忆。"""
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

    DEFAULT_TRAITS = _DEFAULT_TRAITS
    _RELATION_EFFECTS = _RELATION_EFFECTS
    _EXPERIENCE_EFFECTS = _EXPERIENCE_EFFECTS

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.decay_rate = float(self.config.get("decay_rate", Config.decay_rate))
        self.forget_threshold = float(
            self.config.get("forget_threshold", Config.forget_threshold))
        self.summary_threshold = int(
            self.config.get("summary_threshold", Config.summary_threshold))
        self.stability = float(self.config.get("stability", Config.stability))
        self.reset()

    # ------------------------------------------------------------------
    # 基础
    # ------------------------------------------------------------------
    def update(self, dt: float):
        """推进内部时间，并遗忘衰减后强度低于阈值的记忆。"""
        self._time += max(0.0, float(dt))
        self._memories = {
            mid: m for mid, m in self._memories.items()
            if m.strength(self._time, self.decay_rate) >= self.forget_threshold
        }

    def reset(self):
        self._time = 0.0
        self._next_id = 1
        self._memories: Dict[int, Memory] = {}
        self._history: List[DialogueTurn] = []
        self._relationship: Dict[str, float] = {
            dim: 0.0 for dim in _RELATION_DIMENSIONS}
        self._traits: Dict[str, float] = {t: 0.5 for t in _DEFAULT_TRAITS}
        self._experiences: List[str] = []

    # ------------------------------------------------------------------
    # 记忆存储：重要性评分 + 衰减
    # ------------------------------------------------------------------
    def score_importance(self, content: str) -> float:
        """根据内容自动评估重要性（0~1）：情感关键词越多越重要。"""
        score = 0.1
        for kw, weight in _IMPORTANT_KEYWORDS.items():
            if kw in content:
                score += weight
        score += min(0.2, len(content) / 500.0)
        return max(0.0, min(1.0, score))

    def store_memory(self, content: str, importance: Optional[float] = None,
                     tags: Optional[List[str]] = None) -> int:
        """存储一条记忆并返回记忆ID；未指定重要性时自动评分。"""
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
        return set(re.findall(r"[A-Za-z0-9_]+|[一-鿿]", str(text).lower()))

    def _relevance(self, memory: Memory, query_tokens: set) -> float:
        if not query_tokens:
            return 1.0
        content_tokens = self._tokens(memory.content) | set(memory.tags)
        overlap = len(query_tokens & content_tokens)
        return overlap / math.sqrt(len(query_tokens) * max(1, len(content_tokens)))

    def retrieve_memory(self, query: str = "", limit: int = 5) -> List[Memory]:
        """
        按 相关性 x 当前强度（重要性 x 时间衰减）排序检索，
        并通过共享标签做关联扩展：与直接命中记忆共享标签的记忆也会被带出。
        """
        query_tokens = self._tokens(query)

        scored = []
        for m in self._memories.values():
            rel = self._relevance(m, query_tokens)
            base = rel * m.strength(self._time, self.decay_rate)
            scored.append([base, rel, m, 0.0])  # 最后一项为关联分

        # 直接命中的记忆作为关联种子
        seeds = [item for item in scored if item[1] > 0.0]
        if seeds:
            seed_tags = set()
            for _, _, m, _ in seeds:
                seed_tags |= set(m.tags)
            anchor = max(item[0] for item in seeds)
            if seed_tags and anchor > 0:
                for item in scored:
                    base, rel, m, _ = item
                    if rel > 0.0:
                        continue
                    shared = set(m.tags) & seed_tags
                    if shared:
                        # 零文本相关、但共享标签：赋予正向关联分（而非乘以0）
                        item[3] = anchor * 0.5 * (
                            len(shared) / max(1, len(m.tags)))

        scored.sort(key=lambda x: (-(x[0] + x[3]), -x[1], x[2].id))
        results = [
            m for base, rel, m, assoc in scored
            if (base + assoc) > 0.0 or not query_tokens
        ][:limit]
        for m in results:
            m.access_count += 1
        return results

    # ------------------------------------------------------------------
    # 对话历史：完整记录 + 上下文 + 摘要
    # ------------------------------------------------------------------
    def add_dialogue(self, speaker: str, text: str) -> DialogueTurn:
        """记录一轮对话（追加，不再只保留最后一条）。"""
        turn = DialogueTurn(speaker=speaker, text=text, time=self._time)
        self._history.append(turn)
        return turn

    def get_dialogue_history(self, limit: Optional[int] = None) -> List[DialogueTurn]:
        """返回对话历史（默认返回全部，按时间顺序）。"""
        if limit is None:
            return list(self._history)
        return list(self._history[-limit:])

    def summarize_history(self, turns: Optional[List[DialogueTurn]] = None) -> str:
        """将对话历史压缩为摘要：参与说话者、轮次数、高频话题。"""
        turns = self._history if turns is None else turns
        if not turns:
            return ""
        speakers: Dict[str, int] = {}
        keywords: Dict[str, int] = {}
        for turn in turns:
            speakers[turn.speaker] = speakers.get(turn.speaker, 0) + 1
            for kw in _IMPORTANT_KEYWORDS:
                if kw in turn.text:
                    keywords[kw] = keywords.get(kw, 0) + 1
        speaker_part = "、".join(f"{s}({n}次)" for s, n in speakers.items())
        summary = f"共{len(turns)}轮对话，参与者：{speaker_part}"
        if keywords:
            topic_part = "、".join(
                sorted(keywords, key=keywords.get, reverse=True)[:5])
            summary += f"，话题涉及：{topic_part}"
        return summary

    def get_context(self, recent: int = 5) -> Dict:
        """返回对话上下文：超长历史给出摘要，外加最近几轮原文。"""
        if len(self._history) > self.summary_threshold:
            old_turns = self._history[:-recent] if recent > 0 else self._history
            return {
                "summary": self.summarize_history(old_turns),
                "recent": list(self._history[-recent:]) if recent > 0 else [],
            }
        return {"summary": "", "recent": list(self._history)}

    # ------------------------------------------------------------------
    # 情感关系：多维度 + 行为驱动变化
    # ------------------------------------------------------------------
    def get_relationship(self, entity: Optional[str] = None) -> Dict[str, float]:
        """返回好感度/信任度/恐惧度三个维度（取值 -1~1）。"""
        return dict(self._relationship)

    def record_interaction(self, kind: str, intensity: float = 1.0,
                           entity: Optional[str] = None):
        """记录一次玩家互动，按行为类型更新好感/信任/恐惧。"""
        effects = _RELATION_EFFECTS.get(kind)
        if effects is None:
            return
        intensity = max(0.0, float(intensity))
        for dim, delta in zip(_RELATION_DIMENSIONS, effects):
            value = self._relationship[dim] + delta * intensity
            self._relationship[dim] = max(-1.0, min(1.0, value))

    # ------------------------------------------------------------------
    # 性格演变：经历驱动 + 稳定性
    # ------------------------------------------------------------------
    def get_personality(self) -> Dict[str, float]:
        """返回当前性格特质（0~1，初始 0.5）。"""
        return dict(self._traits)

    def record_experience(self, kind: str, intensity: float = 1.0):
        """
        记录一次经历并推动性格演变。
        单次变化量 = 经历影响 x (1 - 稳定性)：稳定性越高，性格改变越慢。
        """
        effects = _EXPERIENCE_EFFECTS.get(kind)
        if effects is None:
            return
        self._experiences.append(kind)
        rate = (1.0 - self.stability) * max(0.0, float(intensity))
        for trait, delta in effects.items():
            if trait in self._traits:
                value = self._traits[trait] + delta * rate
                self._traits[trait] = max(0.0, min(1.0, value))
