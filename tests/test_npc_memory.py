"""
游戏内NPC记忆与对话历史系统 - 验收测试
运行方式：python -m pytest tests/test_npc_memory.py -q
共 15 个测试用例
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.npc_memory import NPCMemory


class TestNPCMemory:
    def setup_method(self):
        self.system = NPCMemory()

    # ---- 记忆存储：重要性评分 + 衰减 ----

    def test_case_01(self):
        """存储的记忆可以被取回，不会对话过就忘"""
        mid = self.system.store_memory("玩家救了我的命", tags=["player", "rescue"])
        results = self.system.retrieve_memory("救命")
        assert any(m.id == mid for m in results)

    def test_case_02(self):
        """重要性评分：重要内容得分高于琐事"""
        important = self.system.score_importance("玩家从怪物手中救了我的命")
        trivial = self.system.score_importance("今天天气不错")
        assert important > trivial
        assert 0.0 <= trivial <= 1.0
        assert 0.0 <= important <= 1.0

    def test_case_03(self):
        """记忆随时间衰减：同样的记忆，越久强度越低"""
        mid = self.system.store_memory("玩家送了我一份礼物", importance=0.8)
        m = self.system._memories[mid]
        before = m.strength(self.system._time, self.system.decay_rate)
        self.system.update(10.0)
        after = m.strength(self.system._time, self.system.decay_rate)
        assert after < before

    def test_case_04(self):
        """不重要的旧记忆会被遗忘，重要的记忆保留更久"""
        weak = self.system.store_memory("今天天气不错", importance=0.06)
        strong = self.system.store_memory("玩家救了我的命", importance=0.95)
        self.system.update(50.0)
        assert weak not in self.system._memories
        assert strong in self.system._memories

    # ---- 记忆检索：上下文相关 + 关联 + 时间衰减 ----

    def test_case_05(self):
        """检索能根据上下文找到相关记忆"""
        self.system.store_memory("玩家在森林里救了我", tags=["forest"])
        self.system.store_memory("铁匠铺今天打折", tags=["shop"])
        results = self.system.retrieve_memory("森林 玩家")
        assert results
        assert "森林" in results[0].content

    def test_case_06(self):
        """检索有关联：共享标签的相关记忆会被一起找到"""
        self.system.store_memory("玩家在森林里救了我", tags=["player", "forest"], importance=0.8)
        self.system.store_memory("玩家送我一把剑", tags=["player", "gift"], importance=0.8)
        self.system.store_memory("今天下雨了", tags=["weather"], importance=0.8)
        results = self.system.retrieve_memory("森林", limit=3)
        contents = [m.content for m in results]
        assert any("剑" in c for c in contents)  # 通过 player 标签关联到
        assert not any("下雨" in c for c in contents)

    def test_case_07(self):
        """检索有时间衰减：相关性相同时新记忆排在旧记忆前"""
        self.system.store_memory("玩家第一次帮助我", tags=["player", "help"], importance=0.6)
        self.system.update(20.0)
        self.system.store_memory("玩家再次帮助我", tags=["player", "help"], importance=0.6)
        results = self.system.retrieve_memory("玩家 帮助", limit=2)
        assert len(results) == 2
        assert "再次" in results[0].content

    # ---- 对话历史：完整记录 + 上下文 + 摘要 ----

    def test_case_08(self):
        """对话历史完整记录所有消息，而不是只记最后一条"""
        self.system.add_dialogue("player", "你好")
        self.system.add_dialogue("npc", "你好，旅行者")
        self.system.add_dialogue("player", "我叫小明")
        history = self.system.get_dialogue_history()
        assert len(history) == 3
        assert history[0].text == "你好"
        assert history[2].text == "我叫小明"

    def test_case_09(self):
        """对话历史有上下文：能知道之前在聊什么"""
        self.system.add_dialogue("player", "我想买一把剑")
        self.system.add_dialogue("npc", "这把剑很适合你")
        self.system.add_dialogue("player", "多少钱")
        context = self.system.get_context()
        recent_texts = [t.text for t in context["recent"]]
        assert any("剑" in t for t in recent_texts)

    def test_case_10(self):
        """长对话历史会生成摘要"""
        system = NPCMemory({"summary_threshold": 5})
        for i in range(12):
            system.add_dialogue("player", f"我帮助了村民第{i}次")
            system.add_dialogue("npc", "谢谢你")
        context = system.get_context(recent=4)
        assert context["summary"]
        assert "帮助" in context["summary"]
        assert len(context["recent"]) == 4

    # ---- 情感关系：多维度 + 行为驱动变化 ----

    def test_case_11(self):
        """情感关系有好感度、信任度、恐惧度多个维度"""
        rel = self.system.get_relationship()
        assert "affection" in rel
        assert "trust" in rel
        assert "fear" in rel

    def test_case_12(self):
        """玩家做好事NPC会感激，做坏事会恐惧"""
        self.system.record_interaction("help")
        self.system.record_interaction("gift")
        rel = self.system.get_relationship()
        assert rel["affection"] > 0
        assert rel["trust"] > 0
        self.system.record_interaction("attack")
        rel = self.system.get_relationship()
        assert rel["fear"] > 0
        assert rel["trust"] < 0.1

    # ---- 性格演变：经历驱动 + 稳定性 ----

    def test_case_13(self):
        """性格随经历演变，NPC不再是同一个性格"""
        before = self.system.get_personality()
        for _ in range(10):
            self.system.record_experience("combat_victory")
        after = self.system.get_personality()
        assert after["brave"] > before["brave"]

    def test_case_14(self):
        """性格演变有稳定性：稳定性越高变化越慢"""
        stable = NPCMemory({"stability": 0.95})
        unstable = NPCMemory({"stability": 0.2})
        stable.record_experience("combat_victory")
        unstable.record_experience("combat_victory")
        stable_change = abs(stable.get_personality()["brave"] - 0.5)
        unstable_change = abs(unstable.get_personality()["brave"] - 0.5)
        assert 0 < stable_change < unstable_change

    # ---- 综合：玩家帮了100次，NPC必须记得并感激 ----

    def test_case_15(self):
        """玩家帮助NPC多次后，NPC记得玩家且关系变好"""
        for i in range(100):
            self.system.store_memory(f"玩家第{i}次帮助了我", tags=["player", "help"])
            self.system.record_interaction("help")
        rel = self.system.get_relationship()
        assert rel["affection"] > 0.9
        results = self.system.retrieve_memory("玩家 帮助", limit=5)
        assert len(results) > 0
        assert all("帮助" in m.content for m in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
