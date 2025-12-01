# tests/test_basic.py
from src.main_agent import Agent
import pytest

def test_agent_search_and_sum():
    # this test assumes data/meta.json and faiss.index exist (small build recommended)
    agent = Agent()
    res = agent.retrieve_chunks("transformer architecture", top_k=2)
    assert isinstance(res, list)
    assert len(res) <= 2
    for r in res:
        assert "text" in r and "score" in r
