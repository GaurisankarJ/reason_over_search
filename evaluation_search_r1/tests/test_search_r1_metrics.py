from flashrag.evaluator.metrics import SearchR1Reward


class DummyItem:
    def __init__(self, reward, format_valid, retrieval_hit):
        self.search_r1_reward = reward
        self.search_r1_format_valid = format_valid
        self.search_r1_retrieval_hit = retrieval_hit


def test_search_r1_metrics_aggregate():
    cfg = {"dataset_name": "bamboogle"}
    metric = SearchR1Reward(cfg)
    data = [
        DummyItem(1.0, True, True),
        DummyItem(0.2, True, False),
        DummyItem(0.1, False, False),
    ]
    result, scores = metric.calculate_metric(data)
    assert scores == [1.0, 0.2, 0.1]
    assert round(result["search_r1_reward"], 6) == round((1.0 + 0.2 + 0.1) / 3, 6)
    assert round(result["search_r1_format_valid_rate"], 6) == round(2 / 3, 6)
    assert round(result["search_r1_retrieval_hit_rate"], 6) == round(1 / 3, 6)

