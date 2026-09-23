import ast
from pathlib import Path


def test_legacy_dp_actor_synchronizes_every_dynamic_batch_split():
    actor_path = Path(__file__).parents[3] / "verl" / "workers" / "actor" / "dp_actor.py"
    tree = ast.parse(actor_path.read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "prepare_dynamic_batch"
    ]

    assert len(calls) == 3
    assert all(any(keyword.arg == "dp_group" for keyword in call.keywords) for call in calls)
