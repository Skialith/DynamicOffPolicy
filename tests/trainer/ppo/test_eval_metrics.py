import json
import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.reward_score import default_compute_score


class EvalMetricsTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temporary_directory.name)
        self.trainer = object.__new__(RayPPOTrainer)
        self.trainer.config = OmegaConf.create(
            {
                "trainer": {"default_local_dir": str(self.output_dir)},
                "data": {"train_batch_size": 1024},
                "actor_rollout_ref": {
                    "actor": {"ppo_mini_batch_size": 256, "ppo_epochs": 1},
                },
            }
        )
        self.trainer.global_steps = 1
        self.trainer.optimizer_steps = 4

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_eval_metrics_are_keyed_by_optimizer_step(self):
        self.trainer._append_eval_metrics({"val-core/math500/acc/mean@1": 0.25})
        self.trainer._append_eval_metrics({"val-core/math500/acc/mean@1": 0.5})
        self.trainer.global_steps = 2
        self.trainer.optimizer_steps = 8
        self.trainer._append_eval_metrics({"val-core/math500/acc/mean@1": 0.75})

        records = [
            json.loads(line) for line in (self.output_dir / "eval_metrics.jsonl").read_text().splitlines()
        ]
        self.assertEqual([record["optimizer_step"] for record in records], [4, 8])
        self.assertEqual(records[0]["rollout_step"], 1)
        self.assertEqual(records[0]["reuse_n"], 4)
        self.assertEqual(records[0]["val-core/math500/acc/mean@1"], 0.5)

    def test_validation_generation_filename_and_optimizer_step(self):
        self.trainer._dump_generations(
            inputs=["question"],
            outputs=["Answer: 1"],
            gts=["1"],
            scores=[1.0],
            reward_extra_infos_dict={"acc": [True]},
            dump_path=self.output_dir,
            filename="optimizer_step_0004.jsonl",
            optimizer_step=4,
        )

        record = json.loads((self.output_dir / "optimizer_step_0004.jsonl").read_text())
        self.assertEqual(record["step"], 1)
        self.assertEqual(record["optimizer_step"], 4)
        self.assertIs(record["acc"], True)

    def test_math500_uses_answer_line_scorer(self):
        result = default_compute_score("math500", "reasoning\nAnswer: 34", "34")
        self.assertEqual(result["score"], 1.0)
        self.assertIs(result["acc"], True)


if __name__ == "__main__":
    unittest.main()
