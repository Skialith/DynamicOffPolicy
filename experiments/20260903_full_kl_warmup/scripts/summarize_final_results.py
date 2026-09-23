"""Summarize observed counts and timers from the preserved cluster snapshot."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
TABLES = ROOT / "tables"
DOCS = ROOT / "docs"
for directory in (TABLES, DOCS):
    directory.mkdir(parents=True, exist_ok=True)
SNAPSHOT = json.loads((RAW / "raw_final_results_snapshot.json").read_text())


def duration(seconds):
    seconds = round(seconds)
    return f"{seconds // 3600:02}:{seconds % 3600 // 60:02}:{seconds % 60:02}"


slurm = {}
for row in csv.DictReader(SNAPSHOT["sacct"].splitlines(), delimiter="|"):
    if "." not in row["JobID"]:
        slurm[row["JobID"]] = row

summary = {}
for run in SNAPSHOT["runs"]:
    n = run["n"]
    cycles = run["cycles"]
    assert [int(c["training/optimizer_step"]) for c in cycles] == list(range(n, 97, n))
    tokens = run["tokens_by_update"]
    assert [r["optimizer_step"] for r in tokens] == list(range(1, 97))
    assert all(r["token_count"] == r["update_token_count"] == r["rollout_token_count"] for r in tokens)
    kl = run["kl_timings"]
    assert [r["optimizer_step"] for r in kl] == list(range(1, 97))
    anchors = {}
    for r in kl:
        if r["anchor_update"] in anchors:
            assert anchors[r["anchor_update"]] == r["anchor_seconds"]
        anchors[r["anchor_update"]] = r["anchor_seconds"]
    assert len(anchors) == 96 // n
    timing = {key: sum(c.get(key, 0) for c in cycles)
              for key in set().union(*(c.keys() for c in cycles)) if key.startswith("timing_s/")}
    total = int(sum(c["perf/total_num_tokens"] for c in cycles))
    response = int(sum(r["token_count"] for r in tokens))
    prompt = total - response
    assert prompt == round(sum(c["prompt_length/mean"] * 256 * n * 8 for c in cycles))
    evals = {r["optimizer_step"]: r["val-core/math_sis/acc/mean@1"] for r in run["eval_metrics"]}
    assert sorted(evals) == [0, 48, 96]
    anchor_time = sum(anchors.values())
    update_measure_time = sum(r["measurement_seconds"] for r in kl)
    measure_time = anchor_time + update_measure_time
    summary[n] = {
        "job_id": run["job_id"], "n": n, "updates": 96, "cycles": len(cycles),
        "trajectory_count": 96 * 256 * 8, "eval": evals,
        "prompt_tokens": prompt, "response_tokens": response, "total_tokens": total,
        "training_cycle_seconds": timing["timing_s/step"],
        "seconds_per_update": timing["timing_s/step"] / 96,
        "kl_anchor_seconds": anchor_time, "kl_update_measurement_seconds": update_measure_time,
        "kl_seconds": measure_time, "kl_percent_of_training": measure_time / timing["timing_s/step"] * 100,
        "eval_48_96_seconds": timing["timing_s/testing"],
        "save_hf_seconds": timing["timing_s/save_checkpoint"],
        "total_tokens_per_second_4gpu": total / timing["timing_s/step"],
        "slurm_seconds": int(slurm[run["job_id"]]["ElapsedRaw"]), "timing_breakdown": timing,
    }

table = []


def add(label, formatter):
    table.append([label, formatter(summary[4]), formatter(summary[8])])


add("作业号", lambda s: s["job_id"])
add("真实 optimizer updates / rollout 周期", lambda s: f"{s['updates']} / {s['cycles']}")
for step in (0, 48, 96):
    add(f"MATH500 sampled Avg@1，update {step}", lambda s, step=step: f"{s['eval'][step] * 100:.1f}%")
add("训练 trajectory 数", lambda s: f"{s['trajectory_count']:,}")
add("实际 response token 数", lambda s: f"{s['response_tokens']:,}")
add("实际 prompt token 数（按 trajectory 计）", lambda s: f"{s['prompt_tokens']:,}")
add("训练 token 合计", lambda s: f"{s['total_tokens']:,}")
add("训练周期累计耗时", lambda s: duration(s["training_cycle_seconds"]))
add("平均每次 update 耗时", lambda s: f"{s['seconds_per_update']:.2f} 秒")
add("逐步全词表 KL 测量计时", lambda s: f"{s['kl_seconds']:.2f} 秒（{duration(s['kl_seconds'])}）")
add("KL 测量计时 / 训练周期耗时", lambda s: f"{s['kl_percent_of_training']:.2f}%")
add("Update 48、96 能力评测计时合计", lambda s: f"{s['eval_48_96_seconds']:.2f} 秒")
add("终点 HF 权重保存计时", lambda s: f"{s['save_hf_seconds']:.2f} 秒")
add("平均吞吐（4 卡合计）", lambda s: f"{s['total_tokens_per_second_4gpu']:,.2f} token/s")
add("Slurm 作业总耗时", lambda s: duration(s["slurm_seconds"]))

with (TABLES / "final_results_table.csv").open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.writer(handle)
    writer.writerow(["指标", "N=4", "N=8"])
    writer.writerows(table)

endpoint = SNAPSHOT["endpoint_kl"]
times = SNAPSHOT["endpoint_file_times"]
endpoint_seconds_estimate = times["endpoint_kl.json"]["mtime"] - times["endpoint.lock"]["mtime"]
a, b = summary[4], summary[8]
comparisons = {
    "n8_training_time_reduction_percent": 100 * (1 - b["training_cycle_seconds"] / a["training_cycle_seconds"]),
    "n8_token_count_reduction_percent": 100 * (1 - b["total_tokens"] / a["total_tokens"]),
    "n8_token_throughput_increase_percent": 100 * (b["total_tokens_per_second_4gpu"] / a["total_tokens_per_second_4gpu"] - 1),
}
output = {"runs": summary, "endpoint_kl": endpoint,
          "endpoint_seconds_estimate_from_file_times": endpoint_seconds_estimate, "comparisons": comparisons}
(TABLES / "final_results_summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")

print("Validated 36 cycles and 192 token/timing records; wrote CSV and JSON summaries.")
