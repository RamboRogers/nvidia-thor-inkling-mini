#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the controlled four-task speed probe against a local Inkling Thor API."""
import argparse
import hashlib
import http.client
import json
import re
import statistics
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing

HOST, PORT, MODEL = "127.0.0.1", 8010, "inkling-mini"

PROMPTS_C1 = {
    "code": (
        "Write only Python. Start by defining double(n) to return n * 2. "
        "Then define class BatchScaler with a scale_batch(self, values) method that yields double(value) "
        "for each input value. Include validation, iterable support, error handling, docstrings, "
        "usage examples and a comprehensive unittest suite of at least 40 cases. Implement the module fully."
    ),
    "text": (
        'A shipping manifest has 17 blue and 25 red crates. Begin with the line "Total: N" replacing N '
        "with their sum. Then write an extensive 120-item numbered receiving checklist for counting, "
        "inspecting and storing the crates. Each item should include a distinct action and its reason. No preamble."
    ),
    "prose": (
        "Write a detailed guide about a library digitizing its archive. It has 600 boxes and can scan 20 per day. "
        'Start with a line "Scanning time: N days" replacing N with the correct number. Then write at least '
        "2500 words about scanning, indexing, metadata, preservation and backup. Use concrete examples; no preamble."
    ),
    "math": (
        "Solve step by step and show every intermediate calculation clearly: "
        "A factory produces widgets in three shifts. Shift A produces 42 widgets/hour for 8 hours, "
        "Shift B produces 15% more per hour for 7.5 hours, and Shift C produces 85% of Shift A per hour "
        "for 6 hours. 4% of Shift A, 6% of Shift B, and 3% of Shift C widgets fail inspection. "
        "Each passing widget sells for $18.50 and costs $7.25 in materials plus $2,400 fixed daily overhead. "
        "Compute exact daily revenue, costs, net profit, and how many days to earn $100,000 net profit. "
        "Then generalize the formula for arbitrary rates and analyze sensitivity to each defect rate."
    ),
}


def http_json(path, body=None, timeout=600):
    data = json.dumps(body).encode() if body is not None else None
    with closing(http.client.HTTPConnection(HOST, PORT, timeout=timeout)) as conn:
        conn.request("GET" if data is None else "POST", path, data, {"Content-Type": "application/json"})
        with conn.getresponse() as r:
            raw = r.read()
            if r.status != 200:
                raise RuntimeError(f"HTTP {r.status} on {path}")
            return raw


def get_metrics():
    raw = http_json("/metrics", timeout=10).decode("utf-8", errors="replace")
    out = {}
    pat = (
        r"^(vllm:(?:num_requests_running|num_requests_waiting|"
        r"spec_decode_num_drafts_total|spec_decode_num_accepted_tokens_total|"
        r"spec_decode_num_draft_tokens_total|prefix_cache_hits_total|"
        r"prefix_cache_queries_total))\{[^}]*\}\s+(\S+)"
    )
    for m in re.finditer(pat, raw, re.M):
        out[m.group(1)] = float(m.group(2))
    return out


def stream_chat(messages, max_tokens, reasoning_effort="none", temperature=0.0, timeout=900, continuous_usage=False):
    opts = {"include_usage": True}
    if continuous_usage:
        opts["continuous_usage_stats"] = True
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
        "reasoning_effort": reasoning_effort,
        "stream": True,
        "stream_options": opts,
    }
    data = json.dumps(body).encode()
    content_parts, reasoning_parts = [], []
    t_first_any = t_first_reason = t_first_content = t_last_any = t_end = None
    first_any_toks = None
    usage = None
    finish = None
    with closing(http.client.HTTPConnection(HOST, PORT, timeout=timeout)) as conn:
        t0 = time.perf_counter()
        conn.request(
            "POST",
            "/v1/chat/completions",
            data,
            {"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        with conn.getresponse() as r:
            if r.status != 200:
                raise RuntimeError(f"HTTP {r.status} on chat completion")
            while True:
                line = r.readline()
                if not line:
                    break
                now = time.perf_counter()
                line = line.rstrip(b"\r\n")
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].lstrip(b" ")
                if payload == b"[DONE]":
                    t_end = now
                    break
                ev = json.loads(payload)
                if ev.get("usage"):
                    usage = ev["usage"]
                choices = ev.get("choices") or []
                if not choices:
                    continue
                ch = choices[0]
                if ch.get("finish_reason") is not None:
                    finish = ch["finish_reason"]
                delta = ch.get("delta") or {}
                c = delta.get("content")
                rc = delta.get("reasoning") or delta.get("reasoning_content")
                if rc:
                    reasoning_parts.append(rc)
                    if t_first_reason is None:
                        t_first_reason = now
                    if t_first_any is None:
                        t_first_any = now
                        if usage:
                            first_any_toks = usage.get("completion_tokens")
                    t_last_any = now
                if c:
                    content_parts.append(c)
                    if t_first_content is None and c.strip():
                        t_first_content = now
                    if t_first_any is None:
                        t_first_any = now
                        if usage:
                            first_any_toks = usage.get("completion_tokens")
                    t_last_any = now
    if t_end is None or usage is None or finish is None:
        raise RuntimeError("incomplete SSE: expected finish reason, native usage and [DONE]")
    pt = usage["prompt_tokens"] if usage else 0
    ct = usage["completion_tokens"] if usage else 0
    rt = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) if usage else 0
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) if usage else 0
    text = "".join(content_parts)
    rtext = "".join(reasoning_parts)
    e2e_s = t_end - t0
    ttft_any = (t_first_any - t0) if t_first_any else e2e_s
    ttft_vis = (t_first_content - t0) if t_first_content else None
    ttft_reas = (t_first_reason - t0) if t_first_reason else None
    dec_s = (t_last_any - t_first_any) if (t_first_any and t_last_any and t_last_any > t_first_any) else None
    dec_toks = (ct - (first_any_toks or 1)) if ct > 1 else 0
    dec_tps = (dec_toks / dec_s) if (dec_s and dec_toks > 0) else (ct / e2e_s)
    return {
        "prompt_tokens": pt,
        "cached_tokens": cached,
        "completion_tokens": ct,
        "reasoning_tokens": rt,
        "finish_reason": finish,
        "ttft_any_s": round(ttft_any, 4),
        "ttft_vis_s": round(ttft_vis, 4) if ttft_vis is not None else None,
        "ttft_reas_s": round(ttft_reas, 4) if ttft_reas is not None else None,
        "e2e_s": round(e2e_s, 4),
        "e2e_tps": round(ct / e2e_s, 2),
        "decode_tps": round(dec_tps, 2),
        "text_sha256": hashlib.sha256((rtext + "\0" + text).encode()).hexdigest()[:16],
    }


def run_c1_suite(effort, max_tokens, repeats):
    results = {}
    for task, prompt in PROMPTS_C1.items():
        m0 = get_metrics()
        rows = []
        for i in range(repeats + 1):
            r = stream_chat(
                [{"role": "user", "content": prompt}],
                max_tokens,
                reasoning_effort=effort,
                continuous_usage=True,
            )
            if i > 0:
                rows.append(r)
        m1 = get_metrics()
        dd = m1.get("vllm:spec_decode_num_drafts_total", 0) - m0.get("vllm:spec_decode_num_drafts_total", 0)
        da = m1.get("vllm:spec_decode_num_accepted_tokens_total", 0) - m0.get(
            "vllm:spec_decode_num_accepted_tokens_total", 0
        )
        tok_per_step = round(1.0 + da / dd, 3) if dd > 0 else 1.0
        med_dec = round(statistics.median(x["decode_tps"] for x in rows), 2)
        med_e2e = round(statistics.median(x["e2e_tps"] for x in rows), 2)
        med_ttft = round(statistics.median(x["ttft_any_s"] for x in rows), 4)
        vis_vals = [x["ttft_vis_s"] for x in rows if x["ttft_vis_s"] is not None]
        med_vis = round(statistics.median(vis_vals), 4) if vis_vals else None
        results[task] = {
            "med_decode_tps": med_dec,
            "med_e2e_tps": med_e2e,
            "med_ttft_any_s": med_ttft,
            "med_ttft_vis_s": med_vis,
            "tok_per_step": tok_per_step,
            "mean_ct": round(statistics.mean(x["completion_tokens"] for x in rows), 1),
            "hashes": [x["text_sha256"] for x in rows],
            "runs": rows,
        }
        print(
            f"  [C1 {effort:6s}] {task:5s}: decode={med_dec:6.2f} tok/s | e2e={med_e2e:6.2f} tok/s "
            f"| ttft_any={med_ttft:.3f}s | ttft_vis={med_vis} | tok/step={tok_per_step:.3f} | hash={rows[0]['text_sha256']}",
            flush=True,
        )
    overall_dec = round(statistics.mean(v["med_decode_tps"] for v in results.values()), 2)
    overall_e2e = round(statistics.mean(v["med_e2e_tps"] for v in results.values()), 2)
    print(f"  >>> [C1 {effort:6s} MEAN] decode={overall_dec:.2f} tok/s | e2e={overall_e2e:.2f} tok/s", flush=True)
    return {"tasks": results, "mean_decode_tps": overall_dec, "mean_e2e_tps": overall_e2e}


def run_prefill_suite(lengths):
    out = {}
    for target_toks in lengths:
        salt = uuid.uuid4().hex
        chunk = (
            f"Record {salt}: warehouse telemetry log entry checking thermal stability, "
            "bandwidth utilization, and calibration markers.\n"
        )
        tok_resp = json.loads(
            http_json(
                "/tokenize",
                {
                    "model": MODEL,
                    "messages": [{"role": "user", "content": chunk * 10}],
                    "add_generation_prompt": True,
                    "chat_template_kwargs": {"reasoning_effort": "none"},
                },
            )
        )
        per_chunk = max(1.0, tok_resp["count"] / 10.0)
        n_chunks = max(1, int((target_toks - 60) / per_chunk))
        doc = f"DOC-SALT-{salt}\n" + "".join(f"[{i:05d}] {chunk}" for i in range(n_chunks))
        q1 = doc + f"\nQuestion 1: Reply with ONLY the doc salt string DOC-SALT-{salt[:6]}..."
        r_cold = stream_chat([{"role": "user", "content": q1}], 32, reasoning_effort="none", continuous_usage=True)
        pf_tps = round(r_cold["prompt_tokens"] / r_cold["ttft_any_s"], 1) if r_cold["ttft_any_s"] > 0 else 0
        q2 = doc + "\nQuestion 2: Summarize the main topic in one sentence."
        r_warm = stream_chat([{"role": "user", "content": q2}], 32, reasoning_effort="none", continuous_usage=True)
        entry = {
            "prompt_tokens": r_cold["prompt_tokens"],
            "cold_ttft_s": r_cold["ttft_any_s"],
            "cold_prefill_tps": pf_tps,
            "warm_ttft_s": r_warm["ttft_any_s"],
            "warm_cached_tokens": r_warm["cached_tokens"],
        }
        if target_toks >= 32000:
            q3 = doc + "\nQuestion 3: Write a detailed 300-word analysis of the telemetry entries above."
            r_dec = stream_chat([{"role": "user", "content": q3}], 256, reasoning_effort="none", continuous_usage=True)
            entry["ctx_decode_tps"] = r_dec["decode_tps"]
            entry["ctx_e2e_tps"] = r_dec["e2e_tps"]
            entry["ctx_decode_cached"] = r_dec["cached_tokens"]
        out[str(target_toks)] = entry
        extra = f" | ctx_decode={entry['ctx_decode_tps']} tok/s" if "ctx_decode_tps" in entry else ""
        print(
            f"  [PREFILL {target_toks:6d}] actual={r_cold['prompt_tokens']:6d} "
            f"| cold_ttft={r_cold['ttft_any_s']:6.2f}s ({pf_tps:6.1f} tok/s) "
            f"| cached_ttft={r_warm['ttft_any_s']:.3f}s (cached={r_warm['cached_tokens']}){extra}",
            flush=True,
        )
    return out


def run_c4_decode():
    tasks = list(PROMPTS_C1.items())
    barrier = threading.Barrier(5, timeout=30)
    peak_running = [0]
    done = threading.Event()

    def monitor():
        while not done.is_set():
            try:
                m = get_metrics()
                peak_running[0] = max(peak_running[0], int(m.get("vllm:num_requests_running", 0)))
            except Exception:
                pass
            time.sleep(0.05)

    def worker(idx):
        name, prompt = tasks[idx]
        salt = f"Session {uuid.uuid4().hex[:8]}: "
        barrier.wait()
        return name, stream_chat(
            [{"role": "user", "content": salt + prompt}], 512, reasoning_effort="none", continuous_usage=True
        )

    mt = threading.Thread(target=monitor, daemon=True)
    mt.start()
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(worker, i) for i in range(4)]
        barrier.wait()
        t_start = time.perf_counter()
        res = [f.result() for f in futs]
    t_wall = time.perf_counter() - t_start
    done.set()
    mt.join(timeout=2)
    tot_toks = sum(r["completion_tokens"] for _, r in res)
    agg_tps = round(tot_toks / t_wall, 2)
    per_stream = {n: r["decode_tps"] for n, r in res}
    ttfts = {n: r["ttft_any_s"] for n, r in res}
    print(
        f"  [C4 DECODE] peak_running={peak_running[0]} | agg={agg_tps:.2f} tok/s | wall={t_wall:.2f}s "
        f"| per_stream={per_stream} | ttft={ttfts}",
        flush=True,
    )
    return {
        "peak_running": peak_running[0],
        "wall_s": round(t_wall, 3),
        "total_tokens": tot_toks,
        "agg_tps": agg_tps,
        "per_stream_decode_tps": per_stream,
        "ttft_s": ttfts,
    }


def main():
    global PORT
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("fast", "full"), default="fast")
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--port", type=int, default=PORT)
    a = p.parse_args()
    if not 1 <= a.port <= 65535:
        p.error("port must be 1–65535")
    PORT = a.port
    models = json.loads(http_json("/v1/models"))
    ids = [m["id"] for m in models["data"]]
    if ids != [MODEL] or models["data"][0].get("max_model_len", 0) < 262144:
        p.error("expected the inkling-mini API with a context limit of at least 262144")
    print(f"=== BENCHMARK [{a.label}] mode={a.mode} models={ids} ===", flush=True)
    report = {"label": a.label, "mode": a.mode}
    short_runs = []
    for i in range(4):
        k = "PING-" + uuid.uuid4().hex[:8]
        r = stream_chat(
            [{"role": "user", "content": f"Reply exactly: {k}"}],
            32,
            reasoning_effort="none",
            continuous_usage=True,
        )
        if i > 0:
            short_runs.append(r["ttft_vis_s"] or r["ttft_any_s"])
    report["short_warm_ttft_s"] = round(statistics.median(short_runs), 4)
    print(f"  [SHORT TTFT] warm median={report['short_warm_ttft_s']:.4f}s (runs={short_runs})", flush=True)

    reps = 2 if a.mode == "fast" else 3
    report["c1_none"] = run_c1_suite("none", 512, reps)
    report["c1_medium"] = run_c1_suite("medium", 1024, 1 if a.mode == "fast" else 2)
    report["prefill"] = run_prefill_suite([8192, 32768] if a.mode == "fast" else [8192, 32768, 131072])
    report["c4_decode"] = run_c4_decode()
    with open(a.out, "x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"SAVED: {a.out}", flush=True)


if __name__ == "__main__":
    main()
