import json, sys

files = [
    "/Users/akky/Documents/HTML/JULIUS/tools/nnue/data/enc/selfplay092/shard1.jsonl",
    "/Users/akky/Documents/HTML/JULIUS/tools/nnue/data/enc/selfplay092/shard2.jsonl",
    "/Users/akky/Documents/HTML/JULIUS/tools/nnue/data/enc/selfplay092/shard3.jsonl",
    "/Users/akky/Documents/HTML/JULIUS/tools/nnue/data/enc/selfplay092/shard4.jsonl",
    "/Users/akky/Documents/HTML/JULIUS/tools/nnue/data/enc/selfplay092/shard5.jsonl",
    "/Users/akky/Documents/HTML/JULIUS/tools/nnue/data/enc/selfplay092/gen110_shard1.jsonl",
    "/Users/akky/Documents/HTML/JULIUS/tools/nnue/data/enc/selfplay092/gen110_shard2.jsonl",
    "/Users/akky/Documents/HTML/JULIUS/tools/nnue/data/enc/selfplay092/gen119_shard1.jsonl",
    "/Users/akky/Documents/HTML/JULIUS/tools/nnue/data/enc/selfplay092/gen124_shard1.jsonl",
    "/Users/akky/Documents/HTML/JULIUS/tools/nnue/data/enc/selfplay092/gen124_shard2.jsonl",
]

base_offset = 70000000  # 既存gen014_13の最大pid 65005500 より十分大きい開始点
out_path = "/Users/akky/Documents/HTML/JULIUS/tools/nnue/data/gen015_13.jsonl"

with open(out_path, "w") as out:
    # まず既存gen014_13をそのままコピー
    n_old = 0
    with open("/Users/akky/Documents/HTML/JULIUS/tools/nnue/data/gen014_13.jsonl") as f:
        for line in f:
            out.write(line)
            n_old += 1
    print(f"copied {n_old} records from gen014_13.jsonl", file=sys.stderr)

    n_new = 0
    offset = base_offset
    for fp in files:
        shard_pids = set()
        with open(fp) as f:
            lines = f.readlines()
        for line in lines:
            r = json.loads(line)
            shard_pids.add(r["pid"])
        # このシャード内のpidをoffset+連番にマップ(シャード間・既存データと衝突しないよう)
        pid_map = {p: offset + i for i, p in enumerate(sorted(shard_pids))}
        offset += len(shard_pids) + 1000  # シャード間の余裕
        for line in lines:
            r = json.loads(line)
            r["pid"] = pid_map[r["pid"]]
            out.write(json.dumps(r) + "\n")
            n_new += 1
    print(f"appended {n_new} new self-play records", file=sys.stderr)
    print(f"total records: {n_old + n_new}", file=sys.stderr)
