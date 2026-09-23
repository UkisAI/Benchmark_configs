#!/usr/bin/env python3
"""Export ERQA (embodiedreasoning/ERQA data/erqa.tfrecord) to images + a hash-audited manifest.
Pure-Python TFRecord + tf.train.Example wire parser (no tensorflow on this box).
Mirrors loading_example.py feature spec: answer(bytes), image/encoded(bytes list), question_type(bytes list), visual_indices(int64 list), question(bytes)."""

import collections, hashlib, json, os, pathlib, struct, tempfile

TFRECORD = pathlib.Path(os.environ["ERQA_TFRECORD"])
OUTPUT = pathlib.Path(os.environ["ERQA_OUT"])
IMAGES = OUTPUT / "images"
MANIFEST = OUTPUT / "manifest.jsonl"
EXPECTED = 400
REPO_COMMIT = os.environ.get("ERQA_REPO_REVISION", "")
TFRECORD_SHA256 = os.environ.get("ERQA_TFRECORD_SHA256", "")


def records(path):
    with open(path, "rb") as f:
        while True:
            head = f.read(8)
            if not head:
                return
            (length,) = struct.unpack("<Q", head)
            f.read(4)  # masked crc32c of length
            data = f.read(length)
            f.read(4)  # masked crc32c of data
            if len(data) != length:
                raise RuntimeError("truncated tfrecord")
            yield data


def varint(buf, i):
    shift = 0
    result = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, i
        shift += 7


def fields(buf):
    """Yield (field_number, wire_type, value) for a protobuf message."""
    i = 0
    while i < len(buf):
        key, i = varint(buf, i)
        fn, wt = key >> 3, key & 7
        if wt == 0:
            v, i = varint(buf, i)
        elif wt == 2:
            ln, i = varint(buf, i)
            v = buf[i : i + ln]
            i += ln
        elif wt == 1:
            v = buf[i : i + 8]
            i += 8
        elif wt == 5:
            v = buf[i : i + 4]
            i += 4
        else:
            raise RuntimeError(f"unsupported wire type {wt}")
        yield fn, wt, v


def parse_feature(buf):
    out = None
    for fn, wt, v in fields(buf):
        if fn == 1:  # BytesList
            out = [x for f2, w2, x in fields(v) if f2 == 1]
        elif fn == 3:  # Int64List (packed or not)
            vals = []
            for f2, w2, x in fields(v):
                if f2 != 1:
                    continue
                if w2 == 2:
                    j = 0
                    while j < len(x):
                        n, j = varint(x, j)
                        vals.append(n - (1 << 64) if n >= (1 << 63) else n)
                else:
                    vals.append(x - (1 << 64) if x >= (1 << 63) else x)
            out = vals
        elif fn == 2:
            raise RuntimeError("float_list unexpected")
    return out if out is not None else []


def parse_example(buf):
    feats = {}
    for fn, wt, v in fields(buf):
        if fn != 1:
            continue  # Example.features
        for f2, w2, entry in fields(v):
            if f2 != 1:
                continue  # Features.feature map entry
            key = None
            val = None
            for f3, w3, x in fields(entry):
                if f3 == 1:
                    key = x.decode()
                elif f3 == 2:
                    val = parse_feature(x)
            feats[key] = val
    return feats


def sniff(data):
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png", "png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg", "jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp", "webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif", "gif"
    raise RuntimeError(f"unknown image format {data[:12]!r}")


def main():
    if (
        TFRECORD_SHA256
        and hashlib.sha256(TFRECORD.read_bytes()).hexdigest() != TFRECORD_SHA256
    ):
        raise RuntimeError("tfrecord sha256 mismatch")
    IMAGES.mkdir(parents=True, exist_ok=True)
    rows = []
    types = collections.Counter()
    nimg = collections.Counter()
    answers = collections.Counter()
    total_bytes = 0
    mimes = collections.Counter()
    for qid, rec in enumerate(records(TFRECORD)):
        ex = parse_example(rec)
        question = ex["question"][0].decode("utf-8")
        answer = ex["answer"][0].decode("utf-8")
        qtype = ex.get("question_type", [])
        qtype = qtype[0].decode("utf-8") if qtype else "Unknown"
        vis = [int(x) for x in ex.get("visual_indices", [])]
        images = []
        for k, data in enumerate(ex["image/encoded"]):
            mime, ext = sniff(data)
            mimes[mime] += 1
            digest = hashlib.sha256(data).hexdigest()
            path = IMAGES / f"{qid:04d}-{k}-{digest[:16]}.{ext}"
            if (
                not path.exists()
                or hashlib.sha256(path.read_bytes()).hexdigest() != digest
            ):
                tmp = path.with_suffix(path.suffix + ".tmp")
                tmp.write_bytes(data)
                os.replace(tmp, path)
            images.append(
                {
                    "path": str(path),
                    "sha256": digest,
                    "bytes": len(data),
                    "mime_type": mime,
                }
            )
            total_bytes += len(data)
        rows.append(
            {
                "question_id": qid,
                "question": question,
                "answer": answer,
                "question_type": qtype,
                "visual_indices": vis,
                "images": images,
            }
        )
        types[qtype] += 1
        nimg[len(images)] += 1
        answers[answer] += 1
    if len(rows) != EXPECTED:
        raise RuntimeError(f"expected {EXPECTED} rows, got {len(rows)}")
    fd, tmpm = tempfile.mkstemp(prefix=MANIFEST.name + ".", dir=OUTPUT)
    with os.fdopen(fd, "w") as h:
        for r in rows:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmpm, MANIFEST)
    print(
        json.dumps(
            {
                "repo_commit": REPO_COMMIT,
                "tfrecord_sha256": TFRECORD_SHA256,
                "rows": len(rows),
                "image_bytes": total_bytes,
                "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
                "question_types": dict(types),
                "images_per_question": dict(nimg),
                "answer_letters": dict(answers),
                "mime_types": dict(mimes),
                "max_images": max(nimg),
                "visual_indices_nonzero_questions": sum(
                    1 for r in rows if any(i != 0 for i in r["visual_indices"])
                ),
                "visual_indices_empty_questions": sum(
                    1 for r in rows if not r["visual_indices"]
                ),
            },
            indent=1,
        )
    )
    print("=== SAMPLE QUESTIONS (first 6 + 3 multi-image) ===")
    shown = rows[:6] + [r for r in rows if len(r["images"]) > 1][:3]
    for r in shown:
        print(
            f"--- q{r['question_id']} type={r['question_type']} answer={r['answer']} vis={r['visual_indices']} nimg={len(r['images'])} qlen={len(r['question'])}"
        )
        print(r["question"])
    print("=== ANSWER-INSTRUCTION AUDIT ===")
    import re

    pats = {
        "single letter": r"single letter|one letter|only the letter|just the letter",
        "answer with": r"answer with",
        "options A-D present": r"(?m)^\s*\(?[A-D][\).:]",
    }
    for name, p in pats.items():
        print(name, sum(1 for r in rows if re.search(p, r["question"], re.I)))
    print(
        "answers not in ABCD:",
        [r["question_id"] for r in rows if r["answer"] not in list("ABCD")][:20],
    )


if __name__ == "__main__":
    main()
