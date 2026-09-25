"""ERQA: 400 image questions x 5 seeds, 100k tokens, images placed where the official harness puts them. Needs setup.sh."""
import base64, struct
from common import flags, chat, run, report, UPSTREAM

a = flags("erqa", max_tokens=100_000, conc=96, timeout=600)


def varint(b, i):
    n = shift = 0
    while True:
        n |= (b[i] & 0x7F) << shift
        i += 1
        if b[i - 1] < 0x80:
            return n, i
        shift += 7


def fields(b):  # minimal protobuf reader: yields (field number, value)
    i = 0
    while i < len(b):
        key, i = varint(b, i)
        if key & 7 == 0:
            v, i = varint(b, i)
        else:  # length-delimited: the only other wire type in tf.train.Example
            n, i = varint(b, i)
            v, i = b[i:i + n], i + n
        yield key >> 3, v


def example(record):  # tf.train.Example -> {name: [bytes] | [int]}
    out = {}
    for _, entry in fields(dict(fields(record))[1]):
        e = dict(fields(entry))
        kind, values = next(fields(e[2]), (3, b""))
        if kind == 1:
            out[e[1].decode()] = [v for _, v in fields(values)]
        else:  # packed int64 list
            ints, j, packed = [], 0, next((v for _, v in fields(values)), b"")
            while j < len(packed):
                v, j = varint(packed, j)
                ints.append(v)
            out[e[1].decode()] = ints
    return out


def records(path):
    data, i = open(path, "rb").read(), 0
    while i < len(data):
        (n,) = struct.unpack("<Q", data[i:i + 8])
        yield data[i + 12:i + 12 + n]
        i += 16 + n


def image(b):
    mime = "png" if b[:4] == b"\x89PNG" else "jpeg" if b[:3] == b"\xff\xd8\xff" else "webp" if b[8:12] == b"WEBP" else "gif"
    return {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{base64.b64encode(b).decode()}"}}


def content(q, imgs, vis):
    """Port of ERQA eval_harness.py: visual_indices are character offsets into the question."""
    if not vis or all(v == 0 for v in vis):
        return [image(b) for b in imgs] + [{"type": "text", "text": q}]
    parts, last = [], 0
    for b, v in sorted(zip(imgs, vis), key=lambda p: p[1]):
        if 0 < v <= len(q) and q[last:v]:
            parts.append({"type": "text", "text": q[last:v]})
        parts.append(image(b))
        last = v if 0 < v <= len(q) else last
    return parts + ([{"type": "text", "text": q[last:]}] if last < len(q) else [])


rows = [example(r) for r in records(UPSTREAM / "ERQA/data/erqa.tfrecord")]
assert len(rows) == 400, len(rows)
rows = rows[: a.limit or None]


def ask(job):
    seed, r = job
    msg = [{"role": "user", "content": content(r["question"][0].decode(), r["image/encoded"], r.get("visual_indices", []))}]
    x = chat(a, msg, seed, mm_processor_kwargs={"min_pixels": 602112, "max_pixels": 4014080})
    return dict(x, seed=seed, gold=r["answer"][0].decode())


out = run(a, {f"s{s}_{i:03}": (s, r) for s in a.seeds for i, r in enumerate(rows)}, ask)
report(a, [dict(x, correct=x["content"].replace(".", "").strip().lower() == x["gold"].strip().lower()) for x in out])
