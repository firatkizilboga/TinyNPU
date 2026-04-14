#!/usr/bin/env python3

H = 16
W = 16
C = 3
ARRAY = 8
K = 3
PAD = 0
STRIDE = 1
BASE_IMG = 0x8000
BASE_A = 0x9000


def tok(y: int, x: int, c: int) -> str:
    return f"h{y:02d}w{x:02d}c{c}"


def build_raw_hwc():
    raw = []
    for y in range(H):
        for x in range(W):
            for c in range(C):
                raw.append(tok(y, x, c))
    return raw


def build_lane_windows():
    lanes = []
    for lane in range(ARRAY):
        oy = 0
        ox = lane
        seq = []
        for ky in range(K):
            for kx in range(K):
                iy = oy * STRIDE + ky - PAD
                ix = ox * STRIDE + kx - PAD
                for c in range(C):
                    if iy < 0 or iy >= H or ix < 0 or ix >= W:
                        seq.append("0")
                    else:
                        seq.append(tok(iy, ix, c))
        lanes.append(seq)
    return lanes


def print_img(raw):
    print("IMG:")
    for w in range(4):
        vals = raw[w * 16 : (w + 1) * 16]
        print(f"0x{BASE_IMG + w:04x}: " + " ".join(vals))


def print_a(lanes):
    print("\nA:")
    steps = (len(lanes[0]) + 1) // 2
    for s in range(steps):
        parts = []
        for lane in range(ARRAY):
            a = lanes[lane][2 * s]
            b = lanes[lane][2 * s + 1] if 2 * s + 1 < len(lanes[lane]) else "0"
            parts.append(f"L{lane}=({a}, {b})")
        print(f"0x{BASE_A + s:04x}: " + "  ".join(parts))


def main():
    raw = build_raw_hwc()
    lanes = build_lane_windows()
    print_img(raw)
    print_a(lanes)


if __name__ == "__main__":
    main()
