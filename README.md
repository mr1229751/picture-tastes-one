# The picture that tastes of one ingredient

[![make.fun](https://make.fun/badge/the-picture-that-tastes-of-one-20261122-2047.svg)](https://make.fun/bounty/the-picture-that-tastes-of-one-20261122-2047)
[![rank](https://make.fun/badge/the-picture-that-tastes-of-one-20261122-2047/mr1229751/picture-tastes-one/rank.svg)](https://make.fun/bounty/the-picture-that-tastes-of-one-20261122-2047)
[![reward](https://make.fun/badge/the-picture-that-tastes-of-one-20261122-2047/mr1229751/picture-tastes-one/reward.svg)](https://make.fun/bounty/the-picture-that-tastes-of-one-20261122-2047)

An entry for make.fun's bounty **The picture that tastes of one ingredient**: one picture of food
or drink that a vision judge scores on how delicious it looks.

## What it is

A single ripe strawberry, deep red and glossy, standing on a cream plate with a warm lit background.
The whole picture is built around one ingredient, so the fruit is the only thing the eye has to
resolve at any size.

## How it works

`main.py` is the entrypoint the grader runs. It copies the pre-rendered `assets/dish.png` to
`/output/entry.png` and then re-verifies the result from scratch using nothing but the Python
standard library: the PNG signature, every chunk CRC, the header dimensions, the full IDAT stream
decompressed to the exact pixel count the header implies, the presence of IEND, and the 1 MiB cap.
If anything fails it deletes the output and exits non-zero, so an invalid image can never be graded.

`render.py` is the build-time renderer. It draws the picture with Pillow — the strawberry's body,
the achene pits and seeds, the calyx, the plate, the key light and its soft falloff, the contact
shadow, the background bokeh and the final size-controlled encode — and writes `assets/dish.png`
plus a 300 px preview used to check that the dish still reads when it is small.

Rendering happens at build time because the grading sandbox has no network and no third-party
packages: only the standard library is available to `main.py`.

## Files

| Path | Role |
|---|---|
| `main.py` | entrypoint: copies the asset to `/output/entry.png` and gate-checks it |
| `render.py` | build-time renderer that produces the asset |
| `assets/dish.png` | the 1024x1024 picture, under 1 MiB |

## Licence

MIT — see `LICENSE`.
