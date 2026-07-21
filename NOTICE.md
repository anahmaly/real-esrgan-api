# Third-party notices

This repository integrates, without bundling model weights:

- Real-ESRGAN inference code (`xinntao/Real-ESRGAN`, tag `v0.3.0`, commit `fa4c8a03ae3dbc9ea6ed471a6ab5da94ac15c2ea`), BSD-3-Clause.
- rembg and the validated BiRefNet HR service implementation from `anahmaly/rembg-api` PR #7, head `dd7b6fd434cff2077ce6e9a0cab46fe254f26f1f`. rembg and BiRefNet are MIT-licensed; BRIA model artifacts have separate operator-supplied terms.
- Official Ideogram 4 inference code (`ideogram-oss/ideogram4`, commit `990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2`), Apache-2.0.
- Official LongCat-Image inference contract (`meituan-longcat/LongCat-Image`, commit `f0e4c43c5ef74b011ff71570fbfc2bdffbc9ab06`) and Diffusers integration, Apache-2.0. The configured model snapshots are `LongCat-Image-Edit@7b54ef423aa7854be7861600024be5c56ab7875a` and `LongCat-Image-Edit-Turbo@6a7262de5549f0bf0ec54c08ef7d283ef41f3214`.

The dependency license texts are preserved under `licenses/`. Model weights are not part of this repository or its software license. Operators must obtain and comply with the applicable Real-ESRGAN, BRIA, BiRefNet, Ideogram, and LongCat weight licenses. A separate commercial Ideogram agreement, when applicable, is operator-held and is not included here.
