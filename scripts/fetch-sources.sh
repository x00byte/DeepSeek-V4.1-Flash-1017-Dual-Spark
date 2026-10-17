#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build
clone_at() {
 local url=$1 dir=$2 commit=$3
 if [[ ! -d "$dir/.git" ]]; then git clone --depth 1 "$url" "$dir"; fi
 [[ -z $(git -C "$dir" status --porcelain) ]] || { echo "Source checkout has local changes: $dir" >&2; exit 1; }
 git -C "$dir" fetch origin "$commit"
 git -C "$dir" checkout --detach "$commit"
}
clone_at https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks build/mia 6f7d1590ad49a2b8995188e45d7b9db31e677452
clone_at https://github.com/jayleaton/deepseek-v41-tensorfold-spark build/jayleaton 2dcce0303451da2a59f80e6c7413da8b3b0f7a57
git -C build/jayleaton submodule update --init vendor/TensorFold
[[ $(git -C build/jayleaton/vendor/TensorFold rev-parse HEAD) == c4646171139ee8a3c38103eaa1699dad226ec12b ]]
# Work on exports, retaining upstream checkouts for review and repeatability.
for target in mia-runtime tensorfold; do
 [[ ! -e "build/$target" ]] || { echo "build/$target already exists; reuse it or move it aside explicitly." >&2; exit 1; }
 mkdir "build/$target"
done
git -C build/mia archive HEAD | tar -x -C build/mia-runtime
git -C build/jayleaton/vendor/TensorFold archive HEAD | tar -x -C build/tensorfold
patch -d build/mia-runtime -p1 < patches/mia-1017.patch
for item in build/jayleaton/patches/*.patch; do patch -d build/tensorfold -p1 < "$item"; done
patch -d build/tensorfold -p1 < patches/tensorfold-1017.patch
echo 'Pinned serving sources prepared. No model or GPU service was started.'
