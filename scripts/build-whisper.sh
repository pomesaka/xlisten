#!/usr/bin/env bash
# Builds whisper-server of whisper.cpp, with Metal, into .cache/whisper.cpp/ of this repository.
# Nothing is installed outside the repository: delete .cache/ to remove it.
set -euo pipefail

# WHY v1.9.5: the latest release when this was written (2026-11-05, released 2026-10-06).
VERSION=v1.9.5

root="$(cd "$(dirname "$0")/.." && pwd)"
src="${root}/.cache/whisper.cpp/src-${VERSION}"
build="${root}/.cache/whisper.cpp/build-${VERSION}"

if [[ ! -d "${src}" ]]; then
	mkdir -p "$(dirname "${src}")"
	git clone --depth 1 --branch "${VERSION}" https://github.com/ggml-org/whisper.cpp.git "${src}"
fi

cmake -S "${src}" -B "${build}" -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON \
	-DWHISPER_BUILD_TESTS=OFF -DBUILD_SHARED_LIBS=OFF
cmake --build "${build}" --config Release -j --target whisper-server whisper-cli

ln -sfn "build-${VERSION}/bin" "${root}/.cache/whisper.cpp/bin"
echo "${root}/.cache/whisper.cpp/bin/whisper-server"
