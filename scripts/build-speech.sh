#!/usr/bin/env bash
# Builds `speech` of speech.cpp, with Metal, into .cache/speech.cpp/, and fetches the GGUF of
# NVIDIA's RE-USE (speech enhancement) it reads into .cache/models/, checking its sha256.
# Nothing is installed outside the repository: delete .cache/ to remove it.
set -euo pipefail

# WHY speech.cpp: the GGUF of RE-USE for the Mac (sakasegawa/NVIDIA-NemotronLabs-RE-USE-GGUF) is
# read by speech.cpp 0.10.1 and later only, as its model card says (read on 2026-11-05).
VERSION=v0.10.1
# The revision of the model's repository, and the sha256 of the file as Hugging Face's API listed
# it (its LFS oid), read on 2026-11-05. A file that changes there is not taken without a look.
MODEL_REVISION=117011d06bba5522006f5fcfeadfdb714b58b236
MODEL_FILE=NVIDIA-NemotronLabs-RE-USE-10M-F32.gguf
MODEL_SHA256=827faf209c6988b9cfd07b08ee7f212a7b2eeb7870a25380df671d55338b3879

root="$(cd "$(dirname "$0")/.." && pwd)"
src="${root}/.cache/speech.cpp/src-${VERSION}"
build="${root}/.cache/speech.cpp/build-${VERSION}"

if [[ ! -d "${src}" ]]; then
	mkdir -p "$(dirname "${src}")"
	git clone --depth 1 --branch "${VERSION}" --recurse-submodules --shallow-submodules \
		https://github.com/nyosegawa/speech.cpp.git "${src}"
fi

cmake -S "${src}" -B "${build}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${build}" --config Release -j
ln -sfn "build-${VERSION}" "${root}/.cache/speech.cpp/build"

models="${root}/.cache/models"
target="${models}/${MODEL_FILE}"
sha256_of() {
	shasum -a 256 "$1" | cut -d ' ' -f 1
}
if [[ ! -f "${target}" ]] || [[ "$(sha256_of "${target}")" != "${MODEL_SHA256}" ]]; then
	mkdir -p "${models}"
	curl --fail --location --show-error --progress-bar --output "${target}.partial" \
		"https://huggingface.co/sakasegawa/NVIDIA-NemotronLabs-RE-USE-GGUF/resolve/${MODEL_REVISION}/${MODEL_FILE}"
	actual="$(sha256_of "${target}.partial")"
	if [[ "${actual}" != "${MODEL_SHA256}" ]]; then
		rm -f "${target}.partial"
		echo "${MODEL_FILE}: sha256 ${actual}, not ${MODEL_SHA256}; not kept" >&2
		exit 1
	fi
	mv "${target}.partial" "${target}"
fi
ls "${root}/.cache/speech.cpp/build/speech" "${target}"
