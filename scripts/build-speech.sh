#!/usr/bin/env bash
# Builds speech.cpp (libspeech and `speech`), with Metal, into .cache/speech.cpp/, and fetches the
# GGUF files of the two models xlisten loads into .cache/models/, checking their sha256: NVIDIA's
# RE-USE (speech enhancement) and Nemotron 3 Diarization (who speaks where).
# Nothing is installed outside the repository: delete .cache/ to remove it.
set -euo pipefail

# WHY speech.cpp: the GGUF of RE-USE for the Mac (sakasegawa/NVIDIA-NemotronLabs-RE-USE-GGUF) is
# read by speech.cpp 0.10.1 and later only, as its model card says (read on 2026-11-05).
VERSION=v0.10.1
# For each model: its repository on Hugging Face, the revision, the file and its sha256 as Hugging
# Face's API listed it (its LFS oid), read on 2026-11-05. A file that changes there is not taken
# without a look.
MODELS=(
	"sakasegawa/NVIDIA-NemotronLabs-RE-USE-GGUF 117011d06bba5522006f5fcfeadfdb714b58b236 NVIDIA-NemotronLabs-RE-USE-10M-F32.gguf 827faf209c6988b9cfd07b08ee7f212a7b2eeb7870a25380df671d55338b3879"
	# WHY Q8_0: the smallest of the three (106 MB); speech.cpp's catalog names it the default.
	"sakasegawa/Nemotron-3-Diarization-GGUF 9096d41cb78bba7ac882b96d453e25239eac692b Nemotron-3-Diarization-99M-Q8_0.gguf 336df6350dce2ea5ff06aa6e2b249d331a859ced13305fbb45c44bdb3c0abe20"
)

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
mkdir -p "${models}"
sha256_of() {
	shasum -a 256 "$1" | cut -d ' ' -f 1
}
for entry in "${MODELS[@]}"; do
	read -r repo revision file sha256 <<<"${entry}"
	target="${models}/${file}"
	if [[ -f "${target}" ]] && [[ "$(sha256_of "${target}")" == "${sha256}" ]]; then
		continue
	fi
	curl --fail --location --show-error --progress-bar --output "${target}.partial" \
		"https://huggingface.co/${repo}/resolve/${revision}/${file}"
	actual="$(sha256_of "${target}.partial")"
	if [[ "${actual}" != "${sha256}" ]]; then
		rm -f "${target}.partial"
		echo "${file}: sha256 ${actual}, not ${sha256}; not kept" >&2
		exit 1
	fi
	mv "${target}.partial" "${target}"
done
ls "${root}/.cache/speech.cpp/build/libspeech.dylib" "${models}"
