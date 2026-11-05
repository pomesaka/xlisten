//! Links libspeech of speech.cpp, which `mise run speech` builds into .cache/speech.cpp/build.
//! WHY the rpath: the dylib stays in .cache/, where nothing outside the repository is changed.

fn main() {
    let root = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
    let lib = format!("{root}/.cache/speech.cpp/build");
    println!("cargo:rustc-link-search=native={lib}");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{lib}");
    println!("cargo:rerun-if-changed={lib}/libspeech.dylib");
}
