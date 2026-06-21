// build.rs — Compile the C bridge code for Windows cross-compilation.
//
// On macOS and Linux the C bridge is compiled separately via the Makefile.
// On Windows (cross-compilation from macOS via cargo-xwin), we compile
// stata_bridge.c and stplugin.c here so they are linked into the cdylib.

fn main() {
    // Only compile C bridge when targeting Windows
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "windows" {
        return;
    }

    let plugin_dir = std::path::Path::new("../plugin");

    cc::Build::new()
        .file(plugin_dir.join("stata_bridge.c"))
        .file(plugin_dir.join("stplugin.c"))
        .include(plugin_dir)
        .define("SYSTEM", "4") // STWIN32
        .warnings(false)
        .opt_level(2)
        .compile("stata_bridge");

    // Force linker to include the Stata entry point symbols
    // (they are __declspec(dllexport) in C but not referenced by Rust code)
    println!("cargo:rustc-link-arg-cdylib=/INCLUDE:stata_call");
    println!("cargo:rustc-link-arg-cdylib=/INCLUDE:pginit");
}
