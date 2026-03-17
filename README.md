https://discord.gg/ucBA55bHPR

# POMC

A Minecraft client written in Rust from scratch. POMC connects to vanilla Minecraft servers, renders the world using Vulkan, and handles player physics — all without relying on Mojang's Java codebase.

## Why POMC?

- **Performance** — Native code with zero garbage collection pauses. GPU rendering through Vulkan via ash.
- **Low memory footprint** — No JVM overhead. Runs comfortably on hardware that struggles with vanilla Java Edition.
- **Cross-platform** — Builds natively on Windows, Linux, and macOS from a single codebase.
- **Hackable** — Clean, modular Rust codebase. Easy to understand, modify, and extend.

## Current Status

POMC is in active early development, working through milestones toward a fully playable client.

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Window + GPU initialization | Done |
| 2 | Camera movement + basic rendering | Done |
| 3 | Server connection + protocol handling | Done |
| 4 | Terrain rendering with textures | Done |
| 5 | Player physics + collision | Done |
| 6 | HUD, chat, inventory | Done |
| 7 | Main menu, server list, settings | Done |
| 8 | Vulkan renderer (ash + gpu-allocator) | Done |
| 9 | GPU-driven rendering | Planned |

## Building

Requires the [Vulkan SDK](https://vulkan.lunarg.com/) and a Rust toolchain.

```bash
cargo build --release
```

Assets are automatically downloaded on first run — no manual setup needed.

## Running

```bash
# Launch the client (opens main menu)
cargo run --release

# Connect directly to a server
cargo run --release -- --server localhost:25565 --username Steve

# With authentication (for online-mode servers)
cargo run --release -- \
  --server mc.example.com \
  --username Player \
  --uuid <your-uuid> \
  --access-token <your-token>
```

### Optional flags

| Flag | Description |
|------|-------------|
| `--server <host:port>` | Connect directly to a server |
| `--username <name>` | Player name (default: Steve) |
| `--uuid <uuid>` | Player UUID for authenticated sessions |
| `--access-token <token>` | Minecraft auth token |
| `--game-dir <path>` | Override data directory |
| `--assets-dir <path>` | Override assets directory |
| `--version <ver>` | MC version for asset download (default: 1.21.11) |

## Tech Stack

| Component | Crate |
|-----------|-------|
| Vulkan bindings | `ash` |
| GPU memory | `gpu-allocator` |
| Windowing | `winit` |
| Math | `glam` |
| Protocol | `azalea-protocol` |
| Async runtime | `tokio` |
| Textures | `png`, `image` |
| Parallel meshing | `rayon` |
| Font rendering | `fontdue` |
| Shader compilation | `shaderc` |

## License

GPL-3.0-or-later. This project is not affiliated with Mojang or Microsoft.
