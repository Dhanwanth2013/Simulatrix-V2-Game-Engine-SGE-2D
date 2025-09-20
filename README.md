# Simulatrix-V2 Single file 2D Simulation Engine (SAT)

**Author:** Dhanwanth

**Description:** A high-performance, Python/Pygame 2D physics simulation engine supporting circles and convex polygons with SAT (Separating Axis Theorem) collision detection. Designed for educational purposes, rapid prototyping, and particle-based simulations.

---

## Features

* ✅ Circle-circle, polygon-polygon collisions (SAT)
* ✅ Semi-implicit Euler & Velocity Verlet integrators (toggle with `V`)
* ✅ Spatial hash broadphase for efficient collision checks
* ✅ Object pooling for short-lived particles
* ✅ Springs/constraints demo
* ✅ Debug HUD and grid overlay
* ✅ Frame recording (PNG sequences)
* ✅ Lightweight, single-file engine (pure Python + Pygame)
* ✅ Runtime hotkeys for quick control

---

## Comparison with Simulatrix-V1
| Feature                        | Old Engine (Circle Only) | New Engine (SAT 2D, Polygons + Circles) |
|--------------------------------|-------------------------|----------------------------------------|
| Shape Support                  | Circles only            | Circles + Polygons                     |
| Collision Detection             | Circle-Circle only      | Circle-Circle + Polygon-Polygon  |
| Physics Integration             | Euler                   | Euler / Velocity Verlet (toggleable)  |
| Broadphase Method               | Spatial Hash            | Spatial Hash (optimized)               |
| Impulse Resolution              | Basic                   | Full impulse resolution                |
| Single-File Engine              | ✅                      | ✅                                     |
| Frame-Friendly Memory Usage     | Moderate                | High (minimal temporaries)             |
| Customizability / Extensibility | Medium                  | High                                   |
| CPU/GPU                         | CPU only                | CPU only                               |
| Use Case                        | Basic 2D sims           | Lightweight 2D sims & complex shapes  |


## Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/2d-sat-engine.git
cd 2d-sat-engine
```

2. Install dependencies:

```bash
pip install pygame
```

3. Run the engine:

```bash
python engine.py
```

---

## Controls / Hotkeys

| Key / Button            | Action                                      |
| ----------------------- | ------------------------------------------- |
| `G` / Grid Button       | Toggle grid overlay                         |
| `H` / HUD Button        | Toggle debug HUD                            |
| `V` / Integrator Button | Toggle between Euler and Verlet integration |
| `P` / Pause Button      | Pause / resume simulation                   |
| `R` / Record Button     | Start / stop recording frames               |
| `+` / `+100` Button     | Spawn 100 additional circles                |
| `-` / `-100` Button     | Remove 100 circles                          |
| `Esc`                   | Quit simulation                             |
| Arrow Up                | Increase gravity                            |
| Arrow Down              | Decrease gravity                            |

---

## Usage

* Circles and polygons are automatically spawned on launch.
* Click buttons at the bottom to toggle features or spawn more objects.
* Physics is frame-rate independent and uses spatial hashing to optimize performance.
* Springs/constraints can be added programmatically through the engine API.

### Example: Spawning a polygon programmatically

```python
# Spawn a triangle
engine.world.spawn_polygon(
    x=300, y=200,
    local_verts=[(0,-18),(16,10),(-16,10)],
    color=(200,120,120),
    mass=4.0
)
```

### Example: Adding a spring between two entities

```python
spring = engine.world.add_spring(entity_a, entity_b, k=0.2, rest=50.0, damping=0.02)
```

---

## Performance Notes

* Spatial hash broadphase is enabled by default for optimal collision checking with large numbers of entities.
* Object pooling avoids repeated allocations for short-lived circles.
* Collision resolution uses SAT for polygons and impulse-based responses.
* Frame profiling displayed in HUD: integration, broadphase, collision resolve, constraints.

---

## Dependencies

* Python 3.8+
* [Pygame](https://www.pygame.org/)

---

## License

This project is released under the **MIT License**. See `LICENSE` file for details.

---

## Screenshots

*(Add your own screenshots here, e.g., particles colliding, polygon demos, HUD, and grid overlay)*

---

## Roadmap / Planned Features

* Add concave polygon support
* Add joint constraints (hinges, rods)
* GPU-accelerated collision detection (via OpenCL / CUDA)
* Interactive editor for placing entities dynamically

---

