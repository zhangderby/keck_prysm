## Research and communication
- Provide references (preferably peer-reviewed) for methods already solved in the literature.
- Be concise. Ask clarifying questions on complex or ambiguous problems before writing code.

## Implementation 
- Do not make changes without explicitly asking first. 
- For any and all code you are asked to generate, provide an AI use disclaimer that states which Claude model was used. This should be done at the top of a file of within a docstring depending on context.
- The preferred package for physical optics propagation is [`prysm`](https://github.com/brandondube/prysm/). Make use of the package and its convenience functions where possible.
- Route all array math through `prysm.mathops` (`from prysm.mathops import np`) so
  `set_backend_to_cupy()` / `set_backend_to_defaults()` swap GPU and CPU. Never import numpy/cupy directly in model code.
- Plain Python scalar math (config-derived constants, zenith/airmass factors, host-side scheduling such as extrusion line counts) may use the standard-library `math` module. This keeps scalars on the host, avoiding 0-d device arrays, kernel launches and device→host syncs. Never use `math` on arrays.
- Default to float32 / complex64 using `prysm.conf` unless a test shows more precision is needed.
- If `prysm` does not have a feature implemented, check other optical modeling packages instead. If others contain said feature, translate it to be compatible with `prysm`. 
- If translating code from another package, provide a reference to the original source. This can be done at the top of a file or within a docstring depending on context.
- This model will heavily emphasize speed and computational efficiency. The model will typically be run on GPU using a GeForce RTX 5090 for running this model on GPU or an AMD Ryzen 9 9950X3D 16-Core processor and 32 Gb of RAM for running this model on CPU.
- Functional programming is preferred. Classes are allowed but class inheritance is to be avoided unless necessary. 
- Model code should be easily readable as a series of transformations on the wavefront as it propagates through the system.
- Code must support Keck I single-LGS, KAPA (Keck I, 4 LGS) and HAKA (Keck II, single LGS)
  through configuration only — no system-specific code paths. Vectorize over an LGS/source index.

## Design documents
Read the relevant doc before designing or changing components:
- `docs/design/design_notes.md` — shack-hartmann WFS modeling, atmosphere, LGS source model, focus/tip-tilt handling, open items and references.
If an implementation decision departs from a design doc, say so and ask before proceeding. If implementation proceeds with the departure, update the design doc to reflect the changes.
