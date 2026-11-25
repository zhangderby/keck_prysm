# Keck End-to-End Optical Model Design Notes

*Drafted 2026-11-13 with Claude Opus 5.5 (claude-opus-5-5) from a literature/package survey; decisions confirmed by the project owner. Code architecture added 2026-11-23; document restructured and Shack-Hartmann section added 2026-11-24; atmosphere, LOS sampling, sodium profile and elongation-kernel implementation decisions recorded 2026-11-24 (sections 2.2, 2.3, 4.2, 4.5, 4.6).*

## 1. Code architecture

### 1.1 Scope and target systems

The code must support all three systems through configuration alone (no system-specific code paths):

| System | Telescope | LGS | Launch | Notes |
|---|---|---|---|---|
| Current LGS AO (first target) | Keck I | 1 | Behind secondary (center) | 20x20 Shack-Hartmann |
| KAPA | Keck I | 4 on a 7.6" radius asterism | Behind secondary (center); asterism generator in secondary module | Laser tomography |
| HAKA | Keck II | 1 | Behind secondary (center); verified by project owner 2026-11-24 | ALPAO 2844-actuator DM, finer WFS sampling |

Current scope decisions:
- Diffractive (physical-optics) Shack-Hartmann WFS, not geometric slopes, with the propagation method chosen from the sampling parameters (section 3).
- No control loops yet (no tip-tilt, no LBWFS/truth focus loop, no uplink tip-tilt loop).
- Static sodium profile now; a time-varying profile must be possible later without restructuring.
- Simulation length 1–60 s.
- **End goal: closed loop at about 1 kHz for up to 60 s (about 60,000 steps).**
- **Future WFS:** the model must be able to add a pyramid wavefront sensor (the IWA project on Keck II) later without restructuring.
- **Future science instruments:** the model must support simulations of NIRC2 and OSIRIS.
  - Both are intended for on-sky phase retrieval simulations.
  - NIRC2 is also intended for coronagraphy and dark-hole digging simulations.

### 1.2 Where the time goes

Container choice (dict vs. class) has no measurable effect on speed: attribute or key access costs tens of ns, while one 1024² FFT costs tens of µs on the GPU. At 1 kHz × 60 s (about 60,000 steps), the real costs are:

- **Per-step Python and kernel-launch overhead.** Hundreds of small GPU kernels at about 5–10 µs each can dominate a step. Mitigate by batching over (source, layer, subaperture) and, if needed, CUDA-graph capture (1.5).
- **Allocation inside the loop.** Preallocate buffers; write with `out=` or slice assignment.
- **Host↔device syncs.** No `.get()`, `float(arr)`, or data-dependent Python `if` on device arrays inside the loop.

### 1.3 Data tiers

Data is grouped by how often it changes. Physics lives in module-level pure functions.

| Tier | Contents | Changes | Location |
|---|---|---|---|
| `config` | r0, Cn², wavelengths, LGS asterism, launch position, loop rate, **list of sensors and list of imagers** (kind, sources, wavelength, readout rate, kind-specific parameters), **one control law per actuator** (kind, input measurements, gains, delay) | Never (read from TOML/YAML) | Host |
| `setup` | Grids, pupil/segment masks, LOS coordinates, elongation kernels, FFT plans, preallocated work buffers, **one record per sensor/imager** (its functions and precomputed data), **one record per actuator** (`dm`, `dtt`, `utt`, `focus`: influence functions or modes, rows acted on, law function, reconstructor/control matrix) | Once at init (or on sodium-profile change) | Device |
| `state` | Time, screen offsets/extrusion state, **`dm`** (`actuators` + law memory), **`dtt`** (`command` + law memory), **`utt`** (`command` + law memory), **`focus`** (`command` = LGS WFS focus-stage range + law memory), RNG state. Law memory = integrator state + delay line. | Every step | Device |

### 1.4 Containers and the time step

- **Frozen, slotted dataclasses** (`@dataclass(frozen=True, slots=True)`) as plain data records: no methods, no inheritance. They catch typos (a misspelled field raises instead of silently creating a key) and document the schema; `dataclasses.asdict()` handles serialization. Plain dicts with `typing.TypedDict` are an acceptable alternative.
- `frozen=True` freezes only the bindings, not the arrays. In-place writes into preallocated `setup` buffers are allowed and intended for speed; keep that mutation confined to designated work buffers.
- prysm's own classes (e.g. `Wavefront`) are used as-is.

**Sensors and imagers are chosen by `config`, not hard-coded in `step`.**

- `config` lists any number of sensors (`cfg.sensors`) and imagers (`cfg.imagers`). Each entry gives a `kind` (e.g. `"shack_hartmann"`, `"pyramid"`, `"nirc2"`, `"osiris"`), a name, its source(s), wavelength, readout decimation, and kind-specific parameters.
- At init, `kind` is looked up in a plain registry dict that maps the string to module-level functions:

  ```python
  SENSORS = {"shack_hartmann": (sh_expose, sh_measure),
             "pyramid":        (pyr_expose, pyr_measure)}
  IMAGERS = {"nirc2": nirc2_expose, "osiris": osiris_expose}
  ```

  Each instrument becomes one frozen record in `setup` holding those functions, its source slice, decimation and precomputed data/buffers. All kinds share the same call signature, so adding the pyramid sensor or a coronagraphic NIRC2 mode means adding functions and a registry entry; `step` does not change. No inheritance.
- **The LBWFS is a second `shack_hartmann` entry** (on the NGS, its own lenslet sampling and a large `decim`), not a new sensor kind.
- **Batch inside an instrument, not across entries.** KAPA's four LGS WFS are one `shack_hartmann` entry over four sources, not four entries. The Python loops in `step` then run over only a handful of instruments, and each call is internally batched.
- **Source ordering.** All sources are stacked in one `opd` array of shape `(n_src, ny, nx)`. At setup, sources are ordered so each instrument's sources form one unbroken run of rows, and each instrument stores a `slice`. `opd[x.src]` is then a view: no allocation, no copy. Scattered rows would need index-list ("fancy") indexing such as `opd[[0, 2, 5]]`, which allocates and copies every step and breaks graph capture (1.5).
  - Slices may overlap or be identical. The tip-tilt sensor and the LBWFS always use the same NGS, so both read the same row; the atmosphere is sampled once for that star. Differences between their light paths (non-common-path aberrations, wavelength) are applied inside each sensor's `expose`.
  - A source needs a duplicate row only when the overlap pattern cannot be laid out as unbroken runs, e.g. three instruments that each use a shared star `s` plus one star of their own (`s` can neighbour at most two others). One instrument then gets its own copy of `s`, costing one extra sample per step. None of the planned configurations need this.
  - Example (KAPA): rows `[LGS0, LGS1, LGS2, LGS3, NGS, science]` → LGS WFS `0:4`, tip-tilt sensor and LBWFS `4:5`, NIRC2/OSIRIS `5:6`.
- **Multi-rate.** `decim` is the number of loop steps per readout (1 for the 1 kHz LGS WFS; larger for the LBWFS). The detector frame accumulates in place every step; the measurement is computed only on readout. The readout test uses the host-side integer step counter `k`, so it causes no device sync.
- **Imagers as sensors.** For focal-plane wavefront sensing (dark-hole digging, phase retrieval), an imager kind can also be given a `measure` function and listed under `cfg.sensors`; its output then reaches the control laws like any WFS.
- **Wavelength.** Atmosphere and correction are in OPD (nm), shared by all instruments; each `expose` converts to a prysm `Wavefront` at its own wavelength.

**Actuators and control laws.** Each corrector has its own state record, its own `setup` record and its own control law (inputs and rows below confirmed by project owner 2026-11-24):

| Actuator | State | Rows it acts on | Input |
|---|---|---|---|
| DM | `state.dm.actuators` | All sources | LGS WFS, with **tip and tilt removed** (focus kept) |
| DTT: downlink tip-tilt (fast steering mirror) | `state.dtt.command` | All sources | Tip-tilt sensor on the NGS |
| UTT: laser uplink tip-tilt mirror | `state.utt.command` | LGS rows only | Tilt measured by the LGS WFS |
| LGS WFS focus stage | `state.focus.command` (stage position, expressed as the assumed sodium range) | LGS rows only (read only by the LGS WFS) | Focus measured by the LBWFS (Shack-Hartmann on the NGS) |

- Each record in `setup` stores the rows it acts on as a `slice` (same ordering rule as above), its influence functions or modes, and its control-law function.
- Laws are chosen in `config` through a registry, like sensors (e.g. integrator or leaky integrator now; tomographic or LQG later), and read the named measurements they need from `meas`. Inputs stay configurable even though the table above is the Keck LGS-mode default.
- **Tip/tilt removal costs nothing per step.** It is folded into the LGS WFS reconstructor at setup (reconstructor multiplied by a projector that removes tip and tilt), so the DM law is still one matrix-vector product. The LGS WFS tilt used by the UTT law is a separate small projection of the same slopes.
- A law updates only when its inputs were read out this step (a host-side dict-membership test, no device sync), so each loop runs at its own sensor's rate. Otherwise it returns its state unchanged. The focus stage therefore moves only at the LBWFS rate.
- **Per-loop delay.** Each law has its own `delay` in `config`, in units of base loop steps (1 / `cfg.loop_rate`), because each loop's latency (readout, compute, mirror or stage response) differs.
  - A command computed at step `k` is applied at step `k + delay`. The law pushes each new command into a small preallocated delay line in its state; `state.dm.actuators`, `state.dtt.command`, `state.utt.command` and `state.focus.command` are always the delay line's output, i.e. the command currently on the hardware.
  - The delay line is a shift register updated in place (fixed memory addresses, so it stays graph-capturable, 1.5). It is tiny compared with the wavefront arrays, even for HAKA's 2844 actuators.
  - A fractional delay is applied by linear interpolation between the two nearest stored commands. `delay = 1` reproduces the one-step default.
- Each state record holds the applied command plus that law's own memory (integrator state and delay line).
- **UTT physics:** the UTT command, plus the uplink atmospheric tilt (2.4), sets where the beacon sits on sky. On the downlink this appears as a pure tilt on the LGS rows only; NGS and science rows are unaffected.
- **Focus-stage physics:** the LGS WFS sees a focus term set by the mismatch between the true sodium centroid range and the stage range (2.4). The DM loop keeps focus and nulls that term, so moving the stage changes the focus the DM applies to every source. This is how the NGS-based LBWFS corrects focus errors caused by sodium-altitude changes, which the LGS WFS alone cannot tell apart from atmospheric focus.

The time step reads top to bottom as wavefront transformations. Each correction is added **only to the rows it acts on** (`opd[x.src]`), so, for example, the UTT tilt and the focus-stage term reach only the LGS WFS, while the DTT and DM reach every source:

```python
def step(k, state, setup, screens, cfg):
    """Advance one loop cycle (1 / cfg.loop_rate). k is a host-side int step counter."""
    opd = sample_atmosphere(screens, setup.los_coords, state, cfg)  # (n_src, ny, nx), all sources batched

    # Corrections: in-place adds into the rows each actuator acts on (helpers write into preallocated buffers)
    opd[setup.utt.src]   += tilt(state.utt.command, setup.utt)              # LGS rows only: beacon pointing on sky
    opd[setup.dtt.src]   += tilt(state.dtt.command, setup.dtt)              # all rows: downlink fast steering mirror
    opd[setup.dm.src]    += dm_surface(state.dm.actuators, setup.dm)        # all rows: deformable mirror
    opd[setup.focus.src] += focus_term(state.focus.command, setup.focus)    # LGS rows only: range mismatch -> focus

    meas = {}
    for s in setup.sensors:                                         # from cfg.sensors: LGS SH, TT, LBWFS (SH)
        s.expose(opd[s.src], s)                                     # accumulate detector frame in place
        if (k + 1) % s.decim == 0:                                  # readout step (host int, no sync)
            meas[s.name] = s.measure(s)                             # slopes/signals; resets frame

    for im in setup.imagers:                                        # from cfg.imagers, e.g. NIRC2, OSIRIS
        im.expose(opd[im.src], im)                                  # accumulate science frame in place

    return replace(state,                                           # separate law per actuator; each pushes
        t     = state.t + cfg.dt,                                   # its new command into its own delay line
        dm    = setup.dm.law(meas, state.dm, setup.dm),             # LGS WFS (tip/tilt removed) -> DM
        dtt   = setup.dtt.law(meas, state.dtt, setup.dtt),          # TT sensor -> DTT
        utt   = setup.utt.law(meas, state.utt, setup.utt),          # LGS WFS tilt -> UTT
        focus = setup.focus.law(meas, state.focus, setup.focus),    # LBWFS focus -> LGS WFS focus stage
    )
```

### 1.5 CUDA-graph capture (candidate optimization)

CuPy can record the step into a CUDA graph with `Stream.begin_capture()` / `end_capture()` and replay it with one launch per step, removing per-kernel launch overhead. Constraints that shape the code now:

- Capture requires a non-default stream; synchronous device↔host transfers are not allowed during capture.
- A graph replays fixed memory addresses, so every per-step array must be **preallocated and written in place**, and time-varying scalars (t, wind shifts, gains) must live in **device arrays** rather than Python floats.
- Run one warm-up step before capture so cuFFT plans and memory-pool blocks already exist.
- Python-side branching (e.g. switching screen providers, recomputing LGS kernels) happens outside the graph, between replays.
- Multi-rate readouts change which kernels run, so capture one graph per distinct readout pattern (e.g. "every step" and "every step + LBWFS readout") and pick the graph in Python from `k`.
- The CPU backend (NumPy) runs the same `step` without capture.

Writing the step in this "preallocated, in-place, no host sync" style from the start keeps graph capture a drop-in option rather than a rewrite.

### 1.6 Telemetry

Write per-step outputs (slopes, DM/DTT/UTT/focus-stage commands, Strehl, PSF accumulations) into preallocated device ring buffers; copy to host and to disk in chunks (e.g. every 1,000 steps), never per step.

## 2. LGS implementation

### 2.1 Standard three-stage decomposition

1. **Uplink → spot at the sodium layer.** Gaussian kernel of given FWHM (default). Full uplink wave optics is not needed unless studying spot size or photon return.
2. **Beacon.** Sodium layer as thin sub-layers weighted by the density profile, including geometric dilution and two-way transmission (Brown et al. 2004, eq. 3). Three sub-layers already captured the 3D beacon well in that study.
3. **Downlink.** Trace turbulence **once per LGS at the sodium centroid range**. The differential cone effect across the sodium thickness is negligible at Keck subaperture sizes (Brown et al. 2004). The sodium thickness enters only through the elongation kernel (MDFT method) or the sub-layer focus terms (Fresnel method); see 3.4.

### 2.2 Elongated spots

- Elongation direction and length per subaperture are computed from the **launch position** (a per-system parameter) and the sodium profile.
- All three target systems launch from behind the secondary, so elongation is radial about the pupil center and grows with subaperture distance from it. Keep launch position as a parameter anyway (costs nothing; allows side-launch studies).
- Kernel construction following COMPASS (`shesha/init/lgs_init.py`): convolve the 1D sodium profile with a 1D Gaussian along the elongation axis, multiply by a Gaussian across it, rotate to the subaperture azimuth.
  - *Implemented 2026-11-24 (`lgs.elongation_kernels`):* the same kernel, evaluated directly on the grid as the photon-weighted sum of isotropic Gaussian uplink spots at each sodium sample's angle b·(1/R_c − 1/R). It is separable, so it reduces to one batched matrix multiply, with no rotation or interpolation. Kernels sum to 1 and are centred on the sodium centroid (2.3). One kernel set serves all LGS that share a launch position.
- Precompute kernels; apply as batched FFT convolutions with the subaperture PSFs. This applies to the per-subaperture MDFT method (3.3). The Fresnel method needs a different treatment because the elongation varies across the focal plane (3.4).

### 2.3 Sodium profile interface

- Represent the profile as arrays `(altitudes_m, densities)`.
- A future time-varying profile is a function `profile(t)` returning the same arrays; kernels are recomputed (batched, on GPU) only when the profile changes. MAOS replays measured profiles with linear interpolation in time (Wang & Ellerbroek).
- **Default (confirmed by project owner 2026-11-24):** a single Gaussian at 90 km above sea level with 10 km FWHM (`lgs.sodium_profile`, which also takes a sum of Gaussians or measured arrays). No Maunakea-specific peer-reviewed profile was found. The default follows long-term lidar statistics (mean centroid 90.8 km; Pfrommer & Hickson 2014, near Vancouver). Site altitude is Keck, 4145 m.
- **Centroid definition:** photon weights w ∝ n(h)/h² (geometric dilution, Brown et al. 2004). The centroid height H_c is set by 1/H_c = Σ w/h, because defocus goes as 1/H. With the default profile, H_c ≈ 85.2 km above Keck. This H_c is the LGS height used for line-of-sight sampling (2.1) and the true range behind the focus-stage term (2.4), and the elongation kernels are centred on it, so the kernels add no extra focus.

### 2.4 Focus and tip-tilt (structure now, loops later)

- **Root cause of lost LGS focus:** changes in the sodium centroid altitude are indistinguishable from atmospheric focus (Pfrommer & Hickson; Herriot et al. 2006). The 3D structure separately causes quasi-static "LGS aberrations" (Clare et al.; up to 1200 nm RMS on Keck II).
- Keep the **true sodium centroid range** separate from the **WFS focus-stage (assumed) range**; treat the latter as state (`state.focus.command`), driven by the LBWFS focus loop (1.4). The difference appears as focus in the diffractive WFS images automatically.
- Future additions, each as a separate function: tip-tilt (and differential focus) projection out of LGS measurements (Gilles & Ellerbroek); LBWFS reference-centroid updates, as on Keck; uplink tip-tilt, sampled from the same shared screens over the launch-telescope footprint.

### 2.5 Multi-LGS

Vectorize everything over an LGS index so one LGS (Keck I, HAKA) and four LGS (KAPA) run through the same code path.

## 3. Shack-Hartmann WFS modeling

*prysm functions named here were checked against the prysm 0.22 source (GitHub master, 2026-07-25). prysm units: pupil coordinates and focal lengths in mm, wavelength and focal-plane coordinates in µm.*

### 3.1 Two propagation methods, chosen from the sampling parameters

| Method | Use when | How | prysm functions |
|---|---|---|---|
| `"mdft"` (per subaperture) | Crosstalk between subapertures is negligible | Cut the pupil field into subapertures; propagate each to its own detector window with a matrix DFT | `prysm.propagation.dft.prepare_executor(..., kind="mdft")` |
| `"fresnel"` (full lenslet array) | Crosstalk matters | Multiply the full pupil field by a lenslet-array phase screen; propagate one lenslet focal length by angular spectrum | `prysm.x.shack_hartmann.shack_hartmann`, `prysm.propagation.angular_spectrum_transfer_function` |

- Each `shack_hartmann` entry in `config` sets `method = "auto" | "mdft" | "fresnel"`, the physical lenslet pitch and focal length, pixels per subaperture, pixel scale, oversampling factor and a crosstalk tolerance.
- The choice is made **once at setup**. The `shack_hartmann` builder returns a record whose `expose` is either `sh_expose_mdft` or `sh_expose_fresnel`. Both have the same signature, so `step` is unchanged. `measure` (centroiding) is shared by both methods and is independent of how the spots were formed.
- Both methods hand the same thing to `measure`: a detector frame of shape `(n_src, n_pix, n_pix)` (or per-subaperture windows), so centroiding, noise and telemetry code is written once.

### 3.2 When crosstalk matters

- The key parameter is the **lenslet Fresnel number** N_F = p² / (λ f), with p the lenslet pitch and f its focal length. It is unchanged by pupil magnification, so compute it from the physical lenslet. The detector window under one lenslet (width p) spans N_F diffraction widths (λ f / p).
- Light spills into neighbouring windows when the spot is not small compared with that window. Contributors to spot size:
  - diffraction, including the slowly decaying sinc² wings of a square lenslet;
  - seeing within the subaperture, about λ / r0 when d > r0;
  - LGS elongation, up to about (D/2)·Δh / H² at the pupil edge;
  - the uplink spot size;
  - large tip/tilt excursions (e.g. before the loop closes).
- Diffraction-only sanity check: for a centred spot behind a square lenslet, the fraction of energy outside the window along one axis is about 2 / (π² N_F) (about 1% at N_F = 20).
- **`"auto"` selection (setup only):** run the MDFT method with each window enlarged to 3×3 subaperture widths, for the worst-case subapertures (pupil edge, maximum elongation) and a sample turbulent screen at the configured r0. Measure the fraction of each subaperture's energy that lands outside its own window. Choose `"fresnel"` if it exceeds the crosstalk tolerance, otherwise `"mdft"`, and log the value. This reuses the MDFT code and captures all contributors at once.

### 3.3 MDFT method (no crosstalk)

Per sensor, batched over all sources and subapertures:

1. **Subaperture extraction.** Choose the pupil grid so each subaperture has an integer number of samples m and subaperture edges fall on sample boundaries. A `(n_src, ny, nx)` OPD then reshapes to `(n_src, n_y, m, n_x, m)` and is copied once into a preallocated `(n_src, n_sub, m, m)` buffer. Compute every subaperture in the square grid, including ones outside the pupil, and mask afterwards; this avoids a gather (fancy indexing), and the extra cost is small (400 vs. about 300 valid subapertures on the 20×20).
2. **Field.** Precomputed amplitude (pupil × subaperture mask) × exp(i 2π OPD / λ).
3. **Focus.** One prysm MDFT executor per (wavelength, sampling), shared by all subapertures. prysm's `MDFT` applies `Ey @ U @ Ex.T`, which broadcasts over leading axes, so the whole `(n_src, n_sub, m, m)` stack goes through in one call. This was checked with prysm 0.22: the batched result matches a per-subaperture loop. The output window is (pixels per subaperture × oversampling) samples at (pixel scale / oversampling).
   - Angle-unit convenience: with `efl = 206.265` mm, 1 µm in prysm's focal plane equals 1 arcsec on sky.
   - Light falling outside the window is dropped, not wrapped. That is the no-crosstalk approximation.
4. **Intensity**, |E|².
5. **LGS only:** convolve each subaperture's spot with its precomputed elongation kernel (2.2), as a batched FFT convolution on the oversampled grid.
6. **Detector:** bin the oversampled grid to pixels (reshape and sum), then add photon and read noise.

Cost is dominated by two batched matrix multiplies per step, which is small.

### 3.4 Fresnel method (crosstalk)

1. **Lenslet screen.** `prysm.x.shack_hartmann.shack_hartmann(pitch, n, efl, wavelength, x, y)` returns the complex screen exp(−i 2π r² / (2 f λ)) for each lenslet. Build it once per wavelength at setup; it loops over lenslets in Python, which is fine outside the hot loop. In pupil-referred coordinates use the pupil-referred pitch d and focal length f′ = f·(d/p)², which keeps N_F the same.
2. **Field.** E = amplitude × exp(i 2π OPD / λ) × screen over the full pupil, shape `(n_src, N, N)`.
3. **Propagate a distance f′** by angular spectrum, using a transfer function precomputed with `angular_spectrum_transfer_function`: `ifft2(fft2(E) · tf)`, batched over leading axes. Pad by a guard band of a few lenslets (not necessarily 2×) so light leaving the edge of the array does not wrap onto the opposite side.
4. **Intensity**, crop to the detector, bin to pixels, add noise.

**Sampling constraints.** The angular spectrum method keeps the sample spacing, so the focal-plane sampling equals the pupil sampling. With q samples per lenslet:

- lenslet phase sampled at Nyquist: q ≥ N_F (prysm's docstring recommends at least 32, and 64–128 for strongly aberrated beams);
- spot sampled at Nyquist (dx ≤ λ f / 2p): q ≥ 2 N_F;
- detector pixels: q must be an integer multiple of the pixels per subaperture;
- angular-spectrum transfer function not aliased (Voelz & Roggemann 2009): dx ≥ λ f / L for padded grid width L, i.e. q ≤ (L / p)·N_F.

**LGS elongation.** Elongation varies across subapertures, so it cannot be one convolution of the whole focal plane. Instead, model the sodium layer as N_L thin sub-layers (Brown et al. 2004). Each sub-layer j is a point source at range H_j, which adds a focus term ∝ (1/H_stage − 1/H_j) to the LGS rows, centred on the launch position (pure focus for centre launch; focus plus tilt for side launch). Propagate each sub-layer, then sum the intensities weighted by sodium density. Elongation, and the crosstalk it causes, then come out of the physics. Finish with one convolution by the uplink spot, which is the same angular kernel for every subaperture.

- Enough sub-layers are needed that adjacent images at the pupil edge are offset by no more than about λ / 2d: N_L ≳ elongation / (λ / 2d). Rough Keck estimate: elongation ≈ (5 m × 10 km) / (90 km)² ≈ 1.3″ against λ / 2d ≈ 0.11″ (589 nm, 0.56 m subapertures), so about 12 sub-layers. Brown et al. found 3 adequate for their system.
- All (LGS, sub-layer) fields go through as one batched FFT stack, so cost scales with N_L.

**Cross-check.** Where crosstalk is negligible, the two methods must agree on spot images and centroids. This is the main validation test for both.

### 3.5 Performance

- MDFT method: two batched matrix multiplies per sensor per step; negligible next to the atmosphere.
- Fresnel method example: 20 subapertures × 64 samples = 1280 samples across, padded to about 1400²; complex64 ≈ 16 MB per field. Four LGS × 12 sub-layers ≈ 0.75 GB, and 48 forward + inverse FFTs per step. Benchmark before committing.
- Graph capture: prysm's `MDFT.__call__` and `angular_spectrum` allocate their outputs. Hot-path versions should use `cupy.matmul(..., out=)` and cuFFT plans writing into preallocated buffers, translated from prysm and citing it (open item 6).
- One `opd` grid shared by all instruments (1.4) may be coarser than the Fresnel method's q requires. Either sample the whole stack at the finest grid any instrument needs, or sample the atmosphere separately per sampling group (open item 9).

## 4. Atmosphere

### 4.1 One shared atmosphere for every source

All sources (LGS, NGS, science) sample the **same** set of frozen-flow layers. This is required for correctness: cone effect, anisoplanatism and tomography only emerge when sources share turbulence. It is also the cheapest option, since screens are generated once and each source costs only an interpolation per layer. MAOS, COMPASS, OOMAO and OOPAO all use this design.

### 4.2 Geometric line-of-sight sampling

For source *k* (direction θ_k, range H_k; H = ∞ for an NGS) and layer *l* (height h_l, wind v_l):

φ_k(x, t) = Σ_l φ_l( x·(1 − h_l/H_k) + h_l·θ_k − v_l·t )

- Scale h and H by 1/cos z for zenith angle z; scale r0 by (cos z)^(3/5).
- The (1 − h/H) factor is the cone effect.
- Coordinates depend only on fixed geometry, so they are **precomputed once**; each time step adds only the wind shift.
- **Separable (confirmed by project owner 2026-11-24).** Layers stay aligned with the pupil-grid axes, and the map is a scale plus a shift, so the coordinates are stored as one x vector and one y vector per (source, layer) (`atmosphere.los_coordinates`). Wind direction is arbitrary and enters only as the shift. This rules out per-source pupil rotation, which current scope does not need.
- Heights are given above the telescope at zenith; the zenith angle is applied inside `los_coordinates`.

### 4.3 Units

Store screens as **OPD in nm**, not radians. The same screens then serve 589 nm LGS, visible/IR NGS and science wavelengths, and match prysm's convention of phase in nm (`Wavefront.from_amp_and_phase`).

### 4.4 Validity

Geometric (thin-screen, no scintillation) propagation is valid when the Fresnel scale √(λh) is much smaller than a subaperture: about 8 cm at 589 nm and 10 km, versus about 50 cm subapertures on the current Keck 20x20 WFS. **Re-check for HAKA's finer sampling.** If scintillation is ever needed, keep the shared screens but replace the line-of-sight sum with per-source Fresnel propagation (prysm `angular_spectrum`), at much higher cost per source.

### 4.5 Screen providers (swappable)

Screen generation sits behind a small functional interface so the provider can be chosen per run.

1. **FFT screens (default, runs of a few seconds).** `atmosphere.fft_layers` / `von_karman_screen`. *Departure from the original plan (confirmed by project owner 2026-11-24):* the prysm functions are not used.
   - The local prysm `abc_psd` is a/(1+(ν/b)^c), not a/(1+(ν/b)²)^(c/2), so it cannot produce von Kármán turbulence.
   - `synthesize_surface_from_psd` scales correctly (it gives Σ√(PSD·Δf²)), but every Fourier mode has a fixed amplitude, so the statistics are wrong for turbulence. It also uses an unseeded global RNG.
   - The project generator uses complex-Gaussian coefficients with the exact von Kármán constant (0.0229), a seeded `np.random.default_rng` through `prysm.mathops`, and prysm `fftfreq`/`fftrange` (Johansson & Gavel 1994; Schmidt 2010 listing 9.2).
   - Periodic by default; wrap-around shifts are fine while the screen is larger than the meta-pupil.
   - FFT screens under-represent low spatial frequencies (including tip-tilt). Subharmonics (Lane et al. 1992; Schmidt 2010 listing 9.3, via aotools `ft_sh_phase_screen`) are **opt-in**, because they make the screen non-periodic. With subharmonics on, the footprint plus the full wind travel must stay on the screen; `atmosphere.check_coverage` enforces this.
   - Checked against theory (100 screens, 256², L0 = 25 m): the structure function is 0.97–0.99 of theory from 4 to 64 px with subharmonics, and 0.82 at 64 px without them. At 1 px it reads about 0.9 either way, because the FFT method cuts off power above Nyquist.
2. **Infinite screens (long runs).** `atmosphere.infinite_layers` / `extrude_layers`: Assémat & Wilson (2006) von Kármán extrusion, translated from aotools `PhaseScreenVonKarman` (stencil of 2 full lines). Confirmed by project owner 2026-11-24:
   - Extrudes along both x and y, as COMPASS does, so any wind direction works on an axis-aligned screen (keeping the sampling separable, 4.2). By isotropy, one (A, B) pair serves all four sides. A does not depend on r0; B is built for r0 = 1 m and scaled per layer, so all layers with a common L0 and dx share the operators.
   - Each layer is an n×n **ring buffer**: logical pixel u is stored at u mod n, so extruding a line overwrites the oldest line and nothing is copied. The sampler uses the same modulo addressing as for periodic FFT screens.
   - The window follows w = floor(−v t / dx). The number of lines to add each step is computed from host-side t (no device sync). Lines are added in rounds batched over layers. This runs outside a captured CUDA graph (1.5).
   - **Block size (`block`, default 1; added 2026-11-24 at project owner's request).** Each extrusion draws `block` lines jointly, conditioned on the stencil. B is (block·n)², so per-line work grows with block, while the number of rounds (about 20 small array operations plus Python each) falls as 1/block.
     - CPU is limited by memory bandwidth in the matrix multiplies, so it keeps block = 1. Apple M4, n = 1024, 7 layers: 0.38 ms per line at 1, 0.74 at 4, 3.0 at 16.
     - GPU is limited by kernel launches, so it should use block of about 4–16, set from the RTX 5090 benchmark (open item 7).
     - The window stays within block − 1 lines of the wind shift, so the footprint needs that much extra margin; `check_coverage` enforces it.
   - Initial screens come from the FFT generator with subharmonics. Checked: the structure function is 1.01–1.03 of theory after extruding more than 2 screen widths on both axes. Frozen flow is exact through 400 steps (more than 3 screen widths).
   - Fried & Clark (2008) (aotools `PhaseScreenKolmogorov`) was not adopted because it has no outer scale.

Sizing example: at about 2 cm pupil sampling, a 20 m/s layer moves 1.2 km (about 60,000 pixels) in 60 s. A square FFT screen that size is impractical, which is why the infinite provider is needed for longer runs.

### 4.6 Performance guidelines

- One batched bilinear gather over (source, layer) on the GPU per time step.
  - *Implemented (`atmosphere.sample_atmosphere`, resolves open item 4):* the corner indices and weights of all four bilinear corners are outer sums and products of the separable per-axis vectors. Everything is gathered with one `np.take` over the flattened layer stack into preallocated buffers (`make_sample_buffers`), then summed over corners and layers. There are no per-step allocations and no host syncs. The wind shift is computed from device-side t. Checked against `ndimage.map_coordinates(order=1, mode="grid-wrap")`: max error 0.005 nm on 886 nm RMS.
  - Cost is about 40 passes of memory traffic over (n_src, n_layer, ny, nx). A fused CuPy kernel would cut this about 10× if the benchmark (open item 5) needs it. It would be the first CuPy-specific code, so it needs a decision first.
- float32 throughout unless a test shows otherwise.
- All array operations through prysm's backend shim (`prysm.mathops`), so `set_backend_to_cupy()` / `set_backend_to_defaults()` swap GPU and CPU.
- Memory: an 8192² float32 screen is about 256 MB; a 5–9 layer profile fits comfortably in the RTX 5090's 32 GB.

## 5. Open items

1. ~~Verify `synthesize_surface_from_psd` normalization and units.~~ Resolved 2026-11-24: the scaling is correct but the statistics are wrong (fixed amplitudes); replaced by the project's own generator (4.5).
2. Choose Mauna Kea Cn² profiles (e.g. those used in the KOLA MAOS study, which was validated against Keck I on-sky data). Project owner will provide KAON data (noted 2026-11-24). r0, L0 and the uplink spot FWHM come from config files; the code sets no defaults for them.
3. Confirm HAKA WFS order and re-check the Fresnel-scale validity.
4. ~~Decide the interpolation primitive.~~ Resolved 2026-11-24: separable bilinear gather (4.2, 4.6). Still to do: an RTX 5090 benchmark, and whether a fused kernel is needed.
5. Set a wall-clock budget per step (e.g. a 60 s, 1 kHz run in X minutes) to judge optimizations against.
6. Check whether prysm's propagation functions allocate new arrays internally (blocks in-place/graph capture); wrap or translate the hot-path ones if so. Confirmed for `MDFT.__call__` and `angular_spectrum` (3.5).
7. Benchmark CUDA-graph replay vs. eager execution of `step` on the RTX 5090.
8. Define the LBWFS lenslet sampling and readout rate, and the focus-stage law (gain, delay, stage units and range).
9. Decide how pupil sampling is shared when instruments need different grids (one finest grid vs. per-group atmosphere sampling), mainly driven by the Fresnel SH method (3.5).
10. Collect lenslet pitch, focal length (hence N_F), pixels per subaperture and pixel scale for each Shack-Hartmann (LGS WFS, LBWFS; HAKA later).
11. Choose the crosstalk metric and tolerance for `"auto"` (energy fraction outside the window vs. centroid bias) and the number of sodium sub-layers for the Fresnel method; validate both methods against each other where crosstalk is negligible.

## 6. References

- Brown, Fedrigo & van der Werf 2004, *Simulations of AO with an LGS for SINFONI*, Proc. SPIE. https://arxiv.org/abs/astro-ph/0407445
- Conan & Correia 2014, *Object-oriented Matlab adaptive optics toolbox*, Proc. SPIE 9148, 91486C. https://github.com/cmcorreia/oomao
- OOPAO (Python port of OOMAO). https://github.com/cheritier/OOPAO
- COMPASS (GPU AO simulation), LGS initialization. https://lesia.obspm.fr/compass/latest/namespaceshesha_1_1init_1_1lgs__init.html
- MAOS (Wang & Ellerbroek). https://github.com/lianqiw/maos
- KOLA performance simulations with MAOS, validated on Keck I. https://arxiv.org/abs/2511.17488
- Assémat, Wilson & Gendron 2006, *Method for simulating infinitely long and non stationary phase screens with optimized memory storage*, Opt. Express 14, 988. https://doi.org/10.1364/OE.14.000988 · Fried & Clark 2008, extruding Kolmogorov screens. Implementations: https://aotools.readthedocs.io/en/latest/turbulence.html
- Schmidt 2010, *Numerical Simulation of Optical Wave Propagation with Examples in MATLAB*, SPIE Press (FFT screens with subharmonics, listings 9.2–9.3).
- Sasiela 1994, *Electromagnetic Wave Propagation in Turbulence*, Springer (von Kármán PSD and phase covariance).
- Pfrommer & Hickson 2014, *High resolution mesospheric sodium properties for adaptive optics applications*, A&A 565, A102. https://doi.org/10.1051/0004-6361/201423460
- Lane, Glindemann & Dainty 1992; Johansson & Gavel 1994 (FFT screen subharmonics).
- Tyler 1994 (cone-effect variance); Tallon & Foy 1990 (multiple LGS / tomography).
- Ellerbroek 2002, efficient tomographic reconstruction (sparse Laplacian prior).
- Ellerbroek 2014, *Strategies to cope with sodium layer profile variations*, Proc. SPIE 9148. https://core.ac.uk/download/216288144.pdf
- Herriot et al. 2006, *Focus errors from tracking sodium layer altitude variations*, Proc. SPIE 6272. https://circle.ubc.ca/handle/2429/37479
- Pfrommer & Hickson 2008–2014, sodium lidar studies. https://www.astro.ubc.ca/LMT/lzt/lidar/papers/pfrommer2008.pdf
- Davis, Hickson & Herriot, *Temporal variability of the telluric sodium layer*. https://arxiv.org/abs/astro-ph/0609307
- Clare, van Dam & Bouchez 2007, *Modeling LGS aberrations* (KAON 477). https://www.oir.caltech.edu/twiki_oir/pub/Keck/NGAO/NewKAONs/KAON477_Clare07.pdf
- van Dam et al. 2006, *Keck LGS AO: performance characterization*, PASP 118, 310. https://www2.keck.hawaii.edu/inst/ao/aodocs/MvDetal2006PASP118_310.pdf
- Gilles & Ellerbroek 2008, *Split atmospheric tomography*, JOSA A 25, 2427.
- Wang & Ellerbroek, *Optimizing LGS WFS pixel processing with evolving turbulence and sodium profile*. https://escholarship.org/content/qt8ft440wn/qt8ft440wn.pdf
- KAPA: on-sky performance https://arxiv.org/abs/2608.07769 · asterism generator https://par.nsf.gov/servlets/purl/10357849 · daytime calibration https://par.nsf.gov/servlets/purl/10357047
- Soummer, Pueyo, Sivaramakrishnan & Vanderbei 2007, *Fast computation of Lyot-style coronagraph propagation*, Opt. Express 15, 15935 (matrix Fourier transform). https://doi.org/10.1364/OE.15.015935
- Voelz & Roggemann 2009, *Digital simulation of scalar optical diffraction: revisiting chirp function sampling criteria and consequences*, Appl. Opt. 48, 6132 (angular-spectrum sampling). https://doi.org/10.1364/AO.48.006132
- HCIPy (Por et al. 2018, Proc. SPIE 10703), Shack-Hartmann modeled as a micro-lens array followed by Fresnel propagation. https://github.com/ehpor/hcipy
- prysm source: `prysm/x/shack_hartmann.py`, `prysm/propagation/dft.py`, `prysm/propagation/angular_spectrum.py`, `prysm/fttools.py`. https://github.com/brandondube/prysm
- prysm API. https://prysm.readthedocs.io/en/latest/api/index.html
- CuPy CUDA-graph capture (`cupy.cuda.Stream.begin_capture`). https://docs.cupy.dev/en/latest/reference/generated/cupy.cuda.Stream.html
- NVIDIA CUDA Programming Guide, CUDA Graphs / stream capture. https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html

## 7. CLAUDE.md checklist

The repo's CLAUDE.md should cover at least:

- [ ] Ask before implementing any change.
- [ ] Use prysm first; if a feature is missing, check other packages and translate to prysm-compatible functions.
- [ ] Cite the original source of any translated code (file header or docstring).
- [ ] AI-use disclaimer naming the Claude model, in every generated file or function.
- [ ] Functional style; classes only when useful, no inheritance.
- [ ] Data in frozen, slotted dataclasses split into `config` / `setup` / `state`; logic in pure functions.
- [ ] Propagation written as a readable sequence of wavefront transformations.
- [ ] Hot loop: no allocation, no host↔device sync, time-varying scalars on device (keeps CUDA-graph capture possible).
- [ ] All array math through `prysm.mathops` so CPU/GPU backends swap; float32 default.
- [ ] Speed is a primary requirement; target 1 kHz × 60 s closed loop on RTX 5090 / Ryzen 9 9950X3D / 32 GB RAM.
- [ ] Prefer peer-reviewed references for nontrivial methods.
- [ ] Code must support Keck I single LGS, KAPA four LGS and Keck II HAKA via configuration.
- [ ] Pointer to this design-notes document.
