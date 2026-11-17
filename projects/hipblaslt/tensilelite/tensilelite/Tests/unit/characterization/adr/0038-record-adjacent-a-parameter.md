# ADR 0038: Record the adjacent A layout parameter

Status: Accepted

## Context

`WaveTileAdjacent` adds an optional A-row layout for gfx1250 kernels. Its
accepted values are `[False, True]`, with `False` as the default. The parameter
roster, solution state and serialized defaults include this new public setting.

## Decision

Update the affected nodes in three snapshot files:

- `ValidParameters/test_builders_char.ambr`: add the name and Boolean domain.
- `SolutionClass/test_solution_class_char.ambr`: add the state key and increase
  the key count from 364 to 365.
- `LibraryIO/test_logiccontract_char.ambr`: add `WaveTileAdjacent: False` to
  serialized `DefaultSolution` maps.

Existing parameter values and default kernel names stay unchanged. Enabled
kernels include `_WTA1` in their names. No assembly goldens are updated.

## Consequences

The snapshots cover the new setting. Layout correctness remains checked by the
wave-coordinate, native-store and segment-interleave tests.
