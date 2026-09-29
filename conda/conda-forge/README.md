# conda-forge recipe

The recipe to submit to [conda-forge/staged-recipes](https://github.com/conda-forge/staged-recipes)
(as `recipes/pycuampcor/meta.yaml`), and later maintained in the `pycuampcor-feedstock`.
It builds from a tagged release; the variants (cpu, and cuda for the CUDA
versions in the conda-forge global pinning) are defined by conda-forge.

For each release:

1. tag the release, e.g., `git tag v2.1.0 && git push origin v2.1.0`;
2. update `version` in `meta.yaml` (the same as `project(... VERSION ...)` in CMakeLists.txt);
3. update `sha256` with
   `curl -sL https://github.com/lijun99/cuAmpcor/archive/refs/tags/v<version>.tar.gz | sha256sum`.

See *../recipe* for building the packages locally from the source tree.
