# Project-owned composition layer.
#
# Skellington creates this file only when it is absent. Copier updates the
# imported .tooling files, while this file is free to diverge with the project.

import '.tooling/core.just'
import '.tooling/python-quality.just'
import '.tooling/python-test.just'

import '.tooling/python-package.just'


[private]
default: help

[group('convenience')]
fix:
  @just _run tooling-sync "just tooling-sync"
  @just _run_soft syntax "just syntax"
  @just _run_soft format "just format"
  @just _run_soft lint "just lint"
  @just _run_soft typecheck "just typecheck"
  @just _run_soft lint-imports "just lint-imports"
  @just _run "test --fast" "just test --fast"
  @just _run_soft cov "just cov"

[group('convenience')]
check:
  @just _run tooling-check "just tooling-check"
  @just _run syntax "just syntax"
  @just _run format "just format --check"
  @just _run lint "just lint --no-fix"
  @just _run typecheck "MODE=ci just typecheck"
  @just _run lint-imports "just lint-imports"
  @just _run test "just test"
  @just _run cov "just cov"


[group('production')]
release-check:
  @just _run check "just check"
  @just _run build-release "just build-release"
  @just _run release-smoke "just release-smoke"


# Add project-specific recipes and project-specific gates below this line.
