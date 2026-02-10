from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import jax.numpy as jnp

Array = Any


@dataclass(frozen=True)
class VacuumField:
    """Callable vacuum field solution.

    Parameters
    ----------
    phi:
        Scalar potential evaluator ``phi(X) -> (n,)``.
    B:
        Magnetic field evaluator ``B(X) -> (n,3)``.
    metadata:
        Lightweight, JSON-serializable metadata about the solve.
    """

    phi: Callable[[Array], Array]
    B: Callable[[Array], Array]
    metadata: Mapping[str, Any]

    def __call__(self, X: Array) -> Array:
        """Alias for ``B(X)``."""
        return self.B(X)

    def B_mag(self, X: Array) -> Array:
        """Return ``|B|`` at points ``X``."""
        Bx = self.B(X)
        return jnp.linalg.norm(Bx, axis=-1)

