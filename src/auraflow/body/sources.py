"""Mesh -> FW-H source adapters for ``auraflow.body``.

This task delivers only the **permeable-surface** half. A closed
:class:`~auraflow.body.mesh.TriMesh` is the general-body counterpart of
:class:`auraflow.cfd.sphere.PermeableSphere`: its face centroids, outward
normals and areas are exactly the ``(y_panels [F, 3], normal [F, 3],
area [F])`` triple that :func:`auraflow.fwh.f1a_permeable_static` (and CFD
surface sampling) consume.

.. note::
   The **impermeable / loading** and **loudspeaker** adapters -- turning
   :func:`auraflow.body.panel_histories` plus surface pressure or membrane
   velocity into ``(y, v, a, L, Q_n)`` -- are the NEXT task and are not
   implemented here. See ``docs/architecture.md`` -> ``auraflow.body`` /
   ``sources.py`` for the full contract.
"""

from jax import Array

from auraflow.body.mesh import TriMesh

__all__ = ["permeable_surface"]


def permeable_surface(mesh: TriMesh) -> tuple[Array, Array, Array]:
    """Permeable FW-H data surface from a (closed) triangle mesh.

    Returns the per-face quadrature geometry the permeable-surface FW-H solver
    needs, matching the layout of :func:`auraflow.cfd.sphere.fibonacci_sphere`.
    The mesh should be watertight (a closed data surface); this is not enforced
    so open patches can still be sampled for diagnostics.

    Args:
        mesh: The permeable-surface :class:`TriMesh`.

    Returns:
        ``(points, normals, areas)`` with

        - ``points`` [m], shape ``[F, 3]`` -- face centroids (``y_panels``);
        - ``normals``, shape ``[F, 3]`` -- outward unit normals;
        - ``areas`` [m^2], shape ``[F]`` -- per-face areas.
    """
    return mesh.centroids(), mesh.normals(), mesh.areas()
