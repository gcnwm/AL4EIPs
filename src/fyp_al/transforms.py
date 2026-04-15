"""Custom transforms for Active Learning with NequIP on non-periodic systems.

Fixes the edge_cell_shift corruption bug: when ASEDataset is used for
non-periodic systems, the neighbor list generates non-zero edge_cell_shift
values with an identity cell matrix. Combined with ``with_edge_vectors_()``,
this corrupts every edge vector by adding integer Ångström shifts.

The fix: zero out ``edge_cell_shift`` after the neighbor list is built.
"""

import torch

from nequip.data import AtomicDataDict


class ZeroEdgeCellShiftTransform(torch.nn.Module):
    """Zero out ``edge_cell_shift`` for non-periodic systems.

    When using ASEDataset with ``NonPeriodicCellTransform``, the neighbor-list
    code can produce non-zero ``edge_cell_shift`` values even though the system
    is non-periodic (``pbc=False``).  These spurious shifts corrupt edge
    vectors inside ``with_edge_vectors_()`` because:

        edge_vec += edge_cell_shift @ cell   (cell = identity for non-periodic)

    This transform sets all ``edge_cell_shift`` entries to zero, eliminating
    the corruption while keeping the ``cell`` key (which other parts of the
    pipeline may expect).

    Place this transform **after** ``NeighborListTransform`` in the transform
    chain.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        if AtomicDataDict.EDGE_CELL_SHIFT_KEY in data:
            data[AtomicDataDict.EDGE_CELL_SHIFT_KEY] = torch.zeros_like(
                data[AtomicDataDict.EDGE_CELL_SHIFT_KEY]
            )
        return data
