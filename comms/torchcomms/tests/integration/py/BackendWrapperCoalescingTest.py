#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Verify ``BackendWrapper`` implements c10d's coalescing hooks so
``dist.batch_isend_irecv`` issues each batch as one ``ncclGroupStart``/
``ncclGroupEnd`` pair via the underlying ``TorchCommBatch`` instead of
running each P2POp ungrouped (which can deadlock for mixed send/recv
batches like the PP 1F1B middle stage).
"""

import os
import unittest

import torch
import torch.distributed as dist
from torchcomms.tests.helpers.py.test_helpers import skip_if_ncclx
from torchcomms.tests.integration.helpers.TorchCommTestHelpers import (
    get_device,
    get_rank_and_size,
)


@skip_if_ncclx
class TestBackendWrapperCoalescing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dist.config.use_torchcomms = True
        rank, world_size = get_rank_and_size()
        dist.init_process_group(
            backend=os.environ["TEST_BACKEND"], rank=rank, world_size=world_size
        )
        device = get_device(os.environ["TEST_BACKEND"], dist.get_rank())
        torch.set_default_device(device)
        cls.device = device

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group()

    def test_supports_coalescing_is_true(self):
        """``supportsCoalescing`` is overridden to ``True`` so c10d's
        ``_coalescing_manager`` (used by ``batch_isend_irecv``) calls
        ``startCoalescing`` / ``endCoalescing`` instead of issuing each
        P2POp individually."""
        pg = dist.distributed_c10d._get_default_group()
        backend = pg._get_backend(self.device)
        self.assertTrue(
            backend.supports_coalescing,
            "BackendWrapper.supports_coalescing must be True so "
            "batch_isend_irecv takes the coalescing path",
        )

    def test_batch_isend_irecv_mixed_send_recv(self):
        """A mixed isend+irecv batch in a single ``batch_isend_irecv``
        call delivers correctly. Without coalescing this pattern (mirrors
        PP 1F1B middle stage) deadlocks because each tc.send / tc.recv is
        enqueued ungrouped on the same NCCL stream."""
        world_size = dist.get_world_size()
        if world_size < 2:
            self.skipTest("need at least 2 ranks for batch_isend_irecv")

        rank = dist.get_rank()
        peer = (rank + 1) % world_size
        recv_peer = (rank - 1) % world_size
        send_tensor = torch.full((4,), float(rank), dtype=torch.float32)
        recv_tensor = torch.empty(4, dtype=torch.float32)

        ops = [
            dist.P2POp(dist.isend, send_tensor, peer),
            dist.P2POp(dist.irecv, recv_tensor, recv_peer),
        ]
        for req in dist.batch_isend_irecv(ops):
            req.wait()
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        expected = torch.full((4,), float(recv_peer), dtype=torch.float32)
        self.assertTrue(
            torch.equal(recv_tensor, expected),
            f"recv mismatch: got {recv_tensor.tolist()}, expected {expected.tolist()}",
        )

    def test_batch_isend_irecv_multiple_peers(self):
        """A batch of N sends + N recvs across multiple peers in a single
        coalesced batch — exercises the ncclGroupStart/End grouping over
        more than one P2P pair."""
        world_size = dist.get_world_size()
        if world_size < 3:
            self.skipTest("need at least 3 ranks for multi-peer batch")

        rank = dist.get_rank()
        # Send to (rank+1) and (rank+2), receive from (rank-1) and (rank-2).
        send_tensors = [
            torch.full((4,), float(rank * 10 + i), dtype=torch.float32)
            for i in range(2)
        ]
        recv_tensors = [torch.empty(4, dtype=torch.float32) for _ in range(2)]
        send_peers = [(rank + 1) % world_size, (rank + 2) % world_size]
        recv_peers = [(rank - 1) % world_size, (rank - 2) % world_size]

        ops = []
        for i in range(2):
            ops.append(dist.P2POp(dist.isend, send_tensors[i], send_peers[i]))
            ops.append(dist.P2POp(dist.irecv, recv_tensors[i], recv_peers[i]))

        for req in dist.batch_isend_irecv(ops):
            req.wait()
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        for i in range(2):
            expected_value = float(recv_peers[i] * 10 + i)
            expected = torch.full((4,), expected_value, dtype=torch.float32)
            self.assertTrue(
                torch.equal(recv_tensors[i], expected),
                f"slot {i} (from rank {recv_peers[i]}): got "
                f"{recv_tensors[i].tolist()}, expected {expected.tolist()}",
            )

    def test_allreduce_coalesced_multi_tensor(self):
        """Multiple all_reduce calls inside a coalescing manager produce
        correct results via the multi-tensor ``allreduce_coalesced`` path."""
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        t1 = torch.full((4,), float(rank), dtype=torch.float32)
        t2 = torch.full((8,), float(rank * 10), dtype=torch.float32)

        with dist._coalescing_manager():
            dist.all_reduce(t1)
            dist.all_reduce(t2)

        expected_sum = float(world_size * (world_size - 1) // 2)
        self.assertTrue(
            torch.allclose(t1, torch.full_like(t1, expected_sum)),
            f"t1 mismatch: {t1.tolist()}",
        )
        self.assertTrue(
            torch.allclose(t2, torch.full_like(t2, expected_sum * 10)),
            f"t2 mismatch: {t2.tolist()}",
        )

    def test_allgather_into_tensor_coalesced_multi_tensor(self):
        """Multiple all_gather_into_tensor calls inside a coalescing manager
        produce correct results via ``allgather_into_tensor_coalesced``."""
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        in1 = torch.full((4,), float(rank), dtype=torch.float32)
        in2 = torch.full((6,), float(rank + 100), dtype=torch.float32)
        out1 = torch.empty(4 * world_size, dtype=torch.float32)
        out2 = torch.empty(6 * world_size, dtype=torch.float32)

        with dist._coalescing_manager():
            dist.all_gather_into_tensor(out1, in1)
            dist.all_gather_into_tensor(out2, in2)

        for r in range(world_size):
            chunk1 = out1[r * 4 : (r + 1) * 4]
            expected1 = torch.full((4,), float(r), dtype=torch.float32)
            self.assertTrue(
                torch.equal(chunk1, expected1),
                f"out1 rank {r}: got {chunk1.tolist()}, expected {expected1.tolist()}",
            )
            chunk2 = out2[r * 6 : (r + 1) * 6]
            expected2 = torch.full((6,), float(r + 100), dtype=torch.float32)
            self.assertTrue(
                torch.equal(chunk2, expected2),
                f"out2 rank {r}: got {chunk2.tolist()}, expected {expected2.tolist()}",
            )

    def test_reduce_scatter_tensor_coalesced_multi_tensor(self):
        """Multiple reduce_scatter_tensor calls inside a coalescing manager
        produce correct results via ``reduce_scatter_tensor_coalesced``."""
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        in1 = torch.full((4 * world_size,), float(rank), dtype=torch.float32)
        in2 = torch.full((6 * world_size,), float(rank * 10), dtype=torch.float32)
        out1 = torch.empty(4, dtype=torch.float32)
        out2 = torch.empty(6, dtype=torch.float32)

        with dist._coalescing_manager():
            dist.reduce_scatter_tensor(out1, in1)
            dist.reduce_scatter_tensor(out2, in2)

        expected_sum = float(world_size * (world_size - 1) // 2)
        self.assertTrue(
            torch.allclose(out1, torch.full_like(out1, expected_sum)),
            f"out1 mismatch: {out1.tolist()}",
        )
        self.assertTrue(
            torch.allclose(out2, torch.full_like(out2, expected_sum * 10)),
            f"out2 mismatch: {out2.tolist()}",
        )


if __name__ == "__main__":
    unittest.main()
