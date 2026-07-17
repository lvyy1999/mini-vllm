import pytest

torch = pytest.importorskip("torch")

from minivllm.utils.context import get_context, reset_context, set_context


@pytest.fixture(autouse=True)
def clean_context():
    reset_context()
    yield
    reset_context()


class TestContextLifecycle:

    def test_default_context(self):
        context = get_context()

        assert context.is_prefill is False
        assert context.cu_seqlens_q is None
        assert context.cu_seqlens_k is None
        assert context.max_seqlen_q == 0
        assert context.max_seqlen_k == 0
        assert context.slot_mapping is None
        assert context.context_lens is None
        assert context.block_tables is None

    def test_get_context_returns_current_instance(self):
        assert get_context() is get_context()

    def test_set_context_preserves_all_values(self):
        cu_seqlens_q = torch.tensor([0, 2, 5], dtype=torch.int32)
        cu_seqlens_k = torch.tensor([0, 4, 8], dtype=torch.int32)
        slot_mapping = torch.tensor([0, 1, 4, 5, 6], dtype=torch.int32)
        context_lens = torch.tensor([4, 8], dtype=torch.int32)
        block_tables = torch.tensor([[1, -1], [2, 3]], dtype=torch.int32)

        set_context(
            is_prefill=True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=3,
            max_seqlen_k=4,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )
        context = get_context()

        assert context.is_prefill is True
        assert context.cu_seqlens_q is cu_seqlens_q
        assert context.cu_seqlens_k is cu_seqlens_k
        assert context.max_seqlen_q == 3
        assert context.max_seqlen_k == 4
        assert context.slot_mapping is slot_mapping
        assert context.context_lens is context_lens
        assert context.block_tables is block_tables

    def test_set_context_replaces_omitted_previous_values(self):
        set_context(
            is_prefill=True,
            slot_mapping=torch.tensor([1]),
            max_seqlen_q=8,
        )

        set_context(is_prefill=False)
        context = get_context()

        assert context.is_prefill is False
        assert context.slot_mapping is None
        assert context.max_seqlen_q == 0

    def test_reset_context_replaces_current_instance(self):
        set_context(is_prefill=True, max_seqlen_q=5)
        previous = get_context()

        reset_context()
        current = get_context()

        assert current is not previous
        assert current.is_prefill is False
        assert current.max_seqlen_q == 0
