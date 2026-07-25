"""检索 store 缓存失效策略测试。"""

from qa_core.retrieval.factory import (
    clear_retrieval_store_cache,
    get_hybrid_store,
    sync_retrieval_cache_for_active_version,
)


def test_active_kb_version_change_clears_cached_hybrid_stores() -> None:
    """active 版本变化后，已缓存的 Milvus store wrapper 必须失效。

    调用顺序：pytest/unittest 测试入口 -> test_active_kb_version_change_clears_cached_hybrid_stores()。
    """
    clear_retrieval_store_cache()
    try:
        first = get_hybrid_store("unit_test_collection")
        assert get_hybrid_store("unit_test_collection") is first

        sync_retrieval_cache_for_active_version("enterprise_knowledge", "kb_v1")
        assert get_hybrid_store("unit_test_collection") is first

        sync_retrieval_cache_for_active_version("enterprise_knowledge", "kb_v2")
        assert get_hybrid_store("unit_test_collection") is not first
    finally:
        clear_retrieval_store_cache()
