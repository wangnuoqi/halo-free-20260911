from refine_dataparallel import effective_batch_size, visible_gpu_count


def test_visible_gpu_count_and_effective_batch_size():
    assert visible_gpu_count("0,1,2,3") == 4
    assert visible_gpu_count("") == 1
    assert effective_batch_size(1, "0,1,2,3") == 4
    assert effective_batch_size(8, "0,1,2,3") == 8

