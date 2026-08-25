import torch

import cuda_operator


def test_forward_cpu():
    a = torch.randn(32, 64)
    b = torch.randn_like(a)

    actual = cuda_operator.vector_add(a, b)
    expected = a + b

    torch.testing.assert_close(actual, expected)


def test_forward_cuda():
    a = torch.randn(
        1024,
        1024,
        device="cuda",
        dtype=torch.float32,
    )
    b = torch.randn_like(a)

    actual = cuda_operator.vector_add(a, b)
    expected = a + b

    torch.testing.assert_close(actual, expected)


def test_non_contiguous_cuda():
    a = torch.randn(32, 64, device="cuda").transpose(0, 1)
    b = torch.randn_like(a)

    assert not a.is_contiguous()

    actual = cuda_operator.vector_add(a, b)
    expected = a + b

    torch.testing.assert_close(actual, expected)


def test_opcheck():
    a = torch.randn(
        16,
        32,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    b = torch.randn_like(a, requires_grad=True)

    torch.library.opcheck(
        torch.ops.cuda_operator.vector_add.default,
        (a, b),
    )


def test_gradcheck():
    a = torch.randn(
        4,
        8,
        device="cuda",
        dtype=torch.double,
        requires_grad=True,
    )
    b = torch.randn(
        4,
        8,
        device="cuda",
        dtype=torch.double,
        requires_grad=True,
    )

    assert torch.autograd.gradcheck(
        cuda_operator.vector_add,
        (a, b),
    )


def test_torch_compile():
    a = torch.randn(32, 64, device="cuda")
    b = torch.randn_like(a)

    compiled_operator = torch.compile(
        cuda_operator.vector_add,
        fullgraph=True,
    )

    actual = compiled_operator(a, b)
    expected = a + b

    torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    test_forward_cpu()
    test_forward_cuda()
    test_non_contiguous_cuda()
    test_opcheck()
    test_gradcheck()
    test_torch_compile()

    print("All tests passed.")