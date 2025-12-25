import triton
import triton.language as tl

# ref: https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html#sphx-glr-getting-started-tutorials-01-vector-add-py

@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    #  有多个'程序'（也就是block）处理不同的数据。我们在这里标识我们是哪个程序：
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    # 该程序将处理与初始数据偏移的输入。
    # 例如，如果您有长度为 256 的向量和块大小为 64，程序
    # 将分别访问元素[0:64, 64:128, 128:192, 192:256]。
    # 请注意，偏移量是指针的列表：
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    # 创建一个mask以防止内存操作超出范围。
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    # 我们需要预先分配输出。
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    # SPMD启动网格表示并行运行的内核实例数。
    # 它类似于CUDA启动网格。对于add_kernel我们使用一个1D网格，其大小是块的数量：
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # 注意：
    #  - 每个torch.tensor对象都隐式地转换为指向其第一个元素的指针。
    #  - `triton.jit`'ed函数可以通过一个启动网格索引来获得一个可调用的GPU内核。
    #  - 不要忘记将元参数作为关键字参数传递。
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    # 我们返回一个指向z的句柄，但是，由于`torch.cuda.synchronize()`尚未被调用，内核此时仍在异步运行。
    return output
