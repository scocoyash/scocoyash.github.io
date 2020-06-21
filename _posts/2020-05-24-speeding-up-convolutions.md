---
layout: post
comments: false
title: "Speeding up Convolutions"
date: 2020-05-24 00:00:00
tags: learnings convolution deep-learning
---

> Is convolution in the real world, ever performed, in the way we were taught in various courses ?
  Nopes, I don't think so. Let's explore one of the ways to optimize this heavy-computational operation using basic algorithmic cleverness.  

<!--more-->

On my normal laptop, i3 - 3<sup>rd</sup> Gen, I can infer most common CNN models using Tensorflow, Tensorflow Lite or Pytorch within 10-100 milliseconds. Even on latest flagship mobile chipsets such as Qualcomm's Snapdragon 865, using Tensorflow Lite, I am able to execute Mobilenets (V1, V2) in almost (at max) 30ms. <br/>

But you will see, when you [implement](https://github.com/scocoyash/Convolution-To-Gemm) a normal convolution operation, using C/C++, it takes around 1 second for a single layer to execute. So, how are these libraries able to optimize the convolution operation by a factor of around 100x ? It's really interesting to see human-cleverness at play while designing algorithms as well as exploitation of low-level architecture. <br/>

Most of what I have read and written in this article is from a paper titled [Anatomy of High-Performance Many-Threaded Matrix Multiplication](https://www.cs.utexas.edu/users/flame/pubs/blis3_ipdps14.pdf) and the links mentioned in references. 

## Table Of Contents
{: class="table-of-content"}
* TOC
{:toc}

---

## Basic Terminologies

### FLOP/s (floating point operations per second)

FLOPS are a measure of performance used for comparing the peak theoretical performance of a core, microprocessor, or system using floating point operations. This unit is often used in the field of high-performance computing (e.g., supercomputers) in order to evaluate the peak theoretical performance of various scientific workloads.

For my CPU, the intel i3 based on IvyBridge micro-architecture, let me calculate the FLOP/s: 

- 2 physical cores
- each core has a frequency of 1.8 GHz, or 1.8 x 10<sup>9</sup> CPU cycles per second
- in each cycle, it can process 8 FLOPs (only AVX available for i3 - 3<sup>rd</sup> Gen)

Therefore, my PC has 2 x 1.8 x 10<sup>9</sup> x 8 = 28.8 GFLOP/s peak performance.

### Image Data Formats

We imagine images/tensors to be represented in multi-dimensional array format. But actually, they are physically stored in one-dimensional array. We need to define a convention to unroll this multi-dimensional arrays into one-dimensional format.

By far the two most common memory layouts for multi-dimensional array data are *row-major* and *column-major*.<br/>
The *row-major* layout of a matrix puts the first row in contiguous memory, then the second row right after it, then the third, and so on.
![Row Major 2D Matrix](/assets/images/convolutions/row-major.svg)
{: style="display: block;margin: 0 auto;"}
<i><center>Fig. Row Major format for 2D matrix</center></i>

The *column-major* layout puts the first column in contiguous memory, then the second column, etc.
![Column Major 2D Matrix](/assets/images/convolutions/column-major.svg)
{: style="display: block;margin: 0 auto;"}
<i><center>Fig. Column Major format for 2D matrix</center></i>

Common memory layout in Deep-Learning is **row-major**.<br/>

In case of 3D or higher dimensions, Image data format additionally refers to the representation of batches of images. In case of TensorFlow, it supports **NHWC** (TensorFlow default) and **NCHW** (cuDNN default). N refers to the number of images in a batch, H refers to the number of pixels in the vertical dimension, W refers to the number of pixels in the horizontal dimension, and C refers to the channels.

![Different memory layouts](/assets/images/convolutions/data-layouts.svg)
{: style="display: block;margin: 0 auto;"}
<i><center>Fig. Different Physical Data Layouts</center></i>

For this article, I am going to stick to NCHW layout as it is most commonly used in GPU's.

## Naive Convolution

Suppose we have an image with height H, width W and channels C. 
We convolve this with **M** filters of size **K x K** each featuring the same no. of channels C. 

For examples in this article, I have taken a matrix of size *512 x 512 x 512* (H x W x C) and 
a filter of size *3 x 3 x 512*  keeping the channels same.

Let's have a look at naive convolution algorithm first :

```python
'''
Convolve `input` with `kernel` to generate `output`
    input.shape = [input_channels, input_height, input_width]
    output.shape = [num_filters, output_height, output_width]
    kernel.shape = [num_filters, input_channels, kernel_height, kernel_width]
'''
for filter in 0..num_filters
    for channel in 0..input_channels
        for out_h in 0..output_height
            for out_w in 0..output_width
                for k_h in 0..kernel_height
                    for k_w in 0..kernel_width
                        output[filter, out_h, out_w] += 
                            kernel[filter, channel, k_h, k_w] * 
                            input[channel, out_h + k_h, out_w + k_w]

```

On my PC, executing this algorithm gave the following result:

```
Elapsed time: 2475.364990 milliseconds GFlops: 0.108443
```

Eww! 6 nested Loops!!<br/>
Wooping **2.4 seconds** to run a basic convolution operation. It's terribly slow. We can execute a whole CNN network using Tensorflow in less than 100 milliseconds and here we are, just performing a single convolution operation taking ~100x more time.

Optimizing such a complex nested for loop is non-trivial. If you have tried optimizing matrix multiplication, you know that there is a lot of tricks involved in it.
## Convolution as GEMM

As you saw above, a normal convolution operation is too tricky to be performed if you have to match the performances that advanced libraries such as BLAS provide. Maybe we can visualize this problem as a different operation altogether. How about a matrix multiplication ?

That's what developers of Original Caffe Framework did. They converted the convolution operation to be visualized as a normal **Ge**neralized **M**atrix **M**ultiplication(GEMM) operation. The advantage of this was they could build on several decades of optimization of GEMM and get maximal performance as well.  

### Intuition 
If we closely look at the convolution operation, it is nothing but dot-product between the kernel filter and a local region on the input selected by moving window, that samples a patch with the same size as our filter. We can unroll this patch into a column vector which helps us to realize it using GEMM.

So, we are able to achieve much better performance speedups compared to naive-convolution operation but at the expense of more memory, which is fair enough.
The above laying out of the image patches into a matrix is called **im2col**, for image to column. We rearrange the image into columns of a matrix, so that each column corresponds to one patch where the conv filter is applied.

### Example for im2col
Consider this normal convolution operation to be performed.

![Naive Convolution](/assets/images/convolutions/naive-conv.svg)
{: style="display: block;margin: 0 auto;"}
<i><center>Fig. Naive Convolution Operation</center></i>

Below is the same operation implemented as a matrix multiplication. 
The right matrix is the result of im2col – it has to be constructed by copying pixels from the original image. 
The left matrix has the conv weights, which are already stored this way in memory.

![Im2Col Operation](/assets/images/convolutions/im2col.svg)
{: style="display: block;margin: 0 auto;"}
<i><center>Fig. im2col Operation</center></i>

The time taken by this conversion from image to matrix using im2col operation as well as the memory used has to be justified by providing some serious speedups in order to provide performance improvements.

## Gemm and Optimizations

Note: All the matrix multiplications are performed on arrays of size 512 x 512 to match with the above
convolution operation. All the below implementations are at github repo [here](https://github.com/scocoyash/Convolution-To-Gemm).

### Naive Gemm
From our basic linear algebra textbooks, we see how Matrix Multiplication is performed.

```python
'''
Matrix A: size M * K
Matrix B: size K * N
Output Matrix C: size M * N
'''
for i in 0..M:
    for j in 0..N:
        for k in 0..K:
            C[i, j] += A[i, k] * B[k, j]
```
It has 2 Floating point operations - Multiply and Add, within the innermost for loop and these operations
are performed M * N * K times. So, the number of **FLOps** for this GEMM is 2 * M * N * K. 

[Note: **FLOps** stands for Floating Point Operations whereas **FLOP/s** stands for Floating Point Operations per second]

After executing this code for 512x512 arrays, these 3 nested loops, gave the following result:
```python
Naive Elapsed time: 94.907997 milliseconds GFlops: 2.828376
```

Not Bad! More than **~20x** improvements by just using naive-gemm on the same size of inputs but in a matrix form. Not to forget, we have to add time taken by im2col too. 
GFLOps has improved but still we are not utilizing all the processing capacity available.

### Naive Gemm + Caching
CPU caches are small pools of memory that store information the CPU is most likely to need next.
The goal of the cache system is to ensure that the CPU has the next bit of data it will need already loaded into cache by the time it goes looking for it (also called a cache hit).
A cache miss, on the other hand, means the CPU has to go scampering off to find the data elsewhere. 

This is where the L2 cache comes into play — while it’s slower, it’s also much larger. 
If data can’t be found in the L2 cache, the CPU continues down the chain to L3 (typically still on-die), then L4 (if it exists) and main memory (DRAM).

Every time we fetch data from the main memory, the CPU automatically loads it and its neighboring memory into the cache, hoping to utilize locality of reference. Are we utilizing these caches properly? Let's have a look at it.

![Matmul Operation](/assets/images/convolutions/mat-mul.svg)
{: style="display: block;margin: 0 auto;"}
<i><center>Fig. Normal Matrix-Multiplication</center></i>

Let us observe the order in which we are accessing our data. We traverse row-wise on Matrix A and 
column-wise on Matrix B. Since order of storage for matrix A is row-order storage, once we have A[m, k],
the next element A[m, k + 1] is already cached.

But there's something else going with Matrix B. Let's see..

Since we need next element from the same column, we should have it in cache. For e.g: If we have B[0,0],
we will need B[0,1], B[0,2], etc.. But here we have B[1,0], B[2,0], etc..
So, when we fetch B[0,0], the next element B[0,1] isn't present in the cache, leading to cache miss;
CPU stalls while fetching the data from the RAM to the cache wasting CPU cycles.

Another thing is, once we fetch a value, the next values in the row are also fetched which are not
required immediately, but will be required after a few cycles. At that time, we need to fetch these 
values again. This is known as **Cache Pollution**.
Wikipedia defines it as: 
> Cache pollution describes situations where an executing computer program loads data into CPU cache unnecessarily, thus causing other useful data to be evicted from the cache into lower levels of the memory hierarchy, degrading performance.  

![Traversal Order](/assets/images/convolutions/traversal-order.svg)
{: style="display: block;margin: 0 auto;"}
<i><center>Fig. Data Traversal Order</center></i>

In order to deal with this cache pollution, we will need to re-order this loops.<br/>
Re-ordering **i, j, k** to **i, k, j**, the naive algorithm becomes - 

```python
'''
Matrix A: size M * K
Matrix B: size K * N
Output Matrix C: size M * N
'''
for i in 0..M:
    for k in 0..K:
        for j in 0..N:
            C[i, j] += A[i, k] * B[k, j]
```

Using this, we are able to notice improvements and achieve considerable speedup over naive-gemm. Here are the results: 

```python
Caching Elapsed time: 52.035999 milliseconds GFlops: 5.158649
```

Impressive! We almost halved the time by utilizing or knowledge of memory architectures and caches.
Also, the increase in GFlops show us that we are optimally utilizing the processing power now.
Now, we move on to a different approach to deal very large matrices as in the case of deep learning.

### Tiling
Our memory access pattern is still inefficient. We access matrix B many times and multiply the values with different parts of matrix A. 
As a result, we load matrix B into our cache, and then invalidate this cache by loading a different part of matrix B.

The order in which we calculate patches in matrix C affects the order in which we access memory in matrix A and B. An ideal memory access pattern would be one where we load matrix B into our L1 cache and keep it there for a long time. 

One way to make this happen is to tile the calculation of matrix C. This is pretty easy to do, just divide matrix C into small tiles (say, 128 x 256) and calculate the results in C one patch at a time.

The pseudo-code for this is:

```python
'''
Matrix A: size M * K
Matrix B: size K * N
Output Matrix C: size M * N
'''
block_size = 32
for i in 0..N increment by block_size
    imin = min(i + block_size, N)
    for j in 0..M increment by block_size
        jmin = min(j + block_size, M)
        for k in 0..K increment by block_size
            kmin = min(k + block_size, K)
            for x in 0..imin
                for y in 0..jmin
                    for z in 0..kmin
                        C[x * M + y] +=  A[x * M + z] * B[z * M  + y]
```

You might be shocked to see the results with this:

```python
Tiling Elapsed time: 272.923004 milliseconds GFlops: 0.983557
```

It worsened the timings of normal gemm by a fraction of **~3x**. But according to our intuition, it should improve right? What happened here ?

As from experiments, it is found that tiling works better when the arrays are of **size > 1024** elements.
Since our arrays are of size with 512 elements, it degraded performance.

### Tiling + Caching

Just like we utilized caches efficiently in our previous caching algorithm, from our intuition, we might as well utilize it in tiling too.

Here's the pseudo code for it:

```python
'''
Matrix A: size M * K
Matrix B: size K * N
Output Matrix C: size M * N
'''
block_size = 32
for i in 0..N increment by block_size
    imin = min(i + block_size, N)
    for k in 0..K increment by block_size
        kmin = min(k + block_size, K)
        for j in 0..M increment by block_size
            jmin = min(j + block_size, M)
            for x in 0..imin
                for z in 0..kmin
                    for y in 0..jmin
                        C[x * M + y] +=  A[x * M + z] * B[z * M  + y]
```

Tadada -

```python
Tiling + Caching Elapsed time: 63.615002 milliseconds GFlops: 4.219688
```
It was due to caching our performance was degrading. Tiling indeed gives better performance if implemented
correctly utilizing caches properly.
Even so, when you experiment with larger array sizes such as 1024, it outperforms above caching too.
We haven't introduced one more **Bhramastra** of high performance computing yet. Let's utilize that too
in nexr section.

### Parallelization
If you have reached this part of the article, you might see, whatever we have done till now to increase the processing power utilization, we did not mention what cores to use and how many threads to spawn.
We were just asking the processor to execute one instruction at a time.
But since now-a-days we have multiple cores in CPU, we might well utilize those too for our performance improvements.

Just defining parallel computing here:
> In the simplest sense, **parallel computing** is the simultaneous use of multiple compute resources to solve a computational problem:
- A problem is broken into discrete parts that can be solved concurrently
- Each part is further broken down to a series of instructions
- Instructions from each part execute simultaneously on different processors
- An overall control/coordination mechanism is employed

You might want to read more on parallel computing from the link in references.

The easiest way to utilize the power of cores is via spawning threads on the parts of the code that operate on different unrelated data.
For C++, Openmp is the most common used library with which I have experimented and it yielded the following results:

```python
Caching Elapsed time: 35.613998 milliseconds GFlops: 7.537358

Tiling Elapsed time: 163.929993 milliseconds GFlops: 1.637501

Tiling + Caching Elapsed time: 39.122002 milliseconds GFlops: 6.861496
```
These were the best timings we were able to achieve by parallelizing the above algorithms using Openmp.
You can see that performance improvements of about **~90x** compared to normal convolution operation
and further **~3x** improvements from naive gemm. 

We still are not able to utilize the whole processing power which is evident from the GFlops here. At max, we achieved a GFlops of **7.5** which is one-fourth of computing capability of my PC with GFlops of **28**.

## Explore
Also, you can try out **Vectorization** using AVX and SSE instruction sets, which is in common usage in most of the deep learning libraries. I won't be covering it here because it's too deep for a topic to explore as of now.

Having learnt how convolution is usually performed underneath most of the systems was too a roller-coaster ride for me too. The methods discussed above are just few algorithms how Convolution is optimized.
Others being Fast-Fourier Transform, Winograds algorithm for 3 x 3 filters, etc.
Again, will not be covering those here but maybe sometime in future. 

Till then, Stay safe and keep learning!

---
## References

- [Memory Layouts by Eli](https://eli.thegreenplace.net/2015/memory-layout-of-multi-dimensional-arrays)
- [Intel - memory Layout](https://oneapi-src.github.io/oneDNN/understanding_memory_formats.html)
- [Convolution in Caffe](https://github.com/Yangqing/caffe/wiki/Convolution-in-Caffe:-a-memo)
- [Praising Moon](https://praisethemoon.org/demystifying-the-math-and-implementation-of-convolutions-part-iii/)
- [UC Texas Paper](https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf)
- [Pete Warden's Blog](https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/)
- [Parallel Computing](https://computing.llnl.gov/tutorials/parallel_comp/)

---

<i><center>For those who haven't met me before, I am <b>Yash</b>, writing this article with <span style="color:red;"> &#10084; </span> from India.</center></i>