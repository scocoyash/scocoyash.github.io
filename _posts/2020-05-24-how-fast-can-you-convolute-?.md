---
layout: post
comments: false
title: "How fast can you Convolute ?"
date: 2020-05-24 00:00:00
tags: learnings convolution
---

> Is convolution in the real world, ever performed, in the way we were taught in various courses ?
  Ehh, I don't think so. Let's explore how to optimize this heavy-computational operation using basic algorithmic cleverness and parallelization.  

<!--more-->

On my normal laptop, i3 - 3<sup>rd</sup> Gen, I can infer most common CNN models using Tensorflow, Tensorflow Lite or Pytorch within 10-100 mili-seconds. Even on latest flagship mobile chipsets such as Qualcomm's Snapdragon 865, using Tensorflow Lite, I am able to execute Mobilents (V1, V2) in almost (at max) 30ms. <br/>

But you will see, when you implement a normal convolution operation, using C/C++, it takes around 1 second for a single layer to execute. So, how are these libraries able to optimize the convolution operation by a factor of around 100x ? It's really interesting to see human-cleverness at play while designing algorithms as well as exploitation of low-level architecture. <br/>

Most of what I have read and written in this article is from a paper titled [Anatomy of High-Performance Many-Threaded Matrix Multiplication](https://www.cs.utexas.edu/users/flame/pubs/blis3_ipdps14.pdf) and the links mentioned in references. 

{: class="table-of-content"}
* TOC
{:toc}

---

## Basic Terminologies

### FLOP/S (floating point operations per second)

FLOPS are a measure of performance used for comparing the peak theoretical performance of a core, microprocessor, or system using floating point operations. This unit is often used in the field of high-performance computing (e.g., supercomputers) in order to evaluate the peak theoretical performance of various scientific workloads.

For my CPU, the intel i3 based on IvyBridge micro-architecture, let me calculate the FLOP/s: 

- 2 physical cores
- each core has a frequency of 1.8 GHz, or 1.8 x 10<sup>9</sup> CPU cycles per second
- in each cycle, it can process 8 FLOPs (only AVX available for i3 - 3<sup>rd</sup> Gen)

Therefore, my PC has 2 x 1.8 x 10<sup>9</sup> x 8 = 28.8 GFLOP/s peak performance.

### Image Data Formats

We imagine images/tensors to be represented in multi-dimensional array format. But actually, they are physically stored in one-dimensional array. We need to define a convention to unroll this multi-dimensional arrays into one-dimensional format.

By far the two most common memory layouts for multi-dimensional array data are *row-major* and *column-major*
The *row-major* layout of a matrix puts the first row in contiguous memory, then the second row right after it, then the third, and so on.
![Row Major 2D Matrix](/assets/images/convolutions/row-major-2D.png)
{: style="display: block;margin: 0 auto;"}
<i><center>Fig. Row Major format for 2D matrix</center></i>

The *column-major* layout puts the first column in contiguous memory, then the second column, etc.
![Column Major 2D Matrix](/assets/images/convolutions/column-major-2D.png)
{: style="display: block;margin: 0 auto;"}
<i><center>Fig. Column Major format for 2D matrix</center></i>

Every common memory layout in Deep-Learning is row-major.<br/>
Image data format refers to the representation of batches of images. In case of TensorFlow, it supports NHWC (TensorFlow default) and NCHW (cuDNN default). N refers to the number of images in a batch, H refers to the number of pixels in the vertical dimension, W refers to the number of pixels in the horizontal dimension, and C refers to the channels.

![Different memory layouts](/assets/images/convolutions/memory-layouts.png)
{: style="display: block;margin: 0 auto;"}
<i><center>Fig. Different memory layouts</center></i>

For this article, I am going to stick to NCHW layout as it is mostly used for GPU's.

## Naive Convolution

Suppose we have an image with height H, width W and channels C. 
We convolve this with **M** filters of size **K x K** each featuring the same no. of channels C. 

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
Elapsed time: 1192.744019 milliseconds GFlops=0.018544
```

Eww! 6 nested Loops!!<br/>
Wooping **1.2 seconds** to run a basic convolution operation. It's terribly slow. I can execute a whole CNN network using Tensorflow in less than 100 mili-seconds and here I am, just performing a single convolution operation taking ~100x more time.

Optimizing such a complex nested for loop is non-trivial. If you have tried optimizing matrix multiplication, you know that there is a lot of tricks involved in it.

## Convolution as GEMM

As you saw above, a normal convolution operation is too tricky to be performed if you have to match the performances that advanced libraries such as BLAS provide. Maybe we can visualize this problem as a different operation altogether. How about a matrix multiplication ?

That's what developers of Original Caffe Framework did. They converted the convolution operation to be visualized as a normal **Ge**neralized **M**atrix **M**ultiplication(GEMM) operation. The advantage of this was they could build on several decades of optimization of GEMM and get maximal performance as well.  

### Intuition 
If we closely look at the convolution operation, it is nothing but dot-product between the kernel filter and a local region on the input selected by moving window, that samples a patch with the same size as our filter. We can unroll this patch into a column vector which helps us to realize it using GEMM.


## References

- [Memory Layouts by Eli](https://eli.thegreenplace.net/2015/memory-layout-of-multi-dimensional-arrays)
- [Intel - memory Layout](https://oneapi-src.github.io/oneDNN/understanding_memory_formats.html)
- [Convolution in Caffe](https://github.com/Yangqing/caffe/wiki/Convolution-in-Caffe:-a-memo)