# CPython freelists

The [cpython]([GitHub - python/cpython: The Python programming language](https://github.com/python/cpython/)) implementation contains freelists to improve allocation performance of often used objects such as dicts, lists and tuples.

Here we present results on the allocation statistics for freelists. For each freelist we record how many objects allocations are performed for the freelist size. If allocations are performed while the freelist size is zero, this means there are no objects available on the freelist and a normal allocation is used instead.

Statistics have been gathered with branch [GitHub - eendebakpt/cpython at small_list_freelist_statistics](https://github.com/eendebakpt/cpython/tree/small_list_freelist_statistics)



```
![image info]('freelist_allocations_floats.png' "floats")
```

The full statistics are available at [freelist_stats.md](freelist_stats.md).


