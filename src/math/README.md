## `math`

The `math` directory contains the `Math` namespace. It provides wrappers around various math operations. 
Each function utilizes Kokkos kernels to conduct various mathematical 
operations. (Example: computing the inverse of matrix A)
Some things to note:
    1. We use function overloading to access the serial vs team versions 
       of the Kokkos batched functions. For the team versions, we must pass
       the member of the TeamPolicy.
    2. These are mostly batched Kokkos kernels wrappers, meaning
       each time they are called they MUST be called inside a Kokkos 
       parallel_for
    3. There are a few simple c-array functions as well. These also needed to be called inside of Kokkos parallel kernels but do not take in Views as the primary datatype.
    
The `linear_algebra.h` file has comments for each function and their use. Implementations of both `serial` and `team` versions exist for many of the functions. For the `team` versions, users
need to specify the `const MemberType& member`. This allows for the entire team to work on the operation. This can be the source of simple bugs if you are trying to use a serial operation where you
have a team of threads operating. Users should always use team versions in TeamPolicy kernels unless they have isolated a single thread to operate in a hierarchical teampolicy.
