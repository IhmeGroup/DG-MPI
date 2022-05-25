# `memory`

Memory is the location of almost all MPI commands. The idea here is to abstract away MPI level items such that users do not need to know a lot of MPI.
This module has several functions that will be discussed below:

1. `MemoryNetwork` constructor. The constructor sets up `MPI_Init` as well as `Kokkos::Initialize`. This sets up all the basic MPI settings for the rest of the solver.
2. `communicate_face_solution`: This function is the primary communication that occurs between processors. In the DG scheme we only pass the state coefficients at the faces. Therefore, 
this is a face operation only (i.e. there are NO ghost elements, just ghost faces). Currently, the ghost faces are included in the interior faces View. This was done early on because it was thought that it would simplify loops over interior faces. This has come at a cost because it is difficult to hide the communication behind other operations. We offer some ideas of how to improve this:
    1. Separate the ghost faces from the interior faces. Have the ghost faces construct separate helper objects and have separate loops for evaluating fluxes, etc...
    2. Make the communication completely non-blocking. Currently, only the send is non-blocking, but the receive is still blocking.
    3. Use the volume flux to hide the communication of the fluxes. 
3. `print_view`: These functions are designed for debugging. They print various View types with different ranks in a user friendly way. This is specific for debugging.
  
