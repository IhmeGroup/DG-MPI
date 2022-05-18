#include "io/writer.h"

int main(int argc, char *argv[])
{
    int mpi_size, mpi_rank;
    MPI_Comm comm = MPI_COMM_WORLD;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(comm, &mpi_size);
    MPI_Comm_rank(comm, &mpi_rank);

    std::string filename_in = "data.h5";
    if (argc > 1)
        filename_in = argv[1];
    std::string filename_out = filename_in + "_serial.h5";
    if (argc > 2)
        filename_out = argv[2];

    // read the file in parallel and serialize all data
    {
        bool serialize = true;
        Reader reader(filename_in, serialize);

        // write the data in serial
        bool parallel = false;
        Writer writer(filename_out, reader, parallel);
    }

    MPI_Finalize();
}

