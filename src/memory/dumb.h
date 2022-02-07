#ifndef DG_DUMB_H
#define DG_DUMB_H

class DumbConstruct {
    public:
        DumbConstruct(int argc, char* argv[]);
        ~DumbConstruct();

        Kokkos::View<double**> a;
        KOKKOS_INLINE_FUNCTION void operator()(const int i) const;
// KOKKOS_INLINE_FUNCTION void attempt_to_set(Kokkos::View<double*>& U);
// KOKKOS_FUNCTION void attempt_to_set(double* U);

};

#endif //DG_MEMORY_NERWORK_H
