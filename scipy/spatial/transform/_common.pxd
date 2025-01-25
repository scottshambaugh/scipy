cimport cython
from cython.view cimport array
from libc.math cimport sqrt


cdef inline double[:] empty1(int n) noexcept:
    return array(shape=(n,), itemsize=sizeof(double), format=b"d")


cdef inline double[:, :] empty2(int n1, int n2) noexcept:
    return array(shape=(n1, n2), itemsize=sizeof(double), format=b"d")


cdef inline double[:, :, :] empty3(int n1, int n2, int n3) noexcept:
    return array(shape=(n1, n2, n3), itemsize=sizeof(double), format=b"d")


cdef inline double[:, :] zeros2(int n1, int n2) noexcept:
    cdef double[:, :] arr = array(shape=(n1, n2),
        itemsize=sizeof(double), format=b"d")
    arr[:, :] = 0
    return arr


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double dot3(const double[:] a, const double[:] b) noexcept nogil:
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


# flat implementations of numpy functions
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double[:] cross3(const double[:] a, const double[:] b) noexcept:
    cdef double[:] result = empty1(3)
    result[0] = a[1]*b[2] - a[2]*b[1]
    result[1] = a[2]*b[0] - a[0]*b[2]
    result[2] = a[0]*b[1] - a[1]*b[0]
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double norm3(const double[:] elems) noexcept nogil:
    return sqrt(dot3(elems, elems))


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void compose_quat_single( # calculate p * q into r
    const double[:] p, const double[:] q, double[:] r
) noexcept:
    cdef double[:] cross = cross3(p[:3], q[:3])

    r[0] = p[3]*q[0] + q[3]*p[0] + cross[0]
    r[1] = p[3]*q[1] + q[3]*p[1] + cross[1]
    r[2] = p[3]*q[2] + q[3]*p[2] + cross[2]
    r[3] = p[3]*q[3] - p[0]*q[0] - p[1]*q[1] - p[2]*q[2]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double[:, :] compose_quat(
    const double[:, :] p, const double[:, :] q
) noexcept:
    cdef Py_ssize_t n = max(p.shape[0], q.shape[0])
    cdef double[:, :] product = empty2(n, 4)

    # dealing with broadcasting
    if p.shape[0] == 1:
        for ind in range(n):
            compose_quat_single(p[0], q[ind], product[ind])
    elif q.shape[0] == 1:
        for ind in range(n):
            compose_quat_single(p[ind], q[0], product[ind])
    else:
        for ind in range(n):
            compose_quat_single(p[ind], q[ind], product[ind])

    return product
