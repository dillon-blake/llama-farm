// Probe symbols: the two entry points that exist before the shim does anything useful.
//
// They are not filler. ll_probe() is the anchor of the S0-04 version lock, which is the
// only thing standing between a vendor bump and silent memory corruption in the ctypes
// struct mirrors.

#include "farm_api.h"
#include "farm_version.h"

const char * ll_version(void) {
    return LL_VERSION;
}

const char * ll_probe(void) {
    return LL_LLAMA_CPP_COMMIT;
}
