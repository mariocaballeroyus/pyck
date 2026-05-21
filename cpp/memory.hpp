#ifndef PYCK_MEMORY_HPP
#define PYCK_MEMORY_HPP

#if defined(_MSC_VER)
    #include <malloc.h>
    #define PYCK_ALLOCA(size) _alloca(size)
#else
    #include <alloca.h>
    #define PYCK_ALLOCA(size) alloca(size)
#endif

/**
 * @brief Allocate a small, dynamically sized array on the stack. 
 *
 * @warning Only use if the size is guaranteed not to be more than a few hundred 
 *          bytes! Overflow occurs without any warning.
 */
#define STACK_ARRAY(T, name, sz) \
    T* name = static_cast<T*>(PYCK_ALLOCA((sz) * sizeof(T)))

#endif // PYCK_MEMORY_HPP
