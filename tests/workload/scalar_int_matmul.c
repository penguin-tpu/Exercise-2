typedef unsigned int u32;

static u32 A[4] = {1u, 2u, 3u, 4u};
static u32 B[4] = {5u, 6u, 7u, 8u};
static volatile u32* const SCRATCHPAD = (volatile u32*)0x20000000u;

static void halt_with_code(u32 code) {
    __asm__ volatile("mv a0, %0\n ebreak" : : "r"(code) : "a0");
    for (;;) {
    }
}

static u32 mul_pos(u32 lhs, u32 rhs) {
    u32 accum = 0;
    while (rhs != 0u) {
        accum += lhs;
        rhs -= 1u;
    }
    return accum;
}

void _start_c(void) {
    SCRATCHPAD[0] = mul_pos(A[0], B[0]) + mul_pos(A[1], B[2]);
    SCRATCHPAD[1] = mul_pos(A[0], B[1]) + mul_pos(A[1], B[3]);
    SCRATCHPAD[2] = mul_pos(A[2], B[0]) + mul_pos(A[3], B[2]);
    SCRATCHPAD[3] = mul_pos(A[2], B[1]) + mul_pos(A[3], B[3]);
    halt_with_code(SCRATCHPAD[3]);
}

void _start(void) __attribute__((naked));

void _start(void) {
    __asm__ volatile("lui sp, 0x80\naddi sp, sp, -16\njal zero, _start_c");
}
