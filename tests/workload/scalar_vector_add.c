typedef unsigned int u32;

static u32 A[4] = {1u, 2u, 3u, 4u};
static u32 B[4] = {10u, 20u, 30u, 40u};
static volatile u32* const SCRATCHPAD = (volatile u32*)0x20000000u;

static void halt_with_code(u32 code) {
    __asm__ volatile("mv a0, %0\n ebreak" : : "r"(code) : "a0");
    for (;;) {
    }
}

void _start_c(void) {
    u32 index = 0;
    for (; index < 4u; index += 1u) {
        SCRATCHPAD[index] = A[index] + B[index];
    }
    halt_with_code(SCRATCHPAD[3]);
}

void _start(void) __attribute__((naked));

void _start(void) {
    __asm__ volatile("lui sp, 0x80\naddi sp, sp, -16\njal zero, _start_c");
}
