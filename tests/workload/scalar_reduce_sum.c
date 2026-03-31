typedef unsigned int u32;

static u32 INPUT[4] = {2u, 4u, 6u, 8u};
static volatile u32* const SCRATCHPAD = (volatile u32*)0x20000000u;

static void halt_with_code(u32 code) {
    __asm__ volatile("mv a0, %0\n ebreak" : : "r"(code) : "a0");
    for (;;) {
    }
}

void _start_c(void) {
    u32 index = 0;
    u32 sum = 0;
    for (; index < 4u; index += 1u) {
        sum += INPUT[index];
    }
    SCRATCHPAD[0] = sum;
    halt_with_code(sum);
}

void _start(void) __attribute__((naked));

void _start(void) {
    __asm__ volatile("lui sp, 0x80\naddi sp, sp, -16\njal zero, _start_c");
}
