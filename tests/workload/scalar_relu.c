typedef unsigned int u32;
typedef int s32;

static s32 INPUT[4] = {-3, 0, 5, -1};
static volatile u32* const SCRATCHPAD = (volatile u32*)0x20000000u;

static void halt_with_code(u32 code) {
    __asm__ volatile("mv a0, %0\n ebreak" : : "r"(code) : "a0");
    for (;;) {
    }
}

void _start_c(void) {
    u32 index = 0;
    for (; index < 4u; index += 1u) {
        s32 value = INPUT[index];
        if (value < 0) {
            SCRATCHPAD[index] = 0u;
        } else {
            SCRATCHPAD[index] = (u32)value;
        }
    }
    halt_with_code(SCRATCHPAD[3]);
}

void _start(void) __attribute__((naked));

void _start(void) {
    __asm__ volatile("lui sp, 0x80\naddi sp, sp, -16\njal zero, _start_c");
}
