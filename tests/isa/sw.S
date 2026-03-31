.globl _start
_start:
  lui t0, 0x20000
  addi t1, x0, 52
  slli t1, t1, 4
  addi t1, t1, 20
  sw t1, 0(t0)
  lw a0, 0(t0)
  ebreak
