.globl _start
_start:
  addi t0, x0, -16
  srai t1, t0, 2
  addi a0, t1, 12
  ebreak
