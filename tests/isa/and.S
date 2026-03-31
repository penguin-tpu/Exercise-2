.globl _start
_start:
  addi t0, x0, 0x3c
  addi t1, x0, 0x2a
  and a0, t0, t1
  ebreak
