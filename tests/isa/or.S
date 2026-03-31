.globl _start
_start:
  addi t0, x0, 0x12
  addi t1, x0, 0x21
  or a0, t0, t1
  ebreak
