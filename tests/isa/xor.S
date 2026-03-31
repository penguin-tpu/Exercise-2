.globl _start
_start:
  addi t0, x0, 12
  addi t1, x0, 9
  xor a0, t0, t1
  ebreak
