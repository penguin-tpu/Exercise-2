.globl _start
_start:
  addi t0, x0, -1
  addi t1, x0, 1
  sltu a0, t1, t0
  ebreak
