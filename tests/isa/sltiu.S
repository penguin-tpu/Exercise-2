.globl _start
_start:
  addi t0, x0, -1
  sltiu t1, t0, 1
  sltiu t2, x0, 1
  add a0, t1, t2
  ebreak
