.globl _start
_start:
  addi t0, x0, -1
  addi t1, x0, 1
  slt t2, t0, t1
  sltu t3, t0, t1
  add a0, t2, t3
  ebreak
