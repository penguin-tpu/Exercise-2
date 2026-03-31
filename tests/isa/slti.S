.globl _start
_start:
  addi t0, x0, 4
  slti t1, t0, 5
  slti t2, t0, 4
  add a0, t1, t2
  ebreak
