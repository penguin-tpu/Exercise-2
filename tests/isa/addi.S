.globl _start
_start:
  addi t0, x0, 9
  addi t1, x0, 4
  slti t2, t1, 5
  sltiu t3, t1, 9
  add a0, t0, t1
  add a0, a0, t2
  add a0, a0, t3
  ebreak
