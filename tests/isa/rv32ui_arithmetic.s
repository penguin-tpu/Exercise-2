.globl _start
_start:
  addi t0, x0, 9
  addi t1, x0, 4
  add t2, t0, t1
  sub t3, t2, t1
  slti t4, t1, 5
  sltiu t5, t3, 9
  add a0, t2, t3
  add a0, a0, t4
  add a0, a0, t5
  ebreak
