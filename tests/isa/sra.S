.globl _start
_start:
  addi t0, x0, -64
  addi t1, x0, 3
  sra t2, t0, t1
  addi a0, t2, 16
  ebreak
