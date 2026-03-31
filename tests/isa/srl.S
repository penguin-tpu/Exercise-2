.globl _start
_start:
  addi t0, x0, 64
  addi t1, x0, 3
  srl a0, t0, t1
  ebreak
